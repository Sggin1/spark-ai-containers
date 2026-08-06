# File: measure.py
# Location: flightrec/measure.py
# Purpose: One-call utilization-vector extractor from a recorded artifact (the measure arm).
# Dependencies: flightrec.report, flightrec.validate, flightrec.roofline

"""Turn a recorded artifact into the run's utilization vector — validity-gated.

This is the producer-side "measure" contract for any consumer (the optimize board
work, the cuda-specialist discovery loop): wrap a bench in ``FlightRecorder``, then
call ``measure_artifact`` to get ONE dict with —

- the REAL validity verdict (from the samples via ``validate.verdict`` — NOT the
  static clock ceiling in the header, which never moves and so never flags a wedge),
- kernel time from the marked phase (not wall-clock around the whole context),
- achieved GB/s + %-of-wall + arithmetic intensity (``roofline.roofline_point``
  against the MEASURED wall in the header),
- GPU energy (J), and tok/s + J/token when a token count is supplied.

No conclusion is drawn (tool != analyst): it PROVIDES the numbers. A field is None
when its inputs are absent. Never report a metric from a run whose ``valid`` is False.
"""

from flightrec.report import load_run, in_phases
from flightrec.validate import verdict, LOAD_PCT
from flightrec.roofline import roofline_point, MODELED_WALL_GBPS


def measure_artifact(run_dir, bytes_moved=None, flops=0, tokens=None, peak_tflops=None):
    """Extract the validity-gated utilization vector from a recorder artifact.

    Args:
        run_dir: Path to a flightrec artifact directory.
        bytes_moved: Analytic bytes moved by the kernel (enables %-of-wall + AI).
        flops: Floating-point ops, for arithmetic intensity (default 0).
        tokens: Tokens produced (enables tok/s and J/token).
        peak_tflops: Compute ceiling (TFLOP/s) for the kernel's dtype; enables
            pct_of_peak + the two-axis ``regime``. Falls back to the artifact
            header's ``compute_peak_tflops`` when not passed.

    Returns:
        The utilization vector dict — valid, kernel_s, energy_j, tok_s, j_token,
        achieved_gbps, pct_of_wall, pct_of_peak, arithmetic_intensity,
        bandwidth_regime, regime, plus wall provenance and throttle stats.
        Inputs-absent fields are None.
    """
    samples, phases, header = load_run(run_dir)
    active = _active_window(samples, phases)
    result = verdict(active)
    kernel_s = _kernel_seconds(phases, active)
    energy_j = _energy_j(active)
    wall = header.get("bandwidth_wall_gbps", MODELED_WALL_GBPS)
    peak = peak_tflops or header.get("compute_peak_tflops")
    header_tok_s = header.get("throughput", {}).get("tok_s")
    out = {
        "valid": result.get("valid"),
        "kernel_s": kernel_s,
        "energy_j": energy_j,
        "bandwidth_wall_gbps": wall,
        "bandwidth_wall_source": header.get("bandwidth_wall_source"),
        "throttled_pct_of_loaded": result.get("throttled_pct_of_loaded"),
        "sm_clock_median_mhz": result.get("sm_clock_median_mhz"),
        "samples": result.get("samples"),
        "artifact_path": str(run_dir),
    }
    out.update(_per_token(tokens, kernel_s, energy_j, header_tok_s))
    out.update(_roofline(bytes_moved, flops, kernel_s, wall, peak))
    out.update(power_energy_consistency(active))
    return out


def _active_window(samples, phases):
    """Samples inside the marked phase(s), or the whole run if none were marked."""
    scoped = in_phases(samples, phases)
    return scoped if len(scoped) else samples


def _per_token(tokens, kernel_s, energy_j, header_tok_s=None):
    """tok/s and J/token, each None unless its inputs are present.

    When *tokens* is absent but the header carries a parsed ``throughput.tok_s``
    value (written by ``benchy`` during ``record``), that value is used as the
    tok/s fallback.  J/token still requires an explicit *tokens* count.

    When *both* computed (tokens/kernel_s) and benchy-reported (header_tok_s) values
    are present, a cross-check is performed.  A divergence > 20% is flagged via
    ``tok_s_consistent=False`` — it usually means the kernel_s phase window is
    misaligned with what the bench actually timed (e.g. prefill included in kernel_s
    but excluded from the bench's wall-clock elapsed).
    """
    out = {}
    if tokens and kernel_s:
        computed = tokens / kernel_s
        out["tok_s"] = round(computed, 2)
        if header_tok_s is not None:
            divergence = abs(computed - header_tok_s) / max(computed, header_tok_s)
            out["tok_s_benchy"] = header_tok_s
            out["tok_s_divergence_pct"] = round(divergence * 100, 1)
            out["tok_s_consistent"] = divergence < 0.20
    elif header_tok_s is not None:
        out["tok_s"] = header_tok_s
    else:
        out["tok_s"] = None
    out["j_token"] = round(energy_j / tokens, 4) if tokens and energy_j else None
    return out


def _kernel_seconds(phases, active):
    """Kernel time: summed marked-phase span if present, else the sample span."""
    return _phase_span(phases) or _sample_span(active)


def _phase_span(phases):
    """Total seconds across marked phases, or None when unmarked/zero."""
    if phases.empty or "t0_ns" not in phases.columns:
        return None
    secs = round(float((phases["t1_ns"] - phases["t0_ns"]).sum()) / 1e9, 6)
    return secs or None


def _sample_span(active):
    """Wall span of the sample window in seconds, or None when empty."""
    if "t_rel_s" not in active.columns or not len(active):
        return None
    return round(float(active["t_rel_s"].max() - active["t_rel_s"].min()), 6)


def _energy_j(active):
    """GPU energy delta (J) over the window, from the monotonic mJ integral."""
    if "energy_mj" not in active.columns or active["energy_mj"].dropna().empty:
        return None
    energy = active["energy_mj"].dropna()
    return round(float(energy.max() - energy.min()) / 1000.0, 3)


CONSISTENCY_RTOL = 0.15  # energy-rate vs mean-power must agree within 15% under load


def power_energy_consistency(active):
    """Cross-check the two independent measurements of average power, under load.

    ``power_w`` is the fast-path (~20 Hz) instantaneous NVML reading; ``energy_mj``
    is the slow-path (~6 Hz) monotonic integral. Over the window, the time-average
    of instantaneous power MUST equal the integral's rate ``ΔE/Δt`` — they measure
    the same physical quantity. A divergence beyond ``CONSISTENCY_RTOL`` flags one
    of: an NVML energy-counter glitch, the GB10 instantaneous-power smoothing lag,
    or a phase-boundary artefact (energy captured outside the window power saw).

    The check is **scoped to loaded samples** (GPU duty >= ``LOAD_PCT``), like the
    smoke signals. This is deliberate: on GB10 the instantaneous-power reading has a
    ~1-2 W low-end floor, so at idle (a few W) a tiny absolute gap reads as a huge
    *relative* divergence — verified on real idle artifacts (3.5 W mean vs 5.7 W
    rate = 38%). The relative check is only physical under load; an idle/unloaded
    run yields ``power_energy_consistent=None`` (not measured), never a false alarm.

    ``power_energy_consistent`` is always present (None when uncomputable — no
    energy column, no loaded samples), so callers can rely on the key.
    """
    keys = (
        "power_energy_consistent",
        "power_energy_divergence_pct",
        "energy_rate_w",
        "mean_power_w",
    )
    loaded = _loaded(active)
    rate = _energy_rate_w(loaded)
    mean_power = _mean_power_w(loaded)
    if rate is None or mean_power is None:
        return dict.fromkeys(keys)
    divergence = abs(rate - mean_power) / max(rate, mean_power, 1e-9)
    return {
        "power_energy_consistent": bool(divergence < CONSISTENCY_RTOL),
        "power_energy_divergence_pct": round(divergence * 100, 1),
        "energy_rate_w": round(rate, 2),
        "mean_power_w": round(mean_power, 2),
    }


def _loaded(active):
    """Samples with GPU duty >= LOAD_PCT; the whole frame when duty is unrecorded.

    Older artifacts (and synthetic frames) may lack ``gpu_busy_pct`` — there we
    cannot scope, so we keep every sample rather than drop the check entirely.
    """
    if "gpu_busy_pct" not in active.columns:
        return active
    busy = active["gpu_busy_pct"].ffill().fillna(0.0)
    return active[busy >= LOAD_PCT]


def _energy_rate_w(frame):
    """ΔE/Δt in watts across the energy-sampled span, or None when uncomputable."""
    if "energy_mj" not in frame.columns or "t_rel_s" not in frame.columns:
        return None
    pairs = frame[["t_rel_s", "energy_mj"]].dropna()
    if len(pairs) < 2:
        return None
    dt = float(pairs["t_rel_s"].max() - pairs["t_rel_s"].min())
    delta_j = float(pairs["energy_mj"].max() - pairs["energy_mj"].min()) / 1000.0
    return delta_j / dt if dt > 0 else None


def _mean_power_w(frame):
    """Mean instantaneous power over the window, or None when absent."""
    if "power_w" not in frame.columns or frame["power_w"].dropna().empty:
        return None
    return float(frame["power_w"].dropna().mean())


def _roofline(bytes_moved, flops, kernel_s, wall, peak_tflops=None):
    """Roofline coords when we have bytes + a kernel time, else Nones.

    Emits both axes: ``achieved_gbps``/``pct_of_wall`` (bandwidth score) and
    ``achieved_gflops``/``pct_of_peak`` (compute score). With a compute peak the
    two-axis ``regime`` resolves compute-bound vs overhead-bound; without it,
    ``pct_of_peak`` and ``regime`` are None and only ``bandwidth_regime`` stands.
    """
    keys = (
        "achieved_gbps",
        "achieved_gflops",
        "pct_of_wall",
        "pct_of_peak",
        "above_wall",
        "arithmetic_intensity",
        "bandwidth_regime",
        "regime",
    )
    if not bytes_moved or not kernel_s:
        return dict.fromkeys(keys)
    point = roofline_point(bytes_moved, flops or 0, kernel_s, wall, peak_tflops)
    return {key: point.get(key) for key in keys}
