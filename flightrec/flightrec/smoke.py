# File: smoke.py
# Location: flightrec/smoke.py
# Purpose: Pre-flight bottleneck scan — record a SHORT slice of a heavy job, verdict + ETA before committing.
# Dependencies: flightrec.recorder, flightrec.measure, flightrec.report, flightrec.validate

"""Pre-flight bottleneck scan (the ``flightrec smoke`` arm).

Heavy jobs (multi-hour quants, training, long benches) are committed blind, then a
config mistake is discovered hours in. ``smoke`` wraps a SHORT representative slice
of the job (the first K per-unit markers) in ``FlightRecorder``, kills it, then maps
the utilization vector onto a one-screen card:

- **bottleneck verdict** — GPU-starved (low power/clock under load → data-movement/CPU-
  bound), GPU-busy (high SM duty-cycle — work IS reaching the GPU, but duty != FLOP
  utilization, so this is NOT a confirmed compute-bound call), near-OOM (free unified
  memory near the ceiling → OOM-reset risk, a CAPACITY signal not a bandwidth one), or
  throttled (auto-invalidate — same wedge detector as ``validate``).
- **config recommendations** — concrete knobs keyed to the verdict (e.g. GPU-starved →
  ``low_gpu_mem_usage=False``; mem headroom → raise ``batch_size``).
- **full-run ETA** — per-unit wall-time (median of steady-state marker intervals) ×
  total units.

The slice is producer-side "measure before you commit": the same recorder + roofline +
validity machinery applied to a job's warm-up instead of a finished artifact. The
heuristic layer here is the one place flightrec draws a (clearly-labelled) conclusion;
the underlying numbers are still emitted verbatim for the reader to check.
"""

import re
import signal
import subprocess
import sys
import time

import pandas as pd

from flightrec.recorder import FlightRecorder
from flightrec.measure import measure_artifact
from flightrec.report import load_run, in_phases
from flightrec.validate import LOAD_PCT

_RAW = time.CLOCK_MONOTONIC_RAW

# Heuristic thresholds (GB10-specific; see PROTOCOLS §smoke).
STARVE_POWER_W = 25.0  # under load but <25 W => data-movement/CPU-bound (the 15 W AutoRound case)
MEM_HEADROOM_FLOOR_MB = 8192  # <8 GB unified-mem free at peak => near the ceiling (OOM-reset risk)
COMPUTE_BUSY_PCT = 80.0  # GPU duty-cycle that reads as healthily compute-bound
UPSIZE_HEADROOM_MB = 51200  # >50 GB free => room to raise batch_size
IDLE_SLICE_FRAC = 0.5  # <50% of slice under load => marker likely fires before real work


def smoke(
    cmd,
    marker_re,
    out_dir,
    units=3,
    total_units=None,
    settle=1,
    hz=20,
    grace=5.0,
    bytes_moved=None,
    flops=0,
    tokens_per_unit=None,
):
    """Record a K-unit slice of *cmd*, then return a pre-flight bottleneck card.

    Args:
        cmd: Command sequence (the heavy job), run under the recorder and killed early.
        marker_re: Regex; each matching stdout line = one completed unit (layer/step/iter).
        out_dir: Artifact directory for the recorded slice.
        units: How many marker units to observe before killing (default 3).
        total_units: Total units in the full run; enables ETA when given.
        settle: Leading marker intervals discarded as warm-up before timing (default 1).
        hz: FlightRecorder sampling frequency.
        grace: Seconds to wait after SIGTERM before SIGKILL.
        bytes_moved, flops: Optional analytic inputs for the roofline point.
        tokens_per_unit: Optional tokens produced per unit, enabling tok/s + J/token.

    Returns:
        The card dict — verdict, recommendations, signals, eta, vector, and slice
        provenance. Pass it to ``format_card`` for the one-screen view.
    """
    slice_result = _record_slice(cmd, marker_re, out_dir, units, hz, grace)
    samples, phases, _ = load_run(out_dir)
    tokens = _tokens(tokens_per_unit, slice_result["reached_units"])
    vector = measure_artifact(out_dir, bytes_moved=bytes_moved, flops=flops, tokens=tokens)
    sig = signals(samples, phases)
    verdict, recs = diagnose(vector, sig)
    return {
        "verdict": verdict,
        "recommendations": recs,
        "signals": sig,
        "eta": eta(per_unit_seconds(slice_result["mark_times"], settle), total_units),
        "vector": vector,
        "slice": _slice_provenance(slice_result, units),
        "artifact_path": str(out_dir),
    }


def _record_slice(cmd, marker_re, out_dir, units, hz, grace):
    """Run the slice inside a recorder phase so validity + energy are captured."""
    with FlightRecorder(out_dir, hz=hz) as rec:
        with rec.phase("smoke"):
            return run_until_marks(cmd, marker_re, units, grace)


def run_until_marks(cmd, pattern, k_units, grace=5.0):
    """Run *cmd*, echo stdout live, and kill it once *k_units* markers have fired.

    Each line matching *pattern* counts as one completed unit; its CLOCK_MONOTONIC_RAW
    timestamp is recorded so steady-state per-unit time can be derived. If the process
    exits on its own before *k_units* (a genuinely short job), the marks collected so
    far are returned with ``killed=False``.

    Returns:
        Dict with ``mark_times`` (monotonic seconds), ``reached_units``, ``killed``
        (whether we cut it short), and ``exit_code``.
    """
    compiled = re.compile(pattern)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    mark_times = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if compiled.search(line):
            mark_times.append(time.clock_gettime(_RAW))
            if len(mark_times) >= k_units:
                break
    killed = len(mark_times) >= k_units
    code = _terminate(proc, grace) if killed else proc.wait()
    return {
        "mark_times": mark_times,
        "reached_units": len(mark_times),
        "killed": killed,
        "exit_code": code,
    }


def _terminate(proc, grace):
    """SIGTERM the slice, escalate to SIGKILL after *grace* seconds; return exit code."""
    proc.send_signal(signal.SIGTERM)
    try:
        return proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait()


def per_unit_seconds(mark_times, settle=1):
    """Median steady-state seconds per unit, dropping the first *settle* intervals.

    Intervals are gaps between consecutive markers; the warm-up gaps are discarded so
    the ETA reflects the steady cadence, not first-unit JIT/allocator cost. Returns
    None when fewer than two markers were captured.
    """
    if len(mark_times) < 2:
        return None
    intervals = [b - a for a, b in zip(mark_times, mark_times[1:])]
    steady = intervals[settle:] or intervals
    return _median(steady)


def eta(per_unit_s, total_units):
    """Full-run ETA from per-unit time × total units, or Nones when inputs absent."""
    if per_unit_s is None or not total_units:
        return {
            "per_unit_s": per_unit_s,
            "total_units": total_units,
            "eta_s": None,
            "eta_human": None,
        }
    secs = per_unit_s * total_units
    return {
        "per_unit_s": round(per_unit_s, 4),
        "total_units": total_units,
        "eta_s": round(secs, 1),
        "eta_human": _human_time(secs),
    }


def signals(samples, phases):
    """Load-scoped utilization signals the heuristic reads (power/clock/duty/mem).

    Scoped to the marked phase, then to loaded samples (GPU duty >= LOAD_PCT) so a
    benign idle window between units does not drag the medians down.
    """
    active = in_phases(samples, phases)
    scoped = active if len(active) else samples
    busy = _busy(scoped)
    loaded = scoped[busy >= LOAD_PCT]
    frame = loaded if len(loaded) else scoped
    return {
        "power_median_w": round(float(frame["power_w"].median()), 1),
        "sm_clock_median_mhz": int(frame["sm_clock_mhz"].median()),
        "gpu_busy_median_pct": round(float(busy.median()), 1),
        "mem_avail_min_mb": _min_col(scoped, "mem_avail_mb"),
        "loaded_frac": round(len(loaded) / max(len(scoped), 1), 2),
    }


def diagnose(vector, sig):
    """Map (vector, signals) -> (verdict, recommendations). First matching rule wins.

    Throttle is checked first (it invalidates everything else), then idle-slice, then
    the bottleneck rules. A memory-headroom upsize hint is appended when the box has
    plenty of free unified memory and is not already memory-bound or throttled.
    """
    for rule in _RULES:
        if rule[1](vector, sig):
            verdict, recs = rule[0], rule[2](sig)
            return verdict, recs + _headroom_hint(verdict, sig)
    return "INTERMEDIATE", _recs_intermediate(sig)


# --- heuristic rules: (verdict, predicate, recommendation-builder) ---------


def _is_throttled(vector, sig):
    return vector.get("valid") is False and sig["loaded_frac"] > 0


def _is_idle_slice(_vector, sig):
    return sig["loaded_frac"] < IDLE_SLICE_FRAC


def _is_starved(_vector, sig):
    return sig["power_median_w"] < STARVE_POWER_W


def _is_near_oom(_vector, sig):
    free = sig["mem_avail_min_mb"]
    return free is not None and free < MEM_HEADROOM_FLOOR_MB


def _is_gpu_busy(_vector, sig):
    """High GPU duty-cycle. NOTE: duty-cycle, not FLOP utilization — a launch/
    latency-bound job (many tiny kernels) also reads ~100% busy, so this is a
    GPU-BUSY signal, never a confirmed compute-bound verdict."""
    return sig["gpu_busy_median_pct"] >= COMPUTE_BUSY_PCT


def _recs_throttled(sig):
    return [
        f"Clock/power collapsed under load ({sig['sm_clock_median_mhz']} MHz / "
        f"{sig['power_median_w']} W) — measurement INVALID, not a slow config.",
        "Cold-drain the PD/USB-C power-path wedge (AC sag / under-power clamp), then re-run smoke.",
        "Do NOT commit the full job from this state.",
    ]


def _recs_idle(sig):
    return [
        f"GPU under load only {int(sig['loaded_frac'] * 100)}% of the slice — verdict unreliable.",
        "The marker likely fires before real GPU work; widen --units or fix --marker-re.",
        "Re-run smoke before trusting the bottleneck call.",
    ]


def _recs_starved(sig):
    return [
        f"GPU under load but drawing {sig['power_median_w']} W / "
        f"{sig['sm_clock_median_mhz']} MHz — data-movement/CPU-bound, not compute.",
        "If this is AutoRound/GPTQ quant: set low_gpu_mem_usage=False.",
        "Check host->device transfer, dataloader workers, and CPU-side preprocessing.",
        "Full run will idle most of the GPU — fix before committing.",
    ]


def _recs_mem(sig):
    return [
        f"Only {sig['mem_avail_min_mb']} MB unified memory free at peak — near the GB10 ceiling.",
        "Lower batch_size / max_seq_len / KV-cache, or enable CPU offload.",
        "Risk of an OOM-reset partway through the full run (see flightrec-memguard).",
    ]


def _recs_gpu_busy(sig):
    return [
        f"GPU duty-cycle high ({sig['gpu_busy_median_pct']}% busy, "
        f"{sig['power_median_w']} W, {sig['sm_clock_median_mhz']} MHz): work is "
        "reaching the GPU and the config is safe to commit on duty grounds.",
        "NOT a confirmed compute-bound call — duty-cycle reads ~100% for a launch/"
        "latency-bound job too. To confirm compute-bound, compare achieved GFLOP/s "
        "to the measured FP4/FP8 peak (see roofline_predict / a compute-ceiling calibration).",
    ]


def _recs_intermediate(sig):
    base = [
        f"Mixed utilization ({sig['gpu_busy_median_pct']}% busy, {sig['power_median_w']} W) — "
        "no single dominant bottleneck in the slice.",
        "Safe to proceed, but monitor the first units of the real run.",
    ]
    return base + _headroom_hint("INTERMEDIATE", sig)


_RULES = (
    ("THROTTLED", _is_throttled, _recs_throttled),
    ("INCONCLUSIVE", _is_idle_slice, _recs_idle),
    ("GPU-STARVED", _is_starved, _recs_starved),
    ("NEAR-OOM", _is_near_oom, _recs_mem),
    ("GPU-BUSY", _is_gpu_busy, _recs_gpu_busy),
)


def _headroom_hint(verdict, sig):
    """Suggest raising batch_size when utilization is healthy and unified mem is plentiful.

    Only the genuinely-utilizing verdicts get the hint: raising batch_size is wrong
    advice for a starved, idle, throttled, or already-memory-bound slice.
    """
    free = sig["mem_avail_min_mb"]
    if verdict not in ("GPU-BUSY", "INTERMEDIATE") or free is None or free < UPSIZE_HEADROOM_MB:
        return []
    return [f"Memory headroom ~{round(free / 1024)} GB free → room to raise batch_size."]


def format_card(card):
    """Render the pre-flight card as one screen of text."""
    lines = ["", "=" * 64, f"  flightrec smoke — PRE-FLIGHT: {card['verdict']}", "=" * 64]
    lines += _card_signals(card["signals"], card["slice"])
    lines += _card_eta(card["eta"])
    lines.append("  recommendations:")
    lines += [f"    - {rec}" for rec in card["recommendations"]]
    lines.append("=" * 64)
    return "\n".join(lines)


def _card_signals(sig, sl):
    mem = f"{sig['mem_avail_min_mb']} MB" if sig["mem_avail_min_mb"] is not None else "n/a"
    return [
        f"  slice: {sl['reached_units']}/{sl['requested_units']} units"
        f" ({'killed' if sl['killed'] else 'ran to exit'})",
        f"  GPU:   {sig['power_median_w']} W | {sig['sm_clock_median_mhz']} MHz | "
        f"{sig['gpu_busy_median_pct']}% busy | mem free {mem} | loaded {int(sig['loaded_frac'] * 100)}%",
    ]


def _card_eta(e):
    if e["eta_s"] is None:
        per = f"{e['per_unit_s']} s/unit" if e["per_unit_s"] is not None else "n/a"
        return [f"  pace:  {per} (pass --total for full-run ETA)"]
    return [
        f"  pace:  {e['per_unit_s']} s/unit × {e['total_units']} units " f"→ ETA {e['eta_human']}"
    ]


def _slice_provenance(slice_result, requested_units):
    return {
        "requested_units": requested_units,
        "reached_units": slice_result["reached_units"],
        "killed": slice_result["killed"],
        "exit_code": slice_result["exit_code"],
    }


def _tokens(tokens_per_unit, reached_units):
    if not tokens_per_unit or not reached_units:
        return None
    return tokens_per_unit * reached_units


def _busy(scoped):
    if "gpu_busy_pct" not in scoped.columns:
        return pd.Series(0.0, index=scoped.index)
    return scoped["gpu_busy_pct"].ffill().fillna(0.0)


def _min_col(frame, col):
    if col not in frame.columns or frame[col].dropna().empty:
        return None
    return round(float(frame[col].dropna().min()), 1)


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _human_time(seconds):
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
