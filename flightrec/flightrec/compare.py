# File: compare.py
# Location: flightrec/compare.py
# Purpose: Direct hardware comparison of two recorder artifacts (e.g. two DGX Spark boxes).
# Dependencies: flightrec.report, flightrec.validate, flightrec.stats

"""Head-to-head hardware comparison from flight-recorder artifacts.

Same benchmark, two boxes (or two configs): compare not just who is faster but
WHY — power headroom, clock, throttle, energy, thermals — plus a provenance diff
(driver / clock ceiling / modelled wall) so a hardware or firmware delta is
explicit, not guessed. Artifacts are portable parquet+json, so a peer-box run copied
over the network compares directly against a local run.
"""

from flightrec.report import load_run, in_phases
from flightrec.validate import verdict
from flightrec.stats import compare
from flightrec.aggregate import valid_metric

_METRICS = [
    ("power_w", "power W (med/p95)", "med_p95"),
    ("sm_clock_mhz", "SM clock MHz (med/min)", "med_min"),
    ("temp_c", "temp C (max)", "max"),
    ("n_active", "active cores (peak)", "max"),
    ("cpu_freq_mean_mhz", "CPU freq MHz (med)", "med"),
]
_PROV = ["gpu_name", "driver", "sm_clock_max_mhz", "bandwidth_wall_gbps",
         "bandwidth_wall_source"]


def envelope(run_dir):
    """Hardware-state envelope + validity verdict + provenance for one artifact."""
    samples, phases, header = load_run(run_dir)
    scoped = in_phases(samples, phases)
    active = scoped if len(scoped) else samples
    return {
        "header": header,
        "verdict": verdict(active),
        "energy_j": _energy(active),
        "stats": {key: _stat(active, key, kind) for key, _, kind in _METRICS},
    }


def compare_runs(dir_a, dir_b):
    """Print a side-by-side hardware comparison of two artifacts; return both."""
    env_a, env_b = envelope(dir_a), envelope(dir_b)
    print(_fmt_table(env_a, env_b))
    print(_fmt_provenance(env_a["header"], env_b["header"]))
    return env_a, env_b


def compare_distributions(runs_a, runs_b, phase=None):
    """Bootstrap A/B on phase durations across N runs per side (the run-time fact)."""
    side_a = [d for d in (phase_seconds(r, phase) for r in runs_a) if d]
    side_b = [d for d in (phase_seconds(r, phase) for r in runs_b) if d]
    return compare(side_a, side_b)


def replication_verdict(values_a, values_b, min_effect_pct=1.0):
    """Do two boxes agree on the same config? (PROTOCOLS §7 — replication = rigor.)

    REPLICATED when the cross-box median difference is within noise — i.e. NOT both
    statistically significant AND practically large. A box-specific fluke shows up as
    a significant + practical gap and fails replication.
    """
    ab = compare(values_a, values_b, min_effect_pct=min_effect_pct)
    return {
        "replicated": not ab["practical"],
        "rel_pct": ab["rel_pct"],
        "ci95": ab["ci95"],
        "significant": ab["significant"],
        "n_a": len(values_a),
        "n_b": len(values_b),
    }


def replication_over_runs(runs_a, runs_b, metric="kernel_s", min_effect_pct=1.0,
                          bytes_moved=None, flops=0, tokens=None):
    """Replication verdict over artifact dirs from two boxes (INVALID runs dropped)."""
    vals_a, _ = valid_metric(runs_a, metric, bytes_moved, flops, tokens)
    vals_b, _ = valid_metric(runs_b, metric, bytes_moved, flops, tokens)
    if not vals_a or not vals_b:
        return {"replicated": False, "error": "a box has no valid runs for this metric",
                "n_a": len(vals_a), "n_b": len(vals_b)}
    return replication_verdict(vals_a, vals_b, min_effect_pct=min_effect_pct)


def phase_seconds(run_dir, phase=None):
    """Total seconds spent in `phase` (or all phases) for one artifact."""
    _, phases, _ = load_run(run_dir)
    if phases.empty or "t0_ns" not in phases.columns:
        return None
    rows = phases if phase is None else phases[phases["phase"] == phase]
    return float((rows["t1_ns"] - rows["t0_ns"]).sum()) / 1e9 if len(rows) else None


def _energy(samples):
    if "energy_mj" not in samples.columns or samples["energy_mj"].dropna().empty:
        return None
    energy = samples["energy_mj"].dropna()
    return round(float(energy.max() - energy.min()) / 1000.0, 2)


def _stat(samples, key, kind):
    if key not in samples.columns or samples[key].dropna().empty:
        return None
    series = samples[key].dropna()
    if kind == "med_p95":
        return (round(float(series.median()), 1), round(float(series.quantile(0.95)), 1))
    if kind == "med_min":
        return (int(series.median()), int(series.min()))
    if kind == "max":
        return round(float(series.max()), 1)
    return round(float(series.median()), 1)


def _fmt_table(env_a, env_b):
    lines = ["\n=== hardware compare  [A] vs [B] ==="]
    lines.append(_row("valid", env_a["verdict"].get("valid"), env_b["verdict"].get("valid")))
    lines.append(_row("throttled% of loaded",
                      env_a["verdict"].get("throttled_pct_of_loaded"),
                      env_b["verdict"].get("throttled_pct_of_loaded")))
    lines.append(_row("energy J", env_a["energy_j"], env_b["energy_j"]))
    for key, label, _ in _METRICS:
        lines.append(_row(label, env_a["stats"].get(key), env_b["stats"].get(key)))
    return "\n".join(lines)


def _row(label, value_a, value_b):
    return f"  {label:<24} A={value_a!s:<18} B={value_b!s}"


def _fmt_provenance(header_a, header_b):
    lines = ["\n--- provenance diff ---"]
    for key in _PROV:
        value_a, value_b = header_a.get(key), header_b.get(key)
        flag = "" if value_a == value_b else "   <-- DIFFERS"
        lines.append(f"  {key:<28} A={value_a!s:<16} B={value_b!s}{flag}")
    return "\n".join(lines)
