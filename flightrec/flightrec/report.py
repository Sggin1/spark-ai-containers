# File: report.py
# Location: flightrec/report.py
# Purpose: Load a recorder artifact and print provenance + validity + rollup.
# Dependencies: pandas, flightrec.validate

"""Human-readable summary of one flight-recorder artifact."""

import json
from pathlib import Path

import pandas as pd

from flightrec.validate import verdict
from flightrec.phase_tree import build_tree, format_tree


def load_run(run_dir):
    """Load an artifact dir -> (samples_df, phases_df, header_dict)."""
    base = Path(run_dir)
    samples = pd.read_parquet(base / "samples.parquet")
    phases = pd.read_parquet(base / "phases.parquet")
    header = json.loads((base / "header.json").read_text(encoding="utf-8"))
    return samples, phases, header


def in_phases(samples, phases):
    """Samples inside any marked phase window (the active benchmark span)."""
    if phases.empty or "t0_ns" not in phases.columns:
        return samples
    mask = pd.Series(False, index=samples.index)
    for _, row in phases.iterrows():
        mask |= samples["t_ns"].between(row["t0_ns"], row["t1_ns"])
    return samples[mask]


def summarize_run(run_dir):
    """Print header + validity verdict + energy/clock rollup; return the verdict."""
    samples, phases, header = load_run(run_dir)
    active = in_phases(samples, phases)
    scoped = active if len(active) else samples
    result = verdict(scoped)
    print(_fmt_header(header))
    print(_fmt_verdict(result, scoped=len(active) > 0))
    print(_fmt_rollup(scoped))
    tree = build_tree(phases, samples)
    if tree:
        print("\nphase tree (duration · energy · mean clock):\n" + format_tree(tree))
    return result


def _fmt_header(h):
    lines = [f"\n=== {h.get('gpu_name')}  driver {h.get('driver')} ===",
             f"SM clock ceiling {h.get('sm_clock_max_mhz')} MHz | "
             f"BW wall {h.get('bandwidth_wall_gbps')} GB/s "
             f"[{h.get('bandwidth_wall_source')}]"]
    peaks = h.get("compute_peak_tflops")
    if peaks:
        lines.append("compute peak " + ", ".join(f"{dt}={tf} TF" for dt, tf in peaks.items()))
    q = h.get("quiescence")
    if q:
        lines.append(f"pre-flight: {q['verdict']} ({'; '.join(q['reasons'])})")
    lines.append(f"caveat: {h.get('caveats')}")
    return "\n".join(lines)


def _fmt_verdict(v, scoped):
    tag = _verdict_tag(v)
    where = "in-phase" if scoped else "whole-run (no phases marked)"
    return (f"\nverdict [{where}]: {tag}\n"
            f"  samples={v.get('samples')} loaded={v.get('load_pct_of_window')}% | "
            f"throttled-of-loaded={v.get('throttled_pct_of_loaded')}% | "
            f"clock min/med={v.get('sm_clock_min_mhz')}/{v.get('sm_clock_median_mhz')} MHz | "
            f"power {v.get('power_min_w')}-{v.get('power_max_w')} W")


def _verdict_tag(v):
    if not v.get("loaded_samples"):
        return "VALID (GPU idle in window — nothing to judge)"
    return "VALID (under load, no throttle)" if v.get("valid") else "INVALID (throttled under load)"


def _fmt_rollup(samples):
    span = samples["t_rel_s"]
    dur = float(span.max() - span.min())
    joules = float(samples["energy_mj"].max() - samples["energy_mj"].min()) / 1000.0
    cores = int(samples["n_active"].max()) if "n_active" in samples.columns else 0
    return (f"\nrollup: {dur:.1f}s | mean power {samples['power_w'].mean():.1f} W | "
            f"GPU energy {joules:.1f} J | peak active cores {cores}/20")
