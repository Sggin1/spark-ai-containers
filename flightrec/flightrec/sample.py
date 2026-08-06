# File: sample.py
# Location: flightrec/sample.py
# Purpose: Adaptive-N sampler: keep running until CI half-width < X% of median.
# Dependencies: flightrec.recorder, flightrec.aggregate, subprocess

"""Adaptive-N sampling: run a command repeatedly until the bootstrap CI on the median
tightens to a user-supplied threshold.

Instead of requiring the user to pick N up front, ``adaptive_sample`` runs the bench,
calls ``aggregate`` after each batch of runs (starting at ``min_n``), and stops as soon
as the 95% CI half-width falls below ``until_ci_pct``% of the median — or when
``max_n`` runs have been completed.  This is the machinery behind ``flightrec sample``.
"""

import subprocess

from flightrec.recorder import FlightRecorder
from flightrec.aggregate import aggregate


def adaptive_sample(
    cmd,
    out_prefix,
    metric,
    until_ci_pct,
    min_n=5,
    max_n=50,
    bytes_moved=None,
    flops=0,
    tokens=None,
    hz=20,
):
    """Run *cmd* repeatedly until CI half-width < *until_ci_pct*% of median.

    Args:
        cmd: Sequence of strings passed to ``subprocess.call`` (same as ``record``).
        out_prefix: Directory prefix; each run is stored as ``{out_prefix}_{nnn}``.
        metric: Utilization-vector field to converge on (e.g. ``kernel_s``, ``tok_s``).
        until_ci_pct: Stop when bootstrap-95% CI half-width / median < this percentage.
        min_n: Minimum runs before convergence is checked.
        max_n: Hard cap — stop even if CI is still wide.
        bytes_moved, flops, tokens: Passed through to ``aggregate``/``measure_artifact``.
        hz: Sampling frequency for ``FlightRecorder``.

    Returns:
        Dict with keys ``run_dirs``, ``n``, ``converged`` (bool), ``aggregate`` (last
        aggregate summary).
    """
    run_dirs = []
    last_agg = None
    ci_met = False

    while len(run_dirs) < max_n:
        n = len(run_dirs)
        out_dir = f"{out_prefix}_{n:03d}"
        with FlightRecorder(out_dir, hz=hz):
            subprocess.call(cmd)
        run_dirs.append(out_dir)

        if len(run_dirs) >= min_n:
            last_agg = aggregate(run_dirs, metric=metric, bytes_moved=bytes_moved,
                                 flops=flops, tokens=tokens)
            ci = last_agg.get("median_ci95")
            median = last_agg.get("median")
            if ci is not None and median:
                ci_halfwidth_pct = 100.0 * (ci[1] - ci[0]) / 2.0 / median
                print(f"[flightrec] n={len(run_dirs)} median={median:.4f} CI±{ci_halfwidth_pct:.2f}%")
                if ci_halfwidth_pct <= until_ci_pct:
                    ci_met = True
                    break
            else:
                print(f"[flightrec] n={len(run_dirs)} metric '{metric}' not yet available")

    return {
        "run_dirs": run_dirs,
        "n": len(run_dirs),
        "converged": bool(ci_met),
        "aggregate": last_agg,
    }
