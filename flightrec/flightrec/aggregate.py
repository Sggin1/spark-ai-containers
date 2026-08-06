# File: aggregate.py
# Location: flightrec/aggregate.py
# Purpose: N-run distribution over artifacts (median/IQR/CV + bootstrap CI), INVALID dropped.
# Dependencies: flightrec.measure, flightrec.stats

"""Aggregate one metric across N recorded runs of the same config — a distribution, not an anecdote.

Per PROTOCOLS §1/§3: report median + IQR (run times are right-skewed), CV% as a jitter diagnostic, and
a bootstrap CI on the median — and **never include an INVALID (throttled) run**. Each run dir is scored
with ``measure_artifact`` (so validity, kernel_s, energy_j, tok/s, %-of-wall, achieved_gflops are all
available as metrics); INVALID runs are dropped and counted.
"""

from flightrec.measure import measure_artifact
from flightrec.stats import summarize, median_ci


def aggregate(run_dirs, metric="kernel_s", bytes_moved=None, flops=0, tokens=None):
    """Distribution of ``metric`` across valid runs in ``run_dirs``.

    Args:
        run_dirs: Artifact directories for N runs of one config.
        metric: Field from the utilization vector to aggregate (e.g. kernel_s,
            tok_s, energy_j, achieved_gbps, achieved_gflops, pct_of_wall).
        bytes_moved, flops, tokens: Passed to ``measure_artifact`` so derived
            metrics (%-of-wall, tok/s, …) are populated; constant across the runs.

    Returns:
        summarize() stats + n_total/n_valid/n_invalid_dropped + median_ci95,
        or an error dict when no valid run carries the metric.
    """
    values, n_valid = valid_metric(run_dirs, metric, bytes_moved, flops, tokens)
    base = {
        "metric": metric,
        "n_total": len(run_dirs),
        "n_valid": n_valid,
        "n_invalid_dropped": len(run_dirs) - n_valid,
    }
    if not values:
        return {**base, "error": f"no valid run carries metric '{metric}'"}
    summary = summarize(values)
    summary.update(base)
    summary["median_ci95"] = median_ci(values)
    return summary


def valid_metric(run_dirs, metric, bytes_moved=None, flops=0, tokens=None):
    """One metric across runs that passed validity; returns (values, n_valid).

    Throttled/INVALID runs are dropped (PROTOCOLS §3). Shared by aggregate() and the
    cross-box replication verdict so both honour the same drop rule.
    """
    valid = _valid_vectors(run_dirs, bytes_moved, flops, tokens)
    return [v[metric] for v in valid if v.get(metric) is not None], len(valid)


def _valid_vectors(run_dirs, bytes_moved, flops, tokens):
    """Utilization vectors for runs that passed validity (throttled runs dropped)."""
    vectors = (measure_artifact(d, bytes_moved=bytes_moved, flops=flops, tokens=tokens)
               for d in run_dirs)
    return [v for v in vectors if v.get("valid")]
