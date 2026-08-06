# File: gate.py
# Location: flightrec/gate.py
# Purpose: Submission gate: PASS only if VALID + CI-tight + (optionally) replicated.
# Dependencies: flightrec.aggregate, flightrec.compare

"""The submission gate — what turns a provisional number into a reportable fact.

A result is PASS only when it clears every gate, mirroring PROTOCOLS §1/§2/§3/§7:
- **VALID**: enough non-throttled runs (``n_valid >= min_n``) and no throttled run in the set.
- **CI-tight**: the bootstrap CI half-width is a small fraction of the median (``<= max_ci_halfwidth_pct``).
- **replicated** (when a second box's runs are supplied): the two boxes agree (no significant+practical gap).

The CLI exits non-zero on failure, so it can guard a submission script. This is the mechanism that would
have rejected the 112.5 over-claim automatically (3 runs, no CI, no replication).
"""

from flightrec.aggregate import aggregate
from flightrec.compare import replication_over_runs


def gate(runs, replicate_with=None, metric="kernel_s", min_n=20, max_ci_halfwidth_pct=5.0,
         min_effect_pct=1.0, bytes_moved=None, flops=0, tokens=None):
    """Run the submission gates over N runs (and optional cross-box replication)."""
    mkw = {"bytes_moved": bytes_moved, "flops": flops, "tokens": tokens}
    agg = aggregate(runs, metric=metric, **mkw)
    checks = {
        "enough_valid": agg.get("n_valid", 0) >= min_n,
        "no_throttle": agg.get("n_invalid_dropped", 1) == 0,
        "ci_tight": _ci_tight(agg, max_ci_halfwidth_pct),
    }
    rep = None
    if replicate_with:
        rep = replication_over_runs(runs, replicate_with, metric=metric,
                                    min_effect_pct=min_effect_pct, **mkw)
        checks["replicated"] = bool(rep.get("replicated"))
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "aggregate": agg,
        "replication": rep,
        "thresholds": {"min_n": min_n, "max_ci_halfwidth_pct": max_ci_halfwidth_pct,
                       "min_effect_pct": min_effect_pct},
    }


def _ci_tight(agg, max_pct):
    """CI half-width as a % of the median is within the threshold."""
    ci, median = agg.get("median_ci95"), agg.get("median")
    if not ci or not median:
        return False
    half_pct = 100.0 * (ci[1] - ci[0]) / 2.0 / median
    return half_pct <= max_pct
