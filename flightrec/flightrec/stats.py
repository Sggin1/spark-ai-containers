# File: stats.py
# Location: flightrec/stats.py
# Purpose: Distribution stats for run times — median/IQR/CV + bootstrap A/B.
# Dependencies: numpy

"""A single number is an anecdote; a distribution is a fact.

Run times are right-skewed (a throttle or scheduler hiccup only ever adds a long
tail), so report median + IQR, never mean +/- std. ``compare`` decides whether a
config difference is real via a bootstrap CI on the median difference: a CI that
excludes zero is a fact; one that straddles zero is "not yet a fact".
"""

import numpy as np


def summarize(values):
    """Median, IQR, CV%, min/max for a sample of run times."""
    arr = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    mean = arr.mean()
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "iqr": float(q3 - q1),
        "cv_pct": round(100.0 * arr.std() / mean, 2) if mean else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compare(baseline, candidate, iters=10000, seed=0, min_effect_pct=1.0):
    """Bootstrap 95% CI on median(candidate) - median(baseline).

    Reports BOTH statistical significance (CI excludes zero) and PRACTICAL
    significance (|relative effect| >= min_effect_pct). At very low CV the CI
    excludes zero for trivial sub-1% deltas, so a real lever must clear both gates.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(baseline, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    diffs = [_boot_diff(rng, base, cand) for _ in range(iters)]
    low, high = np.percentile(diffs, [2.5, 97.5])
    delta = float(np.median(cand) - np.median(base))
    base_med = float(np.median(base))
    rel_pct = round(100.0 * delta / base_med, 2) if base_med else 0.0
    significant = bool(low > 0 or high < 0)
    return {
        "delta_median": delta,
        "rel_pct": rel_pct,
        "ci95": [float(low), float(high)],
        "significant": significant,
        "practical": bool(significant and abs(rel_pct) >= min_effect_pct),
    }


def median_ci(values, iters=10000, seed=0):
    """Bootstrap 95% CI on the median of a single sample (the N-run distribution)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    meds = [np.median(rng.choice(arr, arr.size)) for _ in range(iters)]
    low, high = np.percentile(meds, [2.5, 97.5])
    return [float(low), float(high)]


def _boot_diff(rng, base, cand):
    rb = rng.choice(base, base.size)
    rc = rng.choice(cand, cand.size)
    return np.median(rc) - np.median(rb)
