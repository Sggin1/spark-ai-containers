# File: test_stats.py
# Location: tests/test_stats.py
# Purpose: median/IQR summary + bootstrap A/B significance behave correctly.
# Dependencies: numpy, flightrec.stats

"""A distribution is a fact; a real difference survives the bootstrap, noise does not."""

import numpy as np

from flightrec.stats import summarize, compare


def test_summarize_median_iqr():
    summary = summarize([10, 10, 10, 10])
    assert summary["median"] == 10
    assert summary["iqr"] == 0
    assert summary["cv_pct"] == 0.0


def test_compare_detects_real_difference():
    rng = np.random.default_rng(0)
    baseline = rng.normal(100, 1, 50)
    candidate = rng.normal(110, 1, 50)
    assert compare(baseline, candidate)["significant"] is True


def test_compare_ignores_noise():
    rng = np.random.default_rng(1)
    baseline = rng.normal(100, 5, 50)
    candidate = rng.normal(100, 5, 50)
    assert compare(baseline, candidate)["significant"] is False


def test_compare_significant_but_not_practical():
    # tiny effect (0.2%) with ultra-low noise: CI excludes zero, but it is negligible
    rng = np.random.default_rng(2)
    baseline = rng.normal(100, 0.05, 80)
    candidate = rng.normal(100.2, 0.05, 80)
    result = compare(baseline, candidate)
    assert result["significant"] is True
    assert result["practical"] is False
