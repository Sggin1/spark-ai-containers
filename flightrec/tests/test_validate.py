# File: test_validate.py
# Location: tests/test_validate.py
# Purpose: verdict() must separate idle from PD-wedge from healthy-under-load.
# Dependencies: pandas, flightrec.validate

"""Throttle auto-invalidation must not confuse a benign idle window with a wedge."""

import pandas as pd

from flightrec.validate import verdict


def _frame(clock, power, busy, rows=20):
    return pd.DataFrame({
        "sm_clock_mhz": [clock] * rows,
        "power_w": [power] * rows,
        "gpu_busy_pct": [busy] * rows,
    })


def test_idle_window_is_valid():
    result = verdict(_frame(305, 6.0, 2.0))  # low clock+power but GPU idle
    assert result["valid"] is True
    assert result["loaded_samples"] == 0


def test_wedge_under_load_is_invalid():
    result = verdict(_frame(611, 13.0, 95.0))  # the 611 MHz / 13 W signature, busy
    assert result["valid"] is False
    assert result["throttled_samples"] > 0


def test_healthy_under_load_is_valid():
    result = verdict(_frame(2300, 60.0, 95.0))
    assert result["valid"] is True
    assert result["loaded_samples"] > 0


def test_partial_droop_flagged():
    """Half at 2300 MHz, half at 600 MHz (both under load) → relative droop → INVALID."""
    healthy = _frame(2300, 60.0, 95.0, rows=10)
    drooped = _frame(600, 60.0, 95.0, rows=10)
    df = pd.concat([healthy, drooped], ignore_index=True)
    result = verdict(df)
    assert result["valid"] is False
    assert result["throttled_samples"] > 0


def test_mid_run_droop_no_power_collapse():
    """Run at 1800 MHz drops to 700 MHz mid-run while loaded; power stays above 30 W.

    The old absolute-floor+power check would miss this (power >= 30 W throughout).
    The relative-droop check must catch it.
    """
    normal = _frame(1800, 50.0, 95.0, rows=15)
    drooped = _frame(700, 50.0, 95.0, rows=5)
    df = pd.concat([normal, drooped], ignore_index=True)
    result = verdict(df)
    assert result["valid"] is False
    assert result["throttled_samples"] > 0
