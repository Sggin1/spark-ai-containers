# File: validate.py
# Location: flightrec/validate.py
# Purpose: Auto-invalidate a run that throttled (PD-wedge detector), from clock+power.
# Dependencies: pandas (df passed in)

"""Run-validity verdict.

A benchmark whose clock/power collapsed is not a slow kernel — it is an invalid
measurement. We detect the GB10 PD/USB-C wedge (611 MHz / 13 W signature)
directly from clock and power. The NVML throttle bitmask is intentionally NOT
used: it was verified to fire spuriously on this box while the clock held.

Pass phase-scoped samples (samples inside a marked phase): genuine idle also
shows low clock+power, so scoping to the active window avoids false positives.
"""

import pandas as pd

PD_CLOCK_FLOOR_MHZ = 1400  # healthy >= 1400; PD/USB-C wedge ~611
PD_POWER_FLOOR_W = 30      # healthy under load >= 40; wedge ~13
LOAD_PCT = 50.0            # GPU duty-cycle that counts as "work pending"
CLOCK_DROOP_PCT = 50       # flag if clock drops > 50% below loaded p75


def verdict(samples, clock_floor=PD_CLOCK_FLOOR_MHZ, power_floor=PD_POWER_FLOOR_W,
            load_pct=LOAD_PCT, clock_droop_pct=CLOCK_DROOP_PCT):
    """Return validity + throttle stats for a sample frame.

    Throttle detection uses the UNION of two criteria, both scoped to loaded samples:

    1. Absolute floor (GB10 PD-wedge): clock < clock_floor AND power < power_floor.
    2. Relative droop: clock < loaded_p75_clock * (1 - clock_droop_pct/100), where
       loaded_p75_clock is the 75th-percentile SM clock among loaded samples. This
       generalises the check to hardware that does not show the 611/13 signature —
       any run where clock falls more than clock_droop_pct% below the run's own
       expected level while under load is flagged as throttled.

    Gating on duty-cycle separates a genuine throttle from a benign idle window.
    """
    count = len(samples)
    if not count:
        return {"valid": False, "reason": "no samples", "samples": 0}
    clock, power = samples["sm_clock_mhz"], samples["power_w"]
    under_load = _busy(samples) >= load_pct

    # Criterion 1: absolute floor (original PD-wedge check).
    abs_throttled = under_load & (clock < clock_floor) & (power < power_floor)

    # Criterion 2: relative droop vs the run's own loaded p75 clock.
    loaded_mask = under_load
    loaded_frame = samples.loc[loaded_mask]
    rel_throttled = pd.Series(False, index=samples.index)
    if not loaded_frame.empty:
        loaded_p75 = loaded_frame["sm_clock_mhz"].quantile(0.75)
        droop_threshold = loaded_p75 * (1.0 - clock_droop_pct / 100.0)
        rel_throttled = under_load & (clock < droop_threshold)

    throttled = samples[abs_throttled | rel_throttled]
    loaded = int(under_load.sum())

    # Energy monotonicity: energy_mj is a monotonic NVML integral — it must never decrease.
    # A drop indicates a counter reset or sampling error; j_token computed from such a run
    # would be silently wrong. Flag and invalidate so the caller retries rather than records
    # a corrupted energy reading.
    energy_ok = _energy_monotonic(samples)

    return {
        "valid": bool(throttled.empty) and energy_ok,
        "samples": count,
        "loaded_samples": loaded,
        "load_pct_of_window": round(100.0 * loaded / count, 1),
        "throttled_samples": len(throttled),
        "throttled_pct_of_loaded": round(100.0 * len(throttled) / loaded, 2) if loaded else 0.0,
        "sm_clock_min_mhz": int(clock.min()),
        "sm_clock_median_mhz": int(clock.median()),
        "power_min_w": round(float(power.min()), 1),
        "power_max_w": round(float(power.max()), 1),
        "energy_monotonic": energy_ok,
    }


def _energy_monotonic(samples):
    """True when energy_mj never decreases between consecutive ticks (or column absent)."""
    if "energy_mj" not in samples.columns:
        return True
    energy = samples["energy_mj"].dropna()
    if len(energy) < 2:
        return True
    return bool((energy.diff().dropna() >= 0).all())


def _busy(samples):
    """Forward-filled GPU duty-cycle (sampled ~6 Hz) aligned to every row."""
    if "gpu_busy_pct" not in samples.columns:
        return pd.Series(0.0, index=samples.index)
    return samples["gpu_busy_pct"].ffill().fillna(0.0)
