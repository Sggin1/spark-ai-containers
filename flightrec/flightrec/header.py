# File: header.py
# Location: flightrec/header.py
# Purpose: Capture one-shot run provenance so each artifact is self-describing.
# Dependencies: nvidia-ml-py (pynvml)

"""Run provenance header.

Every recorder artifact carries its own context (driver, clock ceiling/lock,
idle power, the modelled bandwidth wall) so a result can be trusted and
reproduced months later without guessing the conditions it was taken under.
"""

import time

import pynvml

from flightrec.roofline import MODELED_WALL_GBPS, load_calibration

_SM = pynvml.NVML_CLOCK_SM


def capture_header(gpu_index=0):
    """Snapshot provenance fields at run start."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    header = {
        "wall_clock_unix": time.time(),
        "monotonic_raw_ns": time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW),
        "driver": _decode(pynvml.nvmlSystemGetDriverVersion()),
        "gpu_name": _dev_name(handle),
        "sm_clock_max_mhz": _safe(pynvml.nvmlDeviceGetMaxClockInfo, handle, _SM),
        "power_now_w": _power(handle),
        "cntvct_offset_note": "nsys systemClockNs == CNTVCT_EL0, leads MONOTONIC_RAW "
                              "by a fixed ~+19.32 s; re-measure per boot to align Tier-2.",
        "caveats": "DRAM bandwidth has no GB10 HW counter; the wall is MEASURED "
                   "(STREAM-triad) when calibrated, else a modeled fallback. "
                   "Throttle bitmask unreliable; validity judged on clock+power drop.",
    }
    header.update(_bandwidth_wall())
    header.update(_compute_peak())
    return header


def _bandwidth_wall():
    """The achievable BW wall: measured (calibrated) if available, else modeled."""
    cal = load_calibration()
    if cal and cal.get("bandwidth_wall_gbps_measured"):
        return {
            "bandwidth_wall_gbps": cal["bandwidth_wall_gbps_measured"],
            "bandwidth_wall_source": f"measured ({cal.get('method')})",
            "bandwidth_wall_pct_of_spec": cal.get("pct_of_spec"),
            "bandwidth_wall_measured_unix": cal.get("measured_unix"),
        }
    return {
        "bandwidth_wall_gbps": MODELED_WALL_GBPS,
        "bandwidth_wall_source": "modeled-constant (run `flightrec calibrate` to measure)",
        "bandwidth_wall_pct_of_spec": None,
        "bandwidth_wall_measured_unix": None,
    }


def _compute_peak():
    """The compute PEAK (TFLOP/s per dtype), recorded alongside the BW wall.

    Completes the roofline in-artifact: the wall is one axis, this is the other.
    Returns the persisted ``compute_peak_tflops`` dtype-map (+ its per-dtype
    provenance), or Nones when this box has not run ``flightrec calibrate
    --compute`` — never a guessed peak.
    """
    cal = load_calibration()
    peaks = cal.get("compute_peak_tflops") if cal else None
    if not peaks:
        return {"compute_peak_tflops": None, "compute_peak_source": None}
    return {
        "compute_peak_tflops": peaks,
        "compute_peak_source": cal.get("compute_peak_source"),
    }


def _dev_name(handle):
    return _decode(_safe(pynvml.nvmlDeviceGetName, handle))


def _power(handle):
    raw = _safe(pynvml.nvmlDeviceGetPowerUsage, handle)
    return round(raw / 1000.0, 2) if isinstance(raw, (int, float)) else None


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def _decode(value):
    return value.decode() if isinstance(value, bytes) else value
