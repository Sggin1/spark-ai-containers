# File: sample_gpu.py
# Location: flightrec/sample_gpu.py
# Purpose: Cheap unprivileged GPU-state readers for GB10 (NVML), Tier-1 only.
# Dependencies: nvidia-ml-py (pynvml)

"""GB10 GPU signal readers verified to work on sm_121 (driver 580.159.03).

Deliberately excludes DRAM bandwidth and memory-info: GB10 exposes no live
DRAM-bytes counter and ``nvmlDeviceGetMemoryInfo`` returns NOT_SUPPORTED on the
unified LPDDR5X part. Bandwidth is modelled downstream, never sampled here. The
throttle bitmask IS captured but is unreliable on GB10 (fires spuriously); trust
the clock+power drop in ``validate.py`` instead.
"""

import pynvml

_SM = pynvml.NVML_CLOCK_SM
_TEMP = pynvml.NVML_TEMPERATURE_GPU


def open_gpu(index=0):
    """Initialise NVML and return a device handle (idempotent; call once)."""
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(index)


def gpu_fast(handle):
    """Cheap signals safe at 20-50 Hz (each sub-microsecond; NVML refresh ~1 kHz).

    Energy is deliberately NOT here: ``nvmlDeviceGetTotalEnergyConsumption`` costs
    ~1.6 ms/call (99% of this function's old cost) and is a monotonic integral, so
    it loses nothing sampled slowly — see ``gpu_energy`` on the ~6 Hz path.
    """
    return {
        "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
        "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, _SM),
        "pstate": pynvml.nvmlDeviceGetPerformanceState(handle),
        "temp_c": pynvml.nvmlDeviceGetTemperature(handle, _TEMP),
        "throttle_bits": pynvml.nvmlDeviceGetCurrentClocksEventReasons(handle),
    }


def gpu_energy(handle):
    """Cumulative GPU energy (mJ). Expensive (~1.6 ms) but monotonic: sampled on
    the slow path, only ever used as a max-min delta, so no information is lost."""
    return {"energy_mj": pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)}


def gpu_util(handle):
    """Coarse GPU duty-cycle. NVML refreshes only ~6 Hz; do not oversample."""
    rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return {"gpu_busy_pct": rates.gpu}
