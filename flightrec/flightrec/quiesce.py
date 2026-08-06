# File: quiesce.py
# Location: flightrec/quiesce.py
# Purpose: Pre-flight quiescence gate — is the box AT REST before we measure? (the "patient resting?" vital)
# Dependencies: stdlib (os, time, statistics) + flightrec.sample_gpu (NVML readers, imported lazily)

"""Quiescence gate — flag a measurement taken on a busy box.

flightrec already auto-invalidates a THROTTLED run (``validate.py``). Contention
is the dual failure: a perfectly-clocked run whose numbers are still skewed
because some OTHER process (an embed rebuild, a second model) shared the GPU/CPU.
Every other vital is uninterpretable without it — an SpO2 reading means nothing
if you don't know the patient was jogging.

GB10 gives no per-PID GPU accounting (NVML compute-apps is N/A), so we cannot
subtract the foreigner — but we CAN read the box's resting vitals BEFORE the
workload and refuse to call a measurement clean when it wasn't. The pure verdict
(``classify``) is split from sampling (``read_baseline``) so the threshold logic
is tested without a GPU.
"""

from __future__ import annotations

import os
import time
from statistics import median

# Resting floors. Mirror watch.IDLE_BUSY_PCT / IDLE_POWER_W (the same physical
# "GPU doing nothing" thresholds) plus a CPU-contention floor. Kept local rather
# than cross-imported to avoid same-layer coupling; see watch.py for the source.
GPU_BUSY_FLOOR_PCT = 10.0  # foreign duty above this => GPU not at rest
GPU_POWER_FLOOR_W = 25.0  # foreign power above this => GPU actively working
LOAD_PER_CPU_FLOOR = 0.5  # 1-min loadavg per core above this => CPU contention


def classify(
    busy_pct,
    power_w,
    load_per_cpu,
    *,
    busy_floor=GPU_BUSY_FLOOR_PCT,
    power_floor=GPU_POWER_FLOOR_W,
    load_floor=LOAD_PER_CPU_FLOOR,
):
    """Resting-vitals verdict from foreign-load signals (pure; no I/O).

    Args:
        busy_pct: GPU duty-cycle at baseline (%).
        power_w: GPU power at baseline (W).
        load_per_cpu: 1-min loadavg divided by core count.
        busy_floor, power_floor, load_floor: contention thresholds.

    Returns:
        Dict with ``quiet`` (bool), ``verdict`` (QUIET/CONTENDED), the offending
        ``reasons``, and the three measured signals — always present so a caller
        can stamp the artifact and a human can read the resting vitals verbatim.
    """
    reasons = []
    if busy_pct > busy_floor:
        reasons.append(f"GPU {busy_pct:.1f}% busy (rest <= {busy_floor:.0f}%)")
    if power_w > power_floor:
        reasons.append(f"GPU {power_w:.0f} W (rest <= {power_floor:.0f} W)")
    if load_per_cpu > load_floor:
        reasons.append(f"CPU {load_per_cpu:.2f}/core (rest <= {load_floor:.2f})")
    quiet = not reasons
    return {
        "quiet": quiet,
        "verdict": "QUIET" if quiet else "CONTENDED",
        "reasons": reasons or ["box at rest"],
        "gpu_busy_pct": round(busy_pct, 1),
        "gpu_power_w": round(power_w, 1),
        "load_per_cpu": round(load_per_cpu, 2),
    }


def _load_per_cpu():
    """1-min loadavg normalised by core count (1.0 = fully-committed box)."""
    return os.getloadavg()[0] / (os.cpu_count() or 1)


def read_baseline(seconds=1.0, hz=5.0):
    """Sample resting GPU + CPU vitals over a short window; return the medians.

    Imports the NVML readers lazily so ``classify`` stays GPU-free for tests.
    GPU duty refreshes at ~6 Hz on GB10, so the default 5 Hz does not oversample.
    """
    from flightrec.sample_gpu import open_gpu, gpu_fast, gpu_util

    handle = open_gpu()
    busy, power = [], []
    for _ in range(max(1, int(seconds * hz))):
        power.append(gpu_fast(handle)["power_w"])
        busy.append(gpu_util(handle)["gpu_busy_pct"])
        time.sleep(1.0 / hz)
    return {
        "gpu_busy_pct": median(busy),
        "gpu_power_w": median(power),
        "load_per_cpu": _load_per_cpu(),
    }


def quiescence(seconds=1.0, hz=5.0, **floors):
    """Sample the box's resting vitals and return the quiescence verdict."""
    base = read_baseline(seconds, hz)
    return classify(base["gpu_busy_pct"], base["gpu_power_w"], base["load_per_cpu"], **floors)
