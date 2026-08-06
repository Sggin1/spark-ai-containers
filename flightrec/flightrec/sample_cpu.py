# File: sample_cpu.py
# Location: flightrec/sample_cpu.py
# Purpose: Cheap unprivileged CPU/memory/pressure readers from /proc and sysfs.
# Dependencies: none (stdlib only)

"""Per-core CPU, memory, and pressure-stall readers for the Tier-1 recorder.

Answers the "only N cores active" question directly (per-core busy% + count).
Frequency uses ``cpuinfo_avg_freq`` (CPPC delivered clock) — ``scaling_cur_freq``
is only the governor's requested target on this box and lies under the
performance governor.
"""

_NPROC = 20  # GB10: 10 Cortex-X925 P-cores + 10 A725 E-cores


def read_stat():
    """Parse per-core jiffies from /proc/stat -> {core: (idle_jiffies, total)}."""
    out = {}
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("cpu") or line[3] == " ":
                continue
            parts = line.split()
            vals = [int(x) for x in parts[1:]]
            out[int(parts[0][3:])] = (vals[3] + vals[4], sum(vals))
    return out


def cpu_busy(prev, cur, threshold=20.0):
    """Per-core busy% from jiffie deltas + count of cores above `threshold`."""
    row, active = {}, 0
    for core, (idle, total) in cur.items():
        p_idle, p_total = prev.get(core, (idle, total))
        span = total - p_total
        busy = 100.0 * (1.0 - (idle - p_idle) / span) if span else 0.0
        row[f"c{core}_busy"] = round(busy, 1)
        active += busy >= threshold
    row["n_active"] = active
    return row


def cpu_freq_mhz():
    """Mean delivered core clock (CPPC cpuinfo_avg_freq), slow-cadence signal."""
    vals = [_freq_one(i) for i in range(_NPROC)]
    good = [v for v in vals if v]
    return {"cpu_freq_mean_mhz": round(sum(good) / len(good)) if good else 0}


def mem_sample():
    """System unified-memory availability (NVML memory-info is N/A on GB10)."""
    return {"mem_avail_mb": round(_meminfo_kb("MemAvailable") / 1024.0, 1)}


def psi_sample():
    """Pressure-stall (cpu+mem) avg10 — cheap, hypothesis-free stall signal."""
    return {"psi_cpu_some10": _psi("cpu"), "psi_mem_some10": _psi("memory")}


def _freq_one(core):
    base = f"/sys/devices/system/cpu/cpu{core}/cpufreq/"
    khz = _read_int(base + "cpuinfo_avg_freq") or _read_int(base + "scaling_cur_freq")
    return khz / 1000.0 if khz else 0.0


def _read_int(path):
    """Robust sysfs integer read (binary mode; some /sys files break text decode)."""
    try:
        with open(path, "rb") as handle:
            text = handle.read().decode("utf-8", "ignore").strip()
    except Exception:  # noqa: BLE001 - a bad sysfs read must never kill the sampler
        return 0
    return int(text) if text.isdigit() else 0


def _meminfo_kb(key):
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(key):
                return float(line.split()[1])
    return 0.0


def _psi(resource):
    try:
        with open(f"/proc/pressure/{resource}", "r", encoding="utf-8") as handle:
            tokens = handle.readline().split()
        return float(next(t for t in tokens if t.startswith("avg10="))[6:])
    except (OSError, StopIteration):
        return 0.0
