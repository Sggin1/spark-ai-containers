# File: recorder.py
# Location: flightrec/recorder.py
# Purpose: Tier-1 always-on flight recorder; wraps a bench into a self-describing artifact.
# Dependencies: pandas, flightrec.sample_gpu, flightrec.sample_cpu, flightrec.header

"""Tier-1 always-on flight recorder for GB10 benchmarks.

Samples the broad, cheap machine state (GPU power/clock/throttle/energy, all 20
CPU cores, pressure, memory) on one CLOCK_MONOTONIC_RAW timeline — a ~20 Hz fast
path (sub-millisecond NVML reads) plus a ~6 Hz slow path for the costly readers
(energy, util, freq), with no privilege. The recorder measures its OWN CPU cost
each run and writes it to header.json (``recorder.overhead_pct_of_core``) — the
overhead is reported as data, not asserted. It captures EVERY run so a throttled
run can be auto-invalidated downstream. No hypothesis about which bottleneck
matters is baked in — it records the whole state so any of them can surface.

    with FlightRecorder("results/run01") as rec:
        with rec.phase("decode"):
            run_benchmark()
"""

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from flightrec.sample_gpu import open_gpu, gpu_fast, gpu_energy, gpu_util
from flightrec.sample_cpu import read_stat, cpu_busy, cpu_freq_mhz, mem_sample, psi_sample
from flightrec.header import capture_header

_RAW = time.CLOCK_MONOTONIC_RAW
_CPU = time.CLOCK_THREAD_CPUTIME_ID  # the sampler thread's own CPU time (self-overhead)


class FlightRecorder:
    """Context manager that samples machine state while a benchmark runs."""

    def __init__(self, out_dir, hz=20, util_hz=6, gpu_index=0,
                 quiesce_window=None, quiesce_floors=None):
        self.out = Path(out_dir)
        self.period = 1.0 / hz
        self.util_div = max(1, round(hz / util_hz))
        self.gpu_index = gpu_index
        self.quiesce_window = quiesce_window
        self.quiesce_floors = quiesce_floors or {}
        self._rows, self._phases = [], []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None
        self._diag = {}
        self._quiesce = None

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self._quiesce = self._sample_quiescence()
        self._t0 = time.clock_gettime_ns(_RAW)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _sample_quiescence(self):
        """Resting-vitals verdict sampled BEFORE the workload; None when disabled.

        Taken pre-t0 so the baseline window is not counted as workload time, and
        stamped into header.json: every other vital is uninterpretable if the box
        was already contended when the run started (see quiesce.py).
        """
        if not self.quiesce_window:
            return None
        from flightrec import quiesce
        sample = _safe(lambda: quiesce.quiescence(
            seconds=self.quiesce_window, **self.quiesce_floors))
        return sample or None

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._flush()
        return False

    @contextmanager
    def phase(self, name):
        """Mark a named phase; also emits an NVTX range if torch is importable."""
        _nvtx_push(name)
        start = time.clock_gettime_ns(_RAW)
        try:
            yield
        finally:
            self._phases.append({"phase": name, "t0_ns": start,
                                 "t1_ns": time.clock_gettime_ns(_RAW)})
            _nvtx_pop()

    def _loop(self):
        handle = open_gpu(self.gpu_index)
        prev = _read_stat_safe()
        cpu0, wall0, tick, overruns = time.clock_gettime(_CPU), self._t0, 0, 0
        prev_tick_ns = self._t0
        while not self._stop.is_set():
            now = time.clock_gettime_ns(_RAW)
            actual_dt_ms = (now - prev_tick_ns) / 1e6
            row = {"t_ns": now, "t_rel_s": (now - self._t0) / 1e9,
                   "actual_dt_ms": actual_dt_ms}
            row.update(_safe(lambda: gpu_fast(handle)))
            cur = _read_stat_safe()
            row.update(_safe(lambda: cpu_busy(prev, cur)))
            prev = cur
            if tick % self.util_div == 0:
                row.update(self._slow(handle))
            self._rows.append(row)
            prev_tick_ns = now
            tick += 1
            overruns += _pace(now, self.period)
        self._diag = self._make_diag(cpu0, wall0, tick, overruns)

    def _slow(self, handle):
        """Slow-cadence (~6 Hz) extras; each guarded so none can stall the loop.

        Energy lives here on purpose: it is a ~1.6 ms NVML call but a monotonic
        integral, so 6 Hz loses nothing and keeps the fast loop sub-millisecond.
        """
        out = {}
        out.update(_safe(lambda: gpu_energy(handle)))
        out.update(_safe(lambda: gpu_util(handle)))
        out.update(_safe(mem_sample))
        out.update(_safe(psi_sample))
        out.update(_safe(cpu_freq_mhz))
        return out

    def _make_diag(self, cpu0, wall0_ns, ticks, overruns):
        """Self-measured recorder cost: the honest overhead, emitted per run."""
        cpu_s = time.clock_gettime(_CPU) - cpu0
        wall_s = (time.clock_gettime_ns(_RAW) - wall0_ns) / 1e9
        base_hz = round(1.0 / self.period, 1)
        return {
            "ticks": ticks,
            "wall_s": round(wall_s, 3),
            "cpu_s": round(cpu_s, 4),
            "overhead_pct_of_core": round(100.0 * cpu_s / wall_s, 3) if wall_s else None,
            "mean_period_ms": round(1000.0 * wall_s / ticks, 2) if ticks else None,
            "overrun_ticks": overruns,
            "overrun_pct": round(100.0 * overruns / ticks, 1) if ticks else None,
            "base_hz": base_hz,
            "util_hz": round(base_hz / self.util_div, 1),
        }

    def _flush(self):
        pd.DataFrame(self._rows).to_parquet(self.out / "samples.parquet")
        pd.DataFrame(self._phases or [{"phase": None}]).to_parquet(
            self.out / "phases.parquet")
        header = capture_header(self.gpu_index)
        header["recorder"] = self._diag
        if self._quiesce is not None:
            header["quiescence"] = self._quiesce
        (self.out / "header.json").write_text(
            json.dumps(header, indent=2), encoding="utf-8")


def _safe(thunk):
    """Run a reader thunk; never let one bad read kill the sampling loop."""
    try:
        return thunk()
    except Exception:  # noqa: BLE001 - sampler robustness over precision
        return {}


def _read_stat_safe():
    """Per-core jiffies, guarded — returns {} so a bad read just skips a tick."""
    try:
        return read_stat()
    except Exception:  # noqa: BLE001 - sampler robustness over precision
        return {}


def _pace(start_ns, period_s):
    """Sleep to hold the cadence; return 1 if the tick already overran its period."""
    elapsed = (time.clock_gettime_ns(_RAW) - start_ns) / 1e9
    if elapsed < period_s:
        time.sleep(period_s - elapsed)
        return 0
    return 1


def _nvtx_push(name):
    try:
        import torch
        torch.cuda.nvtx.range_push(name)
    except Exception:  # noqa: BLE001 - torch optional
        pass


def _nvtx_pop():
    try:
        import torch
        torch.cuda.nvtx.range_pop()
    except Exception:  # noqa: BLE001 - torch optional
        pass
