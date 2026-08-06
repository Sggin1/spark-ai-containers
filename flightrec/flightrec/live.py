# File: live.py
# Location: flightrec/live.py
# Purpose: Live one-screen GPU/CPU monitor with a throttle alarm — catch a wedge AS it happens.
# Dependencies: flightrec.sample_gpu, flightrec.sample_cpu, flightrec.validate (stdlib TUI, no extra deps)

"""Live machine-state monitor (the ``flightrec live`` arm).

A throttle today is only diagnosed POST-HOC (at gate time), so a wedged 30-run
campaign isn't caught until it has already burned. ``live`` polls NVML + the CPU
readers independently of any recorder and redraws a one-screen card — SM clock (vs
max), power, energy-rate (J/s), GPU busy %, active cores, PSI, CPU freq — flashing
a **THROTTLE ALARM** the instant a clock droop is detected, so you abort and retry
immediately instead of after the fact.

Stdlib only: a plain ANSI home-and-redraw, no rich/textual/curses dependency. The
alarm reuses ``validate``'s wedge logic exactly (absolute PD-floor OR relative droop
vs the run's own observed peak), so "live alarm" and "post-hoc INVALID" never
disagree. Standalone by design (its own NVML loop) — the simplest thing that catches
a wedge in flight; phase-boundary awareness is the recorder-attached sidecar's job.
"""

import argparse
import time

from flightrec.sample_gpu import open_gpu, gpu_fast, gpu_energy, gpu_util
from flightrec.sample_cpu import (read_stat, cpu_busy, cpu_freq_mhz,
                                  mem_sample, psi_sample)
from flightrec.validate import (PD_CLOCK_FLOOR_MHZ, PD_POWER_FLOOR_W,
                                LOAD_PCT, CLOCK_DROOP_PCT)

_RAW = time.CLOCK_MONOTONIC_RAW
_HOME = "\033[2J\033[H"   # clear screen + cursor home
_RED, _RESET = "\033[1;31m", "\033[0m"


def alarm_state(clock, power, busy, peak_clock):
    """True when, under load, the GPU shows the wedge signature (same rule as validate).

    Two criteria, both gated on duty >= LOAD_PCT so a benign idle window never fires:
    the absolute PD-floor (clock < floor AND power < floor) OR a relative droop more
    than ``CLOCK_DROOP_PCT`` below the run's own observed peak clock.
    """
    if clock is None or (busy or 0.0) < LOAD_PCT:
        return False
    return _pd_floor(clock, power) or _droop(clock, peak_clock)


def _pd_floor(clock, power):
    """Absolute GB10 PD-wedge signature: clock and power both below their floors."""
    return clock < PD_CLOCK_FLOOR_MHZ and (power or 0.0) < PD_POWER_FLOOR_W


def _droop(clock, peak_clock):
    """Relative droop: clock more than CLOCK_DROOP_PCT below the observed peak."""
    return bool(peak_clock and clock < peak_clock * (1.0 - CLOCK_DROOP_PCT / 100.0))


class Dashboard:
    """Rolling state for the live loop: energy-rate diff + peak-clock baseline + alarm."""

    def __init__(self):
        self.peak_clock = 0
        self._prev_energy = None
        self._prev_t = None

    def update(self, row, t_s):
        """Fold a raw reading into a render-ready frame (energy-rate, peak, alarm)."""
        clock = row.get("sm_clock_mhz")
        busy = row.get("gpu_busy_pct") or 0.0
        if clock is not None and busy >= LOAD_PCT:
            self.peak_clock = max(self.peak_clock, clock)
        rate = self._energy_rate(row.get("energy_mj"), t_s)
        alarm = alarm_state(clock, row.get("power_w"), busy, self.peak_clock)
        return {**row, "energy_rate_w": rate, "peak_clock_mhz": self.peak_clock,
                "alarm": alarm}

    def _energy_rate(self, energy_mj, t_s):
        """Watts from consecutive energy samples; None on the first tick."""
        rate = None
        if energy_mj is not None and self._prev_energy is not None:
            dt = t_s - self._prev_t
            if dt > 0:
                rate = round((energy_mj - self._prev_energy) / 1000.0 / dt, 1)
        if energy_mj is not None:
            self._prev_energy, self._prev_t = energy_mj, t_s
        return rate


def format_frame(frame, max_clock=None):
    """Render one frame as a compact card; a THROTTLE ALARM banner when flagged."""
    clock = frame.get("sm_clock_mhz")
    ceil = f"/{max_clock}" if max_clock else ""
    lines = [
        "  flightrec live — GB10",
        "  " + "-" * 46,
        f"  SM clock   {_val(clock)}{ceil} MHz   (peak {frame.get('peak_clock_mhz')})",
        f"  power      {_val(frame.get('power_w'))} W      energy {_val(frame.get('energy_rate_w'))} J/s",
        f"  GPU busy   {_val(frame.get('gpu_busy_pct'))} %      temp   {_val(frame.get('temp_c'))} C",
        f"  cores >20% {_val(frame.get('n_active'))}/20     CPU clk {_val(frame.get('cpu_freq_mean_mhz'))} MHz",
        f"  PSI cpu    {_val(frame.get('psi_cpu_some10'))}      mem free {_val(frame.get('mem_avail_mb'))} MB",
    ]
    if frame.get("alarm"):
        lines.append("")
        lines.append(f"  {_RED}### THROTTLE ALARM — clock droop under load — ABORT & RETRY ###{_RESET}")
    return "\n".join(lines)


def _val(value):
    return "n/a" if value is None else value


def read_row(handle, prev_stat):
    """One raw reading (GPU + single-shot CPU) plus n_active from the stat diff.

    Returns ``(row, cur_stat)`` so the caller can thread ``cur_stat`` into the next
    tick as ``prev_stat`` (n_active needs two jiffie snapshots to difference).
    """
    row = {}
    row.update(_safe(lambda: gpu_fast(handle)))
    row.update(_safe(lambda: gpu_energy(handle)))
    row.update(_safe(lambda: gpu_util(handle)))
    row.update(_safe(mem_sample))
    row.update(_safe(psi_sample))
    row.update(_safe(cpu_freq_mhz))
    cur = _safe(read_stat) or {}
    if prev_stat and cur:
        row.update(_safe(lambda: cpu_busy(prev_stat, cur)))
    return row, cur


def live(hz=2.0, duration=None, gpu_index=0, once=False):
    """Poll and redraw the live card until *duration* elapses, *once*, or Ctrl-C."""
    handle = _safe(lambda: open_gpu(gpu_index))
    max_clock = _safe(lambda: _max_clock(handle))
    board = Dashboard()
    prev_stat, start = None, time.clock_gettime(_RAW)
    period = 1.0 / hz
    try:
        while True:
            now = time.clock_gettime(_RAW)
            row, prev_stat = read_row(handle, prev_stat)
            frame = board.update(row, now)
            print(_HOME + format_frame(frame, max_clock), flush=True)
            if once or _expired(start, now, duration):
                return frame
            time.sleep(period)
    except KeyboardInterrupt:
        print()
        return None


def _expired(start, now, duration):
    """True once *duration* seconds have elapsed since *start* (None => run forever)."""
    return bool(duration and now - start >= duration)


def _max_clock(handle):
    import pynvml
    return pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)


def _safe(thunk):
    try:
        return thunk()
    except Exception:  # noqa: BLE001 - a bad reader must never kill the live loop
        return {}


def main(argv=None):
    """CLI entry for the standalone live monitor."""
    ap = argparse.ArgumentParser(description="flightrec live GPU/CPU monitor + throttle alarm")
    ap.add_argument("--hz", type=float, default=2.0, help="refresh rate (default 2 Hz)")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--once", action="store_true", help="print one frame and exit")
    args = ap.parse_args(argv)
    live(hz=args.hz, duration=args.duration, once=args.once)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
