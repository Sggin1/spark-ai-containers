# File: watch.py
# Location: flightrec/watch.py
# Purpose: Long-run stall/liveness watchdog — alert when a hung job idles the GPU with no progress.
# Dependencies: flightrec.sample_gpu (NVML readers), stdlib (subprocess, threading, signal)

"""Long-run stall watchdog (the ``flightrec watch`` arm).

A multi-hour quant/train can HANG — process alive but GPU at 0 W / 0 %, no progress
for hours — and nothing flags it. Concrete cost: an auto_round re-quant hung at block
16 (an inductor/torch_compile stall) and was only noticed ~2 h later via a manual
status check, because the run was a raw ``nohup``, not flightrec-wrapped. The recorder
already has the signal (the GPU power/clock timeline); ``watch`` is the alerting layer
on top of it.

The verdict is a **conjunction**, deliberately: a job is STALLED only when BOTH

1. its progress marker (a regex over stdout/log, exactly like ``smoke``'s) has not
   advanced for ``stall_s`` seconds, AND
2. the GPU has sat continuously idle (power < ``idle_power_w`` and duty < ``idle_busy_pct``)
   for ``stall_s`` seconds.

Requiring both is what separates a true hang from a legitimately-long compute: a single
huge kernel between markers keeps the GPU BUSY (condition 2 fails → no false alarm),
while a hang drops it to idle. The cost of the conjunction is a miss when a hung job
still draws power; that conservative bias (miss over false-kill) is intentional for a
watchdog that can SIGTERM your job.

Two source modes share one decision core:
  * **wrap**  — ``watch -- <cmd>``: run the job as a child, watch its stdout.
  * **attach** — ``watch --logfile <path> [--pid <pid>]``: tail an already-running
    job's log (the raw-``nohup`` case that motivated this), liveness via ``/proc/<pid>``.

On stall: notify (always), optionally run a checkpoint hook, optionally SIGTERM→SIGKILL.
Complements the other arms: ``smoke`` = before you commit, ``memguard`` = OOM backstop,
``watch`` = during the run (hang/stall). Rule going forward: long runs go through
flightrec, never raw nohup.
"""

import os
import re
import signal
import subprocess
import sys
import threading
import time

import pandas as pd

from flightrec.sample_gpu import open_gpu, gpu_fast, gpu_util

_RAW = time.CLOCK_MONOTONIC_RAW

# Watchdog defaults (GB10-tuned; minutes-scale signal, so polled slowly).
STALL_TIMEOUT_S = 600.0   # 10 min of BOTH marker-silence and GPU-idle => STALLED
POLL_INTERVAL_S = 5.0     # GPU poll cadence; a stall is a minutes-scale event
WARMUP_S = 120.0          # startup grace (weight load is idle + marker-silent but not hung)
IDLE_POWER_W = 20.0       # below this under-expected-load = GPU doing nothing (healthy run >= 40 W)
IDLE_BUSY_PCT = 10.0      # corroborating duty-cycle floor; a hang reads ~0 %
KILL_GRACE_S = 10.0       # seconds after SIGTERM before SIGKILL on --on-stall kill
DEFAULT_LOG = "/tmp/flightrec_watch.log"


class StallMonitor:
    """Pure stall-decision state: marker recency + continuous GPU-idle duration.

    Thread-safe on ``mark`` (called from the reader thread) vs the poll loop's
    reads. Holds no NVML or subprocess handles, so the verdict logic is unit-
    testable against synthetic time and GPU values.
    """

    def __init__(self, stall_s, idle_power_w, idle_busy_pct, warmup_s, start_t):
        self.stall_s = stall_s
        self.idle_power_w = idle_power_w
        self.idle_busy_pct = idle_busy_pct
        self.warmup_s = warmup_s
        self.start_t = start_t
        self.last_marker_t = start_t
        self.idle_since = None
        self.marks = 0
        self._lock = threading.Lock()

    def mark(self, t):
        """Record a progress marker at monotonic time *t* (resets the stall clock)."""
        with self._lock:
            self.last_marker_t = t
            self.marks += 1

    def observe_gpu(self, t, power_w, busy_pct):
        """Fold one GPU poll into the continuous-idle tracker.

        Idle = power below ``idle_power_w`` and (duty unknown or below
        ``idle_busy_pct``). A read failure (power None) is treated as NOT idle so
        a flaky NVML call can never manufacture a stall.
        """
        if self._is_idle(power_w, busy_pct):
            self.idle_since = self.idle_since if self.idle_since is not None else t
        else:
            self.idle_since = None

    def _is_idle(self, power_w, busy_pct):
        """A poll is idle when power is below floor and duty is unknown/below floor."""
        if power_w is None:
            return False
        return power_w < self.idle_power_w and (busy_pct is None or busy_pct < self.idle_busy_pct)

    def stalled(self, t):
        """True iff past warm-up AND both marker-silence and GPU-idle exceed ``stall_s``."""
        if t - self.start_t < self.warmup_s:
            return False
        marker_silent = (t - self.last_marker_t) >= self.stall_s
        gpu_idle = self.idle_since is not None and (t - self.idle_since) >= self.stall_s
        return marker_silent and gpu_idle

    def silent_s(self, t):
        """Seconds since the last marker (the progress-liveness age)."""
        return round(t - self.last_marker_t, 1)


def watch(cmd=None, marker_re=r"$^", logfile=None, pid=None,
          stall_s=STALL_TIMEOUT_S, poll_s=POLL_INTERVAL_S, warmup_s=WARMUP_S,
          idle_power_w=IDLE_POWER_W, idle_busy_pct=IDLE_BUSY_PCT,
          on_stall="notify", checkpoint_cmd=None, notify_cmd=None,
          grace=KILL_GRACE_S, log_path=DEFAULT_LOG, record_dir=None, gpu_index=0):
    """Watch a long job; return a verdict dict when it stalls or finishes.

    Exactly one of *cmd* (wrap mode) or *logfile* (attach mode) drives the marker
    stream; *pid* adds liveness in attach mode and a kill target.

    Args:
        cmd: Job command to run as a child and watch (wrap mode).
        marker_re: Regex; each matching stdout/log line = one progress unit.
        logfile: Path to tail for markers when attaching to a running job.
        pid: Process id to watch for liveness / kill (attach mode).
        stall_s: Seconds of BOTH marker-silence and GPU-idle that mean STALLED.
        poll_s: GPU poll interval.
        warmup_s: Startup grace before stall detection arms.
        idle_power_w, idle_busy_pct: GPU-idle thresholds.
        on_stall: ``"notify"`` (default) or ``"kill"`` (SIGTERM->SIGKILL the job).
        checkpoint_cmd: Optional shell hook run on stall (e.g. trigger a checkpoint).
        notify_cmd: Optional shell hook run on stall (e.g. push an alert).
        grace: Seconds after SIGTERM before SIGKILL.
        log_path: File to append watchdog events to.
        record_dir: If set, write the poll timeline to ``<dir>/samples.parquet`` as
            post-hoc proof of the stall (flat 0 W + frozen mark count).
        gpu_index: NVML device index.

    Returns:
        Verdict dict — ``stalled``, ``reason``, ``marks``, ``power_w``,
        ``sm_clock_mhz``, ``gpu_busy_pct``, ``action``, ``exit_code``.
    """
    handle = _open_gpu_safe(gpu_index)
    start = time.clock_gettime(_RAW)
    monitor = StallMonitor(stall_s, idle_power_w, idle_busy_pct, warmup_s, start)
    stop = threading.Event()
    proc, stream = _start_source(cmd, logfile, stop)
    reader = threading.Thread(target=_marker_reader,
                              args=(stream, marker_re, monitor, stop), daemon=True)
    reader.start()
    rows = []
    with open(log_path, "a") as log:
        _log(log, _banner(cmd, logfile, pid, stall_s, poll_s))
        result = _watch_loop(handle, monitor, proc, pid, poll_s, rows, log,
                             on_stall, checkpoint_cmd, notify_cmd, grace)
    stop.set()
    if record_dir:
        _write_timeline(rows, record_dir)
    return result


def _watch_loop(handle, monitor, proc, pid, poll_s, rows, log,
                on_stall, checkpoint_cmd, notify_cmd, grace):
    """Poll the GPU on a timer; return when stalled or the job exits."""
    while True:
        now = time.clock_gettime(_RAW)
        snap = _poll(handle)
        monitor.observe_gpu(now, snap["power_w"], snap["gpu_busy_pct"])
        rows.append({"t_rel_s": round(now - monitor.start_t, 2),
                     "marks": monitor.marks, **snap})
        if monitor.stalled(now):
            return _handle_stall(monitor, snap, proc, pid, log, now,
                                 on_stall, checkpoint_cmd, notify_cmd, grace)
        done, code = _source_done(proc, pid)
        if done:
            _log(log, f"job exited code={code}; no stall (marks={monitor.marks})")
            return _result(False, "job completed", monitor, snap, "none", code)
        time.sleep(poll_s)


def _handle_stall(monitor, snap, proc, pid, log, now,
                  on_stall, checkpoint_cmd, notify_cmd, grace):
    """Emit the alarm, run hooks, optionally kill; build the stalled verdict."""
    msg = _stall_message(monitor, snap, now)
    _log(log, msg)
    sys.stderr.write("\n" + msg + "\n")
    sys.stderr.flush()
    if notify_cmd:
        _run_hook(notify_cmd, msg, log)
    if checkpoint_cmd:
        _run_hook(checkpoint_cmd, msg, log)
    action = "notify"
    if on_stall == "kill":
        _terminate(proc, pid, grace, log)
        action = "killed"
    return _result(True, "stall detected", monitor, snap, action, None)


# --- source plumbing -------------------------------------------------------

def _start_source(cmd, logfile, stop):
    """Build the marker stream: a child process (wrap) or a log tail (attach)."""
    if cmd:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        return proc, proc.stdout
    return None, _tail(logfile, stop)


def _tail(path, stop):
    """Yield lines appended to *path* from end-of-file until *stop* is set."""
    with open(path, "r") as fh:
        fh.seek(0, os.SEEK_END)
        while not stop.is_set():
            line = fh.readline()
            if line:
                yield line
            else:
                time.sleep(0.2)


def _marker_reader(stream, pattern, monitor, stop):
    """Echo the marker stream and stamp the monitor on each regex match."""
    compiled = re.compile(pattern)
    for line in stream:
        if stop.is_set():
            break
        sys.stdout.write(line)
        sys.stdout.flush()
        if compiled.search(line):
            monitor.mark(time.clock_gettime(_RAW))


def _source_done(proc, pid):
    """(done, exit_code): child exited, or watched pid died, else (False, None)."""
    if proc is not None:
        code = proc.poll()
        return code is not None, code
    if pid is not None:
        return (not _pid_alive(pid)), None
    return False, None


def _terminate(proc, pid, grace, log):
    """SIGTERM the job, escalate to SIGKILL after *grace* (wrap or attach-pid)."""
    if proc is not None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
        _log(log, "SIGTERM->child (escalates to SIGKILL after grace)")
    elif pid is not None and _pid_alive(pid):
        _kill_pid(pid, grace)
        _log(log, f"SIGTERM->{pid} (escalates to SIGKILL after grace)")


def _kill_pid(pid, grace):
    """SIGTERM a foreign pid, then SIGKILL after *grace* if still alive."""
    os.kill(pid, signal.SIGTERM)
    time.sleep(grace)
    if _pid_alive(pid):
        os.kill(pid, signal.SIGKILL)


def _pid_alive(pid):
    return os.path.exists(f"/proc/{pid}")


# --- GPU polling -----------------------------------------------------------

def _open_gpu_safe(gpu_index):
    """Open the NVML handle, or None if NVML is unavailable (tests monkeypatch _poll)."""
    return _safe(lambda: open_gpu(gpu_index), None)


def _poll(handle):
    """One GPU snapshot: power, SM clock, duty-cycle (each None-safe on read error)."""
    fast = _safe(lambda: gpu_fast(handle), {})
    util = _safe(lambda: gpu_util(handle), {})
    return {"power_w": fast.get("power_w"), "sm_clock_mhz": fast.get("sm_clock_mhz"),
            "gpu_busy_pct": util.get("gpu_busy_pct")}


# --- output ----------------------------------------------------------------

def _result(stalled, reason, monitor, snap, action, exit_code):
    return {"stalled": stalled, "reason": reason, "marks": monitor.marks,
            "power_w": snap["power_w"], "sm_clock_mhz": snap["sm_clock_mhz"],
            "gpu_busy_pct": snap["gpu_busy_pct"], "action": action,
            "exit_code": exit_code}


def _stall_message(monitor, snap, now):
    power = snap["power_w"]
    clock = snap["sm_clock_mhz"]
    idle_for = round(now - monitor.idle_since, 1) if monitor.idle_since else None
    return (f"[{time.strftime('%H:%M:%S')}] STALLED — no progress for "
            f"{monitor.silent_s(now)}s and GPU idle for {idle_for}s "
            f"(power={power} W, clock={clock} MHz, marks={monitor.marks}). "
            f"Likely a hung kernel/compile, not slow progress.")


def _banner(cmd, logfile, pid, stall_s, poll_s):
    src = "wrap:" + " ".join(cmd) if cmd else f"attach:{logfile} pid={pid}"
    return (f"[{time.strftime('%H:%M:%S')}] watch up: {src} "
            f"stall>{stall_s}s poll={poll_s}s pid={os.getpid()}")


def _write_timeline(rows, record_dir):
    """Persist the poll timeline (proof of the stall) to <dir>/samples.parquet."""
    out = __import__("pathlib").Path(record_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows or [{}]).to_parquet(out / "samples.parquet")


def format_result(result):
    """One-line human summary of a watch verdict (for the CLI)."""
    if result["stalled"]:
        return (f"STALLED after {result['marks']} marks — GPU "
                f"{result['power_w']} W / {result['sm_clock_mhz']} MHz; action={result['action']}")
    return (f"OK — {result['reason']} ({result['marks']} marks, "
            f"exit_code={result['exit_code']})")


def _run_hook(cmd, msg, log):
    """Run a shell hook (notify/checkpoint), passing the alarm via FLIGHTREC_STALL_MSG."""
    env = dict(os.environ, FLIGHTREC_STALL_MSG=msg)
    try:
        subprocess.run(cmd, shell=True, env=env, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        _log(log, f"hook failed ({cmd!r}): {exc}")


def _log(log, message):
    print(message, file=log, flush=True)


def _safe(thunk, default):
    """Run a thunk; return *default* if it raises (NVML/read robustness)."""
    try:
        return thunk()
    except Exception:  # noqa: BLE001 - watchdog robustness over precision
        return default
