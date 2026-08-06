# File: test_watch.py
# Location: tests/test_watch.py
# Purpose: flightrec watch — StallMonitor conjunction logic, source plumbing, end-to-end stall detection.
# Dependencies: pandas, flightrec.watch

"""The watch arm: a long job is STALLED only when BOTH the progress marker has gone
silent AND the GPU has sat idle, for stall_s. The pure decision (StallMonitor) is
tested against synthetic time/GPU values; the driver is tested against a real
subprocess with _poll monkeypatched so no NVML/GPU is required."""

import sys
import time

from flightrec import watch as W


# ---------------------------------------------------------------------------
# StallMonitor — the conjunction (marker-silent AND gpu-idle, past warmup)
# ---------------------------------------------------------------------------

def _monitor(stall_s=10.0, warmup_s=0.0, start_t=1000.0):
    return W.StallMonitor(stall_s, idle_power_w=20.0, idle_busy_pct=10.0,
                          warmup_s=warmup_s, start_t=start_t)


def test_idle_and_silent_past_stall_is_stalled():
    m = _monitor(stall_s=10.0, start_t=0.0)
    m.observe_gpu(0.0, power_w=0.0, busy_pct=0.0)   # idle since t=0
    assert m.stalled(9.0) is False                  # not yet stall_s
    assert m.stalled(10.0) is True                  # both conditions exceed 10s


def test_busy_gpu_blocks_stall_even_when_marker_silent():
    # A long single kernel: no markers for ages, but the GPU is BUSY -> not a hang.
    m = _monitor(stall_s=10.0, start_t=0.0)
    m.observe_gpu(0.0, power_w=60.0, busy_pct=90.0)
    assert m.stalled(100.0) is False


def test_marker_advance_resets_the_clock():
    m = _monitor(stall_s=10.0, start_t=0.0)
    m.observe_gpu(0.0, power_w=0.0, busy_pct=0.0)   # idle whole time
    m.mark(8.0)                                     # progress at t=8 resets silence
    assert m.stalled(15.0) is False                 # only 7s since last marker
    assert m.stalled(18.0) is True                  # now 10s silent + idle


def test_warmup_suppresses_early_stall():
    m = _monitor(stall_s=5.0, warmup_s=60.0, start_t=0.0)
    m.observe_gpu(0.0, power_w=0.0, busy_pct=0.0)
    assert m.stalled(30.0) is False                 # inside warmup, never stalls
    assert m.stalled(61.0) is True                  # past warmup, conditions hold


def test_busy_poll_clears_idle_timer():
    m = _monitor(stall_s=10.0, start_t=0.0)
    m.observe_gpu(0.0, power_w=0.0, busy_pct=0.0)   # idle starts
    m.observe_gpu(5.0, power_w=55.0, busy_pct=80.0)  # busy -> idle_since cleared
    assert m.idle_since is None
    assert m.stalled(11.0) is False                 # idle timer restarted, not 10s yet


def test_read_failure_never_manufactures_idle():
    m = _monitor(stall_s=10.0, start_t=0.0)
    m.observe_gpu(0.0, power_w=None, busy_pct=None)  # NVML read failed
    assert m.idle_since is None
    assert m.stalled(100.0) is False


# ---------------------------------------------------------------------------
# source plumbing
# ---------------------------------------------------------------------------

def test_source_done_detects_child_exit():
    class _Proc:
        def __init__(self, code):
            self._code = code

        def poll(self):
            return self._code

    assert W._source_done(_Proc(None), None) == (False, None)
    assert W._source_done(_Proc(0), None) == (True, 0)


def test_source_done_attach_pid_liveness(monkeypatch):
    monkeypatch.setattr(W, "_pid_alive", lambda pid: False)
    assert W._source_done(None, 4242) == (True, None)


# ---------------------------------------------------------------------------
# end-to-end watch() — real subprocess, GPU poll monkeypatched
# ---------------------------------------------------------------------------

def _hang_cmd(n_marks=2, interval=0.02):
    """Emit a few markers, then sleep forever (the hang) so watch must catch it."""
    prog = (f"import time\n"
            f"for i in range({n_marks}):\n"
            f"    print('STEP', i, flush=True)\n"
            f"    time.sleep({interval})\n"
            f"time.sleep(3600)\n")
    return [sys.executable, "-c", prog]


def _short_cmd(n_marks=3, interval=0.01):
    prog = (f"import time\n"
            f"for i in range({n_marks}):\n"
            f"    print('STEP', i, flush=True)\n"
            f"    time.sleep({interval})\n")
    return [sys.executable, "-c", prog]


def test_watch_catches_hang_and_kills(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_open_gpu_safe", lambda idx: None)
    monkeypatch.setattr(W, "_poll", lambda h: {"power_w": 0.0, "sm_clock_mhz": 300,
                                               "gpu_busy_pct": 0.0})
    result = W.watch(cmd=_hang_cmd(), marker_re=r"STEP", stall_s=0.3, poll_s=0.05,
                     warmup_s=0.0, on_stall="kill", grace=1.0,
                     log_path=str(tmp_path / "w.log"), record_dir=str(tmp_path / "rec"))
    assert result["stalled"] is True
    assert result["action"] == "killed"
    assert result["marks"] == 2
    assert (tmp_path / "rec" / "samples.parquet").exists()   # proof timeline written


def test_watch_clean_job_is_not_stalled(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_open_gpu_safe", lambda idx: None)
    # GPU reads BUSY so even the brief gap after the last marker can't trip a stall.
    monkeypatch.setattr(W, "_poll", lambda h: {"power_w": 60.0, "sm_clock_mhz": 2300,
                                               "gpu_busy_pct": 90.0})
    result = W.watch(cmd=_short_cmd(), marker_re=r"STEP", stall_s=0.3, poll_s=0.02,
                     warmup_s=0.0, log_path=str(tmp_path / "w.log"))
    assert result["stalled"] is False
    assert result["reason"] == "job completed"
    assert result["exit_code"] == 0


def test_watch_hook_runs_on_stall(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "_open_gpu_safe", lambda idx: None)
    monkeypatch.setattr(W, "_poll", lambda h: {"power_w": 0.0, "sm_clock_mhz": 300,
                                               "gpu_busy_pct": 0.0})
    flag = tmp_path / "checkpointed"
    result = W.watch(cmd=_hang_cmd(), marker_re=r"STEP", stall_s=0.3, poll_s=0.05,
                     warmup_s=0.0, on_stall="kill", grace=1.0,
                     checkpoint_cmd=f"touch {flag}", log_path=str(tmp_path / "w.log"))
    assert result["stalled"] is True
    assert flag.exists()   # checkpoint hook fired before the kill


# ---------------------------------------------------------------------------
# format_result
# ---------------------------------------------------------------------------

def test_format_result_stalled():
    r = {"stalled": True, "marks": 16, "power_w": 0.0, "sm_clock_mhz": 300,
         "action": "killed", "reason": "stall detected", "exit_code": None}
    assert "STALLED" in W.format_result(r) and "killed" in W.format_result(r)


def test_format_result_ok():
    r = {"stalled": False, "marks": 100, "power_w": 60.0, "sm_clock_mhz": 2300,
         "action": "none", "reason": "job completed", "exit_code": 0}
    assert "OK" in W.format_result(r)
