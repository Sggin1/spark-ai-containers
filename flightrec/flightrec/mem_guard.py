# File: mem_guard.py
# Location: flightrec/mem_guard.py
# Purpose: Global OOM backstop — kill runaway model/compute procs before unified memory wedges the box.
# Dependencies: stdlib only (/proc/meminfo, signal, subprocess for docker)

"""Memory guard daemon — the box-safety backstop.

Polls ``MemAvailable`` and, when it drops below a kill threshold, terminates
runaway model/compute processes (and stops model docker containers) BEFORE the
GB10 unified-memory pool exhausts and freezes the box. Born from repeated OOM
freezes (2026-06-08): an uncapped vLLM, or a ballooning embed/rerank stage,
fills the shared 128 GB pool and the kernel reaps the desktop session.

Two design facts this guard respects:
  * RSS undercounts GPU/unified allocations on GB10 (a 40 GB consumer can show
    3 GB RSS), so victims are chosen by COMMAND PATTERN, not by RSS.
  * A poller cannot react as fast as a kernel cgroup limit — pair this GLOBAL
    net (catches manual / hub launches) with per-launch
    ``systemd-run --scope -p MemoryMax=...`` for jobs we own.

Opt-in. Run ``python -m flightrec.mem_guard`` or enable the systemd unit.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import time

# Never kill these — desktop/session/infra and the guard itself.
_PROTECT = re.compile(
    r"gnome|Xorg|/code|claude|sshd|systemd|dbus|NetworkManager|gdm|"
    r"pipewire|wireplumber|ai-model-hub|mem_guard"
)
# Kill these on breach — model servers + heavy compute jobs (not bash wrappers).
_TARGET = re.compile(
    r"vllm|EngineCore|llama-server|sglang|"
    r"python[0-9.]*\s+\S*(_engine/|certify\.py|build_index\.py|semantic\.py|bakeoff)"
)


def _meminfo_gb(key: str, default: float) -> float:
    """Read a /proc/meminfo field (kB) and return it in GB."""
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith(key):
                return int(line.split()[1]) / 1048576
    return default


def mem_available_gb() -> float:
    """Available memory in GB (the trustworthy signal on GB10 — NVML mem is N/A)."""
    return _meminfo_gb("MemAvailable:", float("inf"))


def mem_total_gb() -> float:
    """Total physical memory in GB (to derive used = total - available)."""
    return _meminfo_gb("MemTotal:", 0.0)


def _victims() -> list[tuple[int, str]]:
    """(pid, cmdline) of killable model/compute procs, excluding protected + self."""
    me = os.getpid()
    out: list[tuple[int, str]] = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ").strip()
        except OSError:
            continue
        if cmd and _TARGET.search(cmd) and not _PROTECT.search(cmd):
            out.append((int(pid), cmd))
    return out


def _stop_model_containers(log) -> None:
    """docker stop non-registry containers (model servers hold GPU/unified mem)."""
    try:
        ids = subprocess.run(["docker", "ps", "-q"], capture_output=True,
                             text=True, timeout=5).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return
    for cid in ids:
        name = subprocess.run(["docker", "inspect", "-f", "{{.Name}}", cid],
                              capture_output=True, text=True, timeout=5).stdout
        if "registry" in name:
            continue
        subprocess.run(["docker", "stop", "-t", "5", cid], timeout=30)
        print(f"  docker stop {cid} ({name.strip()})", file=log, flush=True)


def enforce(kill_gb: float, log) -> bool:
    """Kill model/compute targets if memory is below kill_gb. True if it acted."""
    avail = mem_available_gb()
    if avail >= kill_gb:
        return False
    print(f"[{time.strftime('%H:%M:%S')}] BREACH avail={avail:.1f}GB < {kill_gb}GB "
          f"— killing runaways", file=log, flush=True)
    for pid, cmd in _victims():
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"  SIGKILL {pid}: {cmd[:90]}", file=log, flush=True)
        except OSError:
            pass
    _stop_model_containers(log)
    return True


def run(kill_gb: float, warn_used_gb: float, interval: float, log_path: str) -> None:
    """Poll memory; warn at a used-envelope, kill runaways below an avail floor.

    Two graded levels, by design:
      * kill_gb is an ABSOLUTE avail floor (workload-independent safety). Below
        it, runaways are killed.
      * warn_used_gb is a WORKLOAD-RELATIVE used envelope: set it ~20-30 GB above
        the run's expected footprint (e.g. omni 50 + vectors 20 = 70 -> warn 90)
        so an overrun is flagged WELL before the kill floor.

    Args:
        kill_gb: kill targets when MemAvailable drops below this (GB).
        warn_used_gb: log a warning when MemUsed (total - available) exceeds this.
        interval: seconds between polls. Default 0.05s (20 Hz) matches the
            recorder's fast path — balloons move several GB/s, so a 1s poll
            races a freeze; 50ms catches a breach with tens of GB still free.
        log_path: file to append guard events to.
    """
    total = mem_total_gb()
    with open(log_path, "a") as log:
        print(f"[{time.strftime('%H:%M:%S')}] guard up: kill<{kill_gb}GB-avail "
              f"warn>{warn_used_gb}GB-used (total={total:.0f}) "
              f"poll={interval}s ({1 / interval:.0f}Hz) pid={os.getpid()}",
              file=log, flush=True)
        warned = False
        while True:
            avail = mem_available_gb()
            used = total - avail
            if avail < kill_gb:
                enforce(kill_gb, log)
                time.sleep(3.0)  # cooldown: let teardown free memory before re-polling
            elif used > warn_used_gb and not warned:
                print(f"[{time.strftime('%H:%M:%S')}] WARN used={used:.1f}GB "
                      f"> {warn_used_gb}GB (avail={avail:.1f}GB)", file=log, flush=True)
                warned = True
            elif used <= warn_used_gb:
                warned = False
            time.sleep(interval)


def main() -> None:
    """CLI entry: parse thresholds and run the guard loop."""
    ap = argparse.ArgumentParser(description="flightrec memory guard (OOM backstop)")
    ap.add_argument("--kill-gb", type=float, default=20.0,
                    help="ABSOLUTE avail floor: kill runaways below this MemAvailable (GB)")
    ap.add_argument("--warn-used-gb", type=float, default=90.0,
                    help="workload envelope: warn when MemUsed exceeds this (GB). "
                         "Set ~20-30GB above the run's expected footprint.")
    ap.add_argument("--interval", type=float, default=0.05,
                    help="poll seconds (default 0.05 = 20 Hz, matches recorder fast path)")
    ap.add_argument("--log", default="/tmp/flightrec_memguard.log")
    a = ap.parse_args()
    run(a.kill_gb, a.warn_used_gb, a.interval, a.log)


if __name__ == "__main__":
    main()
