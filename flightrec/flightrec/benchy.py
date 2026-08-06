# File: benchy.py
# Location: flightrec/flightrec/benchy.py
# Purpose: Run a subprocess while tee-ing stdout and optionally parsing a float from its output.
# Dependencies: stdlib only (re, subprocess, sys)

"""Subprocess runner that echoes output live and optionally auto-parses a metric float.

Typical use: wrap a bench command and harvest tok/s from a summary line without
requiring the caller to pass ``--tokens`` by hand.

    exit_code, tok_s = run_capturing(cmd, pattern=r"throughput:\\s+([\\d.]+)\\s+tok/s")
"""

import re
import subprocess
import sys


def run_capturing(cmd: list, pattern: str | None) -> tuple:
    """Run *cmd*, echo each output line live, and optionally parse a float.

    Args:
        cmd: Command list passed to ``subprocess.Popen``.
        pattern: A regex string.  If *None*, no parsing is attempted.
            May contain one capture group — the first group is used as the value.
            When multiple lines match, the LAST match wins (benches often emit a
            summary after per-iteration lines).

    Returns:
        A ``(exit_code, float_or_None)`` tuple.  ``float_or_None`` is *None* when
        *pattern* is *None* or no line matched.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    exit_code = proc.wait()

    if pattern is None:
        return exit_code, None

    parsed = None
    compiled = re.compile(pattern)
    for line in lines:
        m = compiled.search(line)
        if m:
            raw = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
            try:
                parsed = float(raw)
            except ValueError:
                pass

    return exit_code, parsed
