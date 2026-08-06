# File: test_sample_cpu.py
# Location: tests/test_sample_cpu.py
# Purpose: per-core busy math, /proc/stat parse, and robust sysfs int read.
# Dependencies: flightrec.sample_cpu

"""CPU readers: correct deltas and a bad read that returns 0 instead of crashing."""

from flightrec.sample_cpu import cpu_busy, read_stat, _read_int


def test_cpu_busy_half_load():
    prev = {0: (100, 200)}            # idle=100, total=200
    cur = {0: (150, 300)}            # +50 idle over +100 total -> 50% busy
    row = cpu_busy(prev, cur)
    assert row["c0_busy"] == 50.0
    assert row["n_active"] == 1


def test_read_stat_has_core_zero():
    stat = read_stat()
    assert 0 in stat
    assert len(stat[0]) == 2


def test_read_int_missing_path_is_zero():
    assert _read_int("/nonexistent/path/xyz") == 0
