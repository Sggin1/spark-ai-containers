# File: test_quiesce.py
# Location: tests/test_quiesce.py
# Purpose: quiesce.classify — resting-vitals verdict thresholds (pure, no GPU).
# Dependencies: flightrec.quiesce

"""The quiescence gate: foreign GPU/CPU load -> QUIET/CONTENDED verdict. Pure
threshold logic tested against synthetic signals; no NVML/GPU required."""

from flightrec import quiesce
from flightrec.quiesce import classify


def test_quiet_when_all_below_floors():
    v = classify(busy_pct=4.7, power_w=14.0, load_per_cpu=0.2)
    assert v["quiet"] is True
    assert v["verdict"] == "QUIET"
    assert v["reasons"] == ["box at rest"]
    assert v["gpu_busy_pct"] == 4.7  # measured signals echoed verbatim


def test_contended_on_gpu_busy():
    v = classify(busy_pct=40.0, power_w=14.0, load_per_cpu=0.1)
    assert v["quiet"] is False and v["verdict"] == "CONTENDED"
    assert any("busy" in r for r in v["reasons"])


def test_contended_on_gpu_power():
    v = classify(busy_pct=2.0, power_w=45.0, load_per_cpu=0.1)
    assert v["quiet"] is False
    assert any("W" in r for r in v["reasons"])


def test_contended_on_cpu_load():
    v = classify(busy_pct=2.0, power_w=14.0, load_per_cpu=1.5)
    assert v["quiet"] is False
    assert any("CPU" in r for r in v["reasons"])


def test_multiple_reasons_accumulate():
    v = classify(busy_pct=40.0, power_w=50.0, load_per_cpu=2.0)
    assert len(v["reasons"]) == 3


def test_floors_are_tunable():
    # a strict 1% busy floor flips an otherwise-quiet 4.7% reading to contended
    v = classify(busy_pct=4.7, power_w=14.0, load_per_cpu=0.1, busy_floor=1.0)
    assert v["quiet"] is False


def test_module_exposes_default_floors():
    assert quiesce.GPU_BUSY_FLOOR_PCT == 10.0
    assert quiesce.GPU_POWER_FLOOR_W == 25.0
