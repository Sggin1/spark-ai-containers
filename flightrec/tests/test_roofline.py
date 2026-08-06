# File: test_roofline.py
# Location: tests/test_roofline.py
# Purpose: roofline_point math + regime hint + calibration round-trip behave correctly.
# Dependencies: flightrec.roofline

"""Roofline coordinates are pure data; calibration persists per-box and reloads."""

import json

from flightrec import roofline
from flightrec.roofline import roofline_point, load_calibration, calibration_path


def test_roofline_point_bandwidth_math():
    # 30 GB moved in 0.1 s -> 300 GB/s; against a 200 GB/s wall -> 150% of wall.
    point = roofline_point(bytes_moved=30e9, flops=60e9, kernel_s=0.1, wall_gbps=200.0)
    assert point["achieved_gbps"] == 300.0
    assert point["achieved_gflops"] == 600.0
    assert point["arithmetic_intensity"] == 2.0  # 60e9 / 30e9
    assert point["pct_of_wall"] == 150.0


def test_roofline_point_bandwidth_regimes():
    near = roofline_point(190e9, 1, 1.0, 200.0)  # 95% of wall
    far = roofline_point(20e9, 1, 1.0, 200.0)  # 10% of wall
    mid = roofline_point(100e9, 1, 1.0, 200.0)  # 50% of wall
    # one-axis labels: only near-wall is a bound-claim; the rest are neutral positions
    assert near["bandwidth_regime"] == "bandwidth-bound"
    assert far["bandwidth_regime"] == "far-from-wall"
    assert mid["bandwidth_regime"] == "mid-wall"


def test_two_axis_regime_needs_peak():
    # without peak_tflops the compute axis is unknown -> regime None (never guessed)
    point = roofline_point(100e9, 50e12, 1.0, 200.0)
    assert point["pct_of_peak"] is None
    assert point["regime"] is None


def test_two_axis_regime_compute_bound():
    # 90% of wall would normally read bandwidth-bound, but flip: low wall, near peak.
    # 50 GB/s vs 200 wall = 25%; 360 TFLOP/s vs 400 peak = 90% -> compute-bound.
    point = roofline_point(50e9, 360e12, 1.0, 200.0, peak_tflops=400.0)
    assert point["regime"] == "compute-bound"
    assert point["pct_of_peak"] == 90.0


def test_two_axis_regime_overhead_bound_diffusiongemma():
    # the real DiffusionGemma read: ~20 TFLOP/s of a 365 peak (5.5%) AND far below
    # the wall -> overhead/latency-bound, the call the bandwidth axis alone can't make.
    point = roofline_point(40e9, 20e12, 1.0, 224.4, peak_tflops=365.0)
    assert point["regime"] == "overhead/latency-bound"
    assert point["bandwidth_regime"] == "far-from-wall"  # one-axis would only say "far"


def test_two_axis_regime_bandwidth_wins_at_wall():
    # near the wall is bandwidth-bound regardless of compute headroom
    point = roofline_point(190e9, 1e12, 1.0, 200.0, peak_tflops=400.0)
    assert point["regime"] == "bandwidth-bound"


def test_roofline_point_rejects_nonpositive():
    assert "error" in roofline_point(1, 1, 0.0, 200.0)
    assert "error" in roofline_point(0, 1, 1.0, 200.0)


def test_calibration_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    path = calibration_path(host="testbox")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"bandwidth_wall_gbps_measured": 219.5}))
    assert load_calibration(host="testbox")["bandwidth_wall_gbps_measured"] == 219.5
    assert load_calibration(host="absent-box") is None


def test_calibration_path_is_per_host(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert "calibration-node-a.json" in str(roofline.calibration_path(host="node-a"))
    assert "calibration-node-b.json" in str(roofline.calibration_path(host="node-b"))
