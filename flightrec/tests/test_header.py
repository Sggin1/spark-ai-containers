# File: test_header.py
# Location: tests/test_header.py
# Purpose: header picks the MEASURED wall when calibrated, else the modeled fallback.
# Dependencies: flightrec.header

"""The BW wall in the artifact must be measured-when-available, never a silent guess."""

from flightrec import header
from flightrec.roofline import MODELED_WALL_GBPS


def test_wall_uses_measured_when_calibrated(monkeypatch):
    monkeypatch.setattr(header, "load_calibration", lambda: {
        "bandwidth_wall_gbps_measured": 219.5,
        "method": "STREAM-triad", "pct_of_spec": 80.4, "measured_unix": 1.0,
    })
    wall = header._bandwidth_wall()
    assert wall["bandwidth_wall_gbps"] == 219.5
    assert wall["bandwidth_wall_source"].startswith("measured")
    assert wall["bandwidth_wall_pct_of_spec"] == 80.4


def test_wall_falls_back_to_modeled_when_uncalibrated(monkeypatch):
    monkeypatch.setattr(header, "load_calibration", lambda: None)
    wall = header._bandwidth_wall()
    assert wall["bandwidth_wall_gbps"] == MODELED_WALL_GBPS
    assert "modeled-constant" in wall["bandwidth_wall_source"]
    assert wall["bandwidth_wall_pct_of_spec"] is None


def test_compute_peak_from_calibration(monkeypatch):
    monkeypatch.setattr(header, "load_calibration", lambda: {
        "compute_peak_tflops": {"bf16": 31.4, "nvfp4": 482.0},
        "compute_peak_source": {"bf16": "matmul-bench", "nvfp4": "recorded"},
    })
    peak = header._compute_peak()
    assert peak["compute_peak_tflops"]["nvfp4"] == 482.0
    assert peak["compute_peak_source"]["bf16"] == "matmul-bench"


def test_compute_peak_none_when_uncalibrated(monkeypatch):
    monkeypatch.setattr(header, "load_calibration", lambda: None)
    peak = header._compute_peak()
    assert peak["compute_peak_tflops"] is None
    assert peak["compute_peak_source"] is None
