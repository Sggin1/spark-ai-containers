# File: test_compute_cal.py
# Location: tests/test_compute_cal.py
# Purpose: compute-ceiling calibration — record/merge/read + the no-torch-FP4 guard.
# Dependencies: flightrec.compute_cal

"""Compute peak persists into the per-box calibration file and is auto-read by
``compute_peak_for``. The record path (peak passed in) needs no GPU; the matmul
bench is only reachable for bf16/fp16, which is asserted via the guard."""

import pytest

from flightrec import compute_cal as cc


def test_record_then_read(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    out = cc.calibrate_compute("nvfp4", peak_tflops=482.0, host="testbox")
    assert out["source"] == "recorded" and out["compute_peak_tflops"] == 482.0
    assert cc.compute_peak_for("nvfp4", host="testbox") == 482.0
    assert cc.compute_peak_for("NVFP4", host="testbox") == 482.0  # case-insensitive


def test_merge_preserves_other_dtypes(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cc.calibrate_compute("nvfp4", peak_tflops=482.0, host="testbox")
    cc.calibrate_compute("fp8", peak_tflops=250.0, host="testbox")
    assert cc.compute_peak_for("nvfp4", host="testbox") == 482.0  # not clobbered
    assert cc.compute_peak_for("fp8", host="testbox") == 250.0


def test_missing_peak_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert cc.compute_peak_for("bf16", host="absent") is None


def test_matmul_bench_rejects_unmeasurable_dtype():
    with pytest.raises(ValueError, match="nvfp4bench"):
        cc.measure_compute_peak("nvfp4")
