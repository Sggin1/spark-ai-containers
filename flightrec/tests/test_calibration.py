# File: test_calibration.py
# Location: tests/test_calibration.py
# Purpose: roofline_predict picks the right bound; calibration_report scores error + prune verdict.
# Dependencies: flightrec.roofline, flightrec.calibration

"""Predict-vs-measured: the predictor's ideal bound, and whether predictions rank well enough to prune."""

from flightrec.roofline import roofline_predict
from flightrec.calibration import calibration_report


def test_predict_bandwidth_bound():
    p = roofline_predict(bytes_moved=220e9, flops=31e12, wall_gbps=220.0, peak_tflops=310.0)
    assert p["predicted_kernel_s"] == 1.0          # t_bw=1.0 > t_cmp=0.1
    assert p["predicted_bound"] == "bandwidth"
    assert p["predicted_pct_of_wall"] == 100.0     # at the wall
    assert p["predicted_pct_of_peak"] == 10.0      # 31/310


def test_predict_compute_bound():
    p = roofline_predict(bytes_moved=22e9, flops=310e12, wall_gbps=220.0, peak_tflops=310.0)
    assert p["predicted_kernel_s"] == 1.0          # t_cmp=1.0 > t_bw=0.1
    assert p["predicted_bound"] == "compute"
    assert p["predicted_pct_of_peak"] == 100.0
    assert p["predicted_pct_of_wall"] == 10.0


def test_calibration_perfect():
    r = calibration_report([1, 2, 3, 4], [1, 2, 3, 4])
    assert r["mae_pct"] == 0.0
    assert r["spearman"] == 1.0
    assert r["correction_factor"] == 1.0
    assert r["prunable"] is True


def test_calibration_optimistic_but_rank_preserving():
    # predictions consistently 2x too fast -> big bias, but ranking intact -> prunable + de-bias factor 2
    r = calibration_report([0.5, 1.0, 1.5, 2.0], [1, 2, 3, 4])
    assert r["bias_pct"] == -50.0
    assert r["correction_factor"] == 2.0
    assert r["spearman"] == 1.0
    assert r["prunable"] is True


def test_calibration_anticorrelated_not_prunable():
    r = calibration_report([4, 3, 2, 1], [1, 2, 3, 4])
    assert r["spearman"] == -1.0
    assert r["prunable"] is False
