# File: test_measure.py
# Location: tests/test_measure.py
# Purpose: measure_artifact() yields a correct, validity-gated utilization vector.
# Dependencies: pandas, flightrec.measure

"""The measure arm: real validity (from samples, not the static ceiling), phase
kernel time, roofline %-of-wall, energy, and tok/s + J/token when tokens given."""

import json

import pandas as pd

from flightrec.measure import measure_artifact


def _write(path, clock=2300, power=60.0, busy=90.0, e0=0.0, e1=5000.0, dur_ns=1_000_000_000):
    path.mkdir()
    pd.DataFrame({
        "t_ns": [0, dur_ns], "t_rel_s": [0.0, dur_ns / 1e9],
        "sm_clock_mhz": [clock, clock], "power_w": [power, power],
        "energy_mj": [e0, e1], "gpu_busy_pct": [busy, busy], "n_active": [4, 4],
    }).to_parquet(path / "samples.parquet")
    pd.DataFrame([{"phase": "bench", "t0_ns": 0, "t1_ns": dur_ns}]).to_parquet(
        path / "phases.parquet")
    (path / "header.json").write_text(json.dumps({
        "bandwidth_wall_gbps": 220.0, "bandwidth_wall_source": "measured (test)"}))


def test_healthy_run_full_vector(tmp_path):
    run = tmp_path / "ok"
    _write(run)
    # 220 GB/s for 1.0 s = exactly the wall; 100 tokens; 5 J energy delta.
    vec = measure_artifact(str(run), bytes_moved=220e9, flops=440e9, tokens=100)
    assert vec["valid"] is True
    assert vec["kernel_s"] == 1.0
    assert vec["energy_j"] == 5.0                 # (5000 - 0) mJ
    assert vec["pct_of_wall"] == 100.0            # 220 GB/s vs 220 wall
    assert vec["achieved_gflops"] == 440.0        # 440e9 flops / 1.0 s (compute axis)
    assert vec["arithmetic_intensity"] == 2.0     # 440e9 / 220e9
    assert vec["tok_s"] == 100.0                  # 100 tokens / 1.0 s
    assert vec["j_token"] == 0.05                 # 5 J / 100 tokens


def test_wedge_run_is_invalid(tmp_path):
    run = tmp_path / "wedge"
    _write(run, clock=611, power=13.0, busy=95.0)  # PD-wedge signature, under load
    vec = measure_artifact(str(run), bytes_moved=220e9)
    assert vec["valid"] is False                    # caught from samples, not the ceiling
    assert vec["throttled_pct_of_loaded"] > 0


def test_no_bytes_no_tokens_leaves_optional_fields_none(tmp_path):
    run = tmp_path / "bare"
    _write(run)
    vec = measure_artifact(str(run))
    assert vec["valid"] is True
    assert vec["kernel_s"] == 1.0
    assert vec["pct_of_wall"] is None
    assert vec["arithmetic_intensity"] is None
    assert vec["tok_s"] is None
    assert vec["j_token"] is None
