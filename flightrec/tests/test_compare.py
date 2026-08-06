# File: test_compare.py
# Location: tests/test_compare.py
# Purpose: envelope() extracts correct hardware-state + energy from an artifact.
# Dependencies: pandas, flightrec.compare

"""Direct-compare envelope: energy integral, validity, and clock stats are correct."""

import json

import pandas as pd

from flightrec.compare import envelope, replication_over_runs


def _write_artifact(path, clock, power, dur_s=1.0):
    path.mkdir()
    pd.DataFrame({
        "t_ns": [0, 1], "t_rel_s": [0.0, dur_s],
        "sm_clock_mhz": [clock, clock], "power_w": [power, power],
        "energy_mj": [0.0, 1000.0], "temp_c": [50, 50],
        "n_active": [3, 3], "gpu_busy_pct": [90.0, 90.0],
        "cpu_freq_mean_mhz": [3000, 3000],
    }).to_parquet(path / "samples.parquet")
    pd.DataFrame([{"phase": "command", "t0_ns": 0, "t1_ns": int(dur_s * 1e9)}]).to_parquet(
        path / "phases.parquet")
    (path / "header.json").write_text(json.dumps({"gpu_name": "GB10", "driver": "580"}))


def test_envelope_energy_validity_clock(tmp_path):
    run = tmp_path / "runA"
    _write_artifact(run, 2300, 60.0)
    env = envelope(str(run))
    assert env["energy_j"] == 1.0                      # (1000 - 0) mJ -> 1 J
    assert env["verdict"]["valid"] is True             # healthy under load
    assert env["stats"]["sm_clock_mhz"] == (2300, 2300)


def _box(tmp_path, name, durations):
    dirs = []
    for i, dur in enumerate(durations):
        run = tmp_path / f"{name}_{i:03d}"
        _write_artifact(run, 2300, 60.0, dur_s=dur)
        dirs.append(str(run))
    return dirs


def test_replication_over_runs_replicated_when_boxes_agree(tmp_path):
    # Two boxes with the same kernel_s distribution -> REPLICATED.
    a = _box(tmp_path, "node-a", [1.00, 1.01, 0.99, 1.00, 1.01])
    b = _box(tmp_path, "node-b", [1.00, 0.99, 1.01, 1.00, 0.99])
    out = replication_over_runs(a, b, metric="kernel_s")
    assert out["replicated"] is True
    assert out["n_a"] == 5 and out["n_b"] == 5


def test_replication_over_runs_fails_on_box_specific_gap(tmp_path):
    # Box B ~30% slower -> significant AND practical gap -> NOT replicated.
    a = _box(tmp_path, "node-a", [1.00, 1.01, 0.99, 1.00, 1.01])
    b = _box(tmp_path, "node-b", [1.30, 1.31, 1.29, 1.30, 1.31])
    out = replication_over_runs(a, b, metric="kernel_s")
    assert out["replicated"] is False
