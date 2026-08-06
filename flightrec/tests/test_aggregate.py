# File: test_aggregate.py
# Location: tests/test_aggregate.py
# Purpose: aggregate() summarizes a metric over N runs and DROPS invalid (throttled) runs.
# Dependencies: pandas, flightrec.aggregate

"""N-run aggregation: median/IQR/CV + bootstrap CI, with INVALID runs excluded per protocol."""

import json

import pandas as pd

from flightrec.aggregate import aggregate


def _write(path, clock, power, busy, e0, e1, dur_ns=1_000_000_000):
    path.mkdir()
    pd.DataFrame({
        "t_ns": [0, dur_ns], "t_rel_s": [0.0, dur_ns / 1e9],
        "sm_clock_mhz": [clock, clock], "power_w": [power, power],
        "energy_mj": [e0, e1], "gpu_busy_pct": [busy, busy], "n_active": [4, 4],
    }).to_parquet(path / "samples.parquet")
    pd.DataFrame([{"phase": "bench", "t0_ns": 0, "t1_ns": dur_ns}]).to_parquet(
        path / "phases.parquet")
    (path / "header.json").write_text(json.dumps({"bandwidth_wall_gbps": 220.0}))


def test_aggregate_drops_invalid_and_summarizes(tmp_path):
    dirs = []
    for i, mj in enumerate((5000.0, 6000.0, 5500.0)):       # 3 healthy runs -> 5, 6, 5.5 J
        d = tmp_path / f"ok{i}"
        _write(d, clock=2300, power=60.0, busy=90.0, e0=0.0, e1=mj)
        dirs.append(str(d))
    wedge = tmp_path / "wedge"                                # throttled -> must be dropped
    _write(wedge, clock=611, power=13.0, busy=95.0, e0=0.0, e1=9000.0)
    dirs.append(str(wedge))

    summary = aggregate(dirs, metric="energy_j")
    assert summary["n_total"] == 4
    assert summary["n_valid"] == 3
    assert summary["n_invalid_dropped"] == 1
    assert summary["median"] == 5.5
    assert len(summary["median_ci95"]) == 2
    assert summary["median_ci95"][0] <= 5.5 <= summary["median_ci95"][1]
