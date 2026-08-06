# File: test_gate.py
# Location: tests/test_gate.py
# Purpose: replication_verdict + the submission gate (VALID + CI-tight + replicated) behave correctly.
# Dependencies: pandas, numpy, flightrec.compare, flightrec.gate

"""The gate must PASS only clean+tight+replicated results, and reject throttled/few/non-replicating ones."""

import json

import numpy as np
import pandas as pd

from flightrec.compare import replication_verdict
from flightrec.gate import gate


def _write(path, e1, clock=2300, power=60.0, busy=90.0, dur_ns=1_000_000_000):
    path.mkdir()
    pd.DataFrame({
        "t_ns": [0, dur_ns], "t_rel_s": [0.0, dur_ns / 1e9],
        "sm_clock_mhz": [clock, clock], "power_w": [power, power],
        "energy_mj": [0.0, e1], "gpu_busy_pct": [busy, busy], "n_active": [4, 4],
    }).to_parquet(path / "samples.parquet")
    pd.DataFrame([{"phase": "bench", "t0_ns": 0, "t1_ns": dur_ns}]).to_parquet(
        path / "phases.parquet")
    (path / "header.json").write_text(json.dumps({"bandwidth_wall_gbps": 220.0}))


def _runs(tmp_path, tag, energies, **kw):
    dirs = []
    for i, e in enumerate(energies):
        d = tmp_path / f"{tag}{i}"
        _write(d, e1=e, **kw)
        dirs.append(str(d))
    return dirs


def test_replication_verdict_agree_vs_diverge():
    rng = np.random.default_rng(0)
    a = list(rng.normal(100, 1, 40))
    b_same = list(rng.normal(100, 1, 40))
    b_diff = list(rng.normal(115, 1, 40))
    assert replication_verdict(a, b_same)["replicated"] is True       # boxes agree
    assert replication_verdict(a, b_diff)["replicated"] is False      # box-specific (15%)


def test_gate_pass_clean_tight(tmp_path):
    runs = _runs(tmp_path, "ok", [5000.0] * 6)        # 6 valid, identical -> CI half-width 0
    res = gate(runs, metric="energy_j", min_n=5, max_ci_halfwidth_pct=5.0)
    assert res["pass"] is True
    assert res["checks"] == {"enough_valid": True, "no_throttle": True, "ci_tight": True}


def test_gate_fails_on_throttled_run(tmp_path):
    runs = _runs(tmp_path, "ok", [5000.0] * 5)
    runs += _runs(tmp_path, "wedge", [9000.0], clock=611, power=13.0, busy=95.0)  # dropped -> no_throttle False
    res = gate(runs, metric="energy_j", min_n=5)
    assert res["pass"] is False
    assert res["checks"]["no_throttle"] is False


def test_gate_requires_replication_when_asked(tmp_path):
    a = _runs(tmp_path, "a", [5000.0] * 6)
    b_diff = _runs(tmp_path, "b", [9000.0] * 6)        # other box ~80% higher -> not replicated
    res = gate(a, replicate_with=b_diff, metric="energy_j", min_n=5)
    assert res["checks"]["replicated"] is False
    assert res["pass"] is False
