# File: test_sample.py
# Location: tests/test_sample.py
# Purpose: Tests for adaptive_sample (adaptive-N sampling).
# Dependencies: flightrec.sample, flightrec.recorder, json, pandas

"""adaptive_sample must converge when the CI tightens and respect the max_n hard cap."""

import json

import pandas as pd
import pytest

from flightrec.sample import adaptive_sample


# ---------------------------------------------------------------------------
# Shared fake-artifact helper (same schema as test_gate._write)
# ---------------------------------------------------------------------------

def _write_artifact(path, energy_mj=5000.0, clock=2300, power=60.0,
                    busy=90.0, dur_ns=1_000_000_000):
    """Write a minimal valid FlightRecorder artifact into *path* (must not exist yet)."""
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "t_ns": [0, dur_ns],
        "t_rel_s": [0.0, dur_ns / 1e9],
        "sm_clock_mhz": [clock, clock],
        "power_w": [power, power],
        "energy_mj": [0.0, energy_mj],
        "gpu_busy_pct": [busy, busy],
        "n_active": [4, 4],
    }).to_parquet(path / "samples.parquet")
    pd.DataFrame([{"phase": "bench", "t0_ns": 0, "t1_ns": dur_ns}]).to_parquet(
        path / "phases.parquet")
    (path / "header.json").write_text(json.dumps({"bandwidth_wall_gbps": 220.0}))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_adaptive_sample_smoke(tmp_path, monkeypatch):
    """Smoke test: adaptive_sample returns a dict with required keys and respects max_n."""
    calls = []

    def _fake_subprocess_call(cmd):
        calls.append(cmd)
        return 0

    def _fake_recorder(out_dir, hz=20):
        """Return a context manager that writes a valid artifact, not real NVML sampling."""
        class _Ctx:
            def __enter__(self_inner):
                from pathlib import Path
                _write_artifact(Path(out_dir))
                return self_inner

            def __exit__(self_inner, *_):
                pass

        return _Ctx()

    monkeypatch.setattr("flightrec.sample.subprocess.call", _fake_subprocess_call)
    monkeypatch.setattr("flightrec.sample.FlightRecorder", _fake_recorder)

    result = adaptive_sample(
        cmd=["true"],
        out_prefix=str(tmp_path / "run"),
        metric="energy_j",
        until_ci_pct=3.0,
        min_n=3,
        max_n=4,
    )

    assert "run_dirs" in result
    assert "n" in result
    assert "converged" in result
    assert "aggregate" in result
    assert result["n"] <= 4


def test_adaptive_sample_converges(tmp_path, monkeypatch):
    """With perfectly identical runs the CI collapses; convergence should trigger at min_n."""
    def _fake_subprocess_call(cmd):
        return 0

    def _fake_recorder(out_dir, hz=20):
        class _Ctx:
            def __enter__(self_inner):
                from pathlib import Path
                _write_artifact(Path(out_dir), energy_mj=5000.0)
                return self_inner

            def __exit__(self_inner, *_):
                pass

        return _Ctx()

    monkeypatch.setattr("flightrec.sample.subprocess.call", _fake_subprocess_call)
    monkeypatch.setattr("flightrec.sample.FlightRecorder", _fake_recorder)

    result = adaptive_sample(
        cmd=["true"],
        out_prefix=str(tmp_path / "conv"),
        metric="energy_j",
        until_ci_pct=3.0,
        min_n=5,
        max_n=20,
    )

    assert result["converged"] is True
    # Identical values -> CI half-width is zero -> converges at exactly min_n
    assert result["n"] == 5
    assert result["aggregate"]["n_valid"] == 5


def test_adaptive_sample_hits_max_n(tmp_path, monkeypatch):
    """Noisy runs that never converge must stop at max_n without raising."""
    import random
    rng = random.Random(42)

    def _fake_subprocess_call(cmd):
        return 0

    def _fake_recorder(out_dir, hz=20):
        energy = rng.uniform(3000.0, 7000.0)   # wide spread -> CI stays wide

        class _Ctx:
            def __enter__(self_inner):
                from pathlib import Path
                _write_artifact(Path(out_dir), energy_mj=energy)
                return self_inner

            def __exit__(self_inner, *_):
                pass

        return _Ctx()

    monkeypatch.setattr("flightrec.sample.subprocess.call", _fake_subprocess_call)
    monkeypatch.setattr("flightrec.sample.FlightRecorder", _fake_recorder)

    result = adaptive_sample(
        cmd=["true"],
        out_prefix=str(tmp_path / "noisy"),
        metric="energy_j",
        until_ci_pct=0.01,   # absurdly tight — will never converge
        min_n=3,
        max_n=6,
    )

    assert result["n"] == 6
    assert result["converged"] is False
