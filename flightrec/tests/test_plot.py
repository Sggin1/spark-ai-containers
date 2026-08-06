# File: test_plot.py
# Location: tests/test_plot.py
# Purpose: flightrec plot — adaptive lane selection + energy-rate derivation (pure),
#          and a matplotlib-guarded end-to-end render of the timeseries figure.
# Dependencies: pandas, flightrec.plot (matplotlib optional)

"""The plot arm: the lane set adapts to whatever columns an artifact carries, and the
GPU-power lane overlays the energy-integral rate. The selection + derivation logic is
tested without matplotlib; the render is skipped when the optional `plot` extra is absent."""

import json

import pandas as pd
import pytest

from flightrec import plot as P


def _samples(extra=None):
    """A minimal sample frame; `extra` adds richer-artifact columns."""
    base = {
        "t_ns": [0, 500_000_000, 1_000_000_000],
        "t_rel_s": [0.0, 0.5, 1.0],
        "power_w": [60.0, 60.0, 60.0],
        "sm_clock_mhz": [2300, 2300, 2300],
        "n_active": [4, 4, 4],
        "gpu_busy_pct": [90.0, 90.0, 90.0],
        "energy_mj": [0.0, 30000.0, 60000.0],
    }
    base.update(extra or {})
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# present_lanes — adaptive to columns actually in the artifact
# ---------------------------------------------------------------------------

def test_old_artifact_yields_core_lanes_only():
    lanes = P.present_lanes(_samples())
    cols = [c for c, _ in lanes]
    assert cols == ["power_w", "sm_clock_mhz", "n_active", "gpu_busy_pct"]


def test_new_artifact_adds_jitter_psi_freq_lanes():
    rich = _samples(extra={
        "actual_dt_ms": [50.0, 50.1, 49.9],
        "psi_cpu_some10": [1.0, 2.0, 1.5],
        "cpu_freq_mean_mhz": [2800, 2800, 2800],
    })
    cols = [c for c, _ in P.present_lanes(rich)]
    assert "actual_dt_ms" in cols and "psi_cpu_some10" in cols and "cpu_freq_mean_mhz" in cols
    assert cols[0] == "power_w"   # power stays first (carries the overlay)


def test_all_nan_column_is_dropped():
    df = _samples(extra={"cpu_freq_mean_mhz": [None, None, None]})
    assert "cpu_freq_mean_mhz" not in [c for c, _ in P.present_lanes(df)]


# ---------------------------------------------------------------------------
# energy_rate_series — d(energy_mj)/dt in watts
# ---------------------------------------------------------------------------

def test_energy_rate_matches_constant_power():
    # 60 J over 1 s in two 0.5 s steps -> 60 W rate at each differenced point.
    times, watts = P.energy_rate_series(_samples())
    assert list(round(w, 1) for w in watts) == [60.0, 60.0]
    assert len(times) == 2   # leading point (no predecessor) dropped


def test_energy_rate_none_without_energy():
    df = _samples().drop(columns=["energy_mj"])
    assert P.energy_rate_series(df) is None


def test_energy_rate_none_with_single_sample():
    df = pd.DataFrame({"t_rel_s": [0.0], "energy_mj": [0.0]})
    assert P.energy_rate_series(df) is None


# ---------------------------------------------------------------------------
# end-to-end render (skipped without matplotlib)
# ---------------------------------------------------------------------------

def _write_run(path, df):
    path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path / "samples.parquet")
    pd.DataFrame([{"phase": "bench", "t0_ns": 0, "t1_ns": 1_000_000_000}]).to_parquet(
        path / "phases.parquet")
    (path / "header.json").write_text(json.dumps({"bandwidth_wall_gbps": 220.0}))


def test_timeseries_renders_png(tmp_path):
    pytest.importorskip("matplotlib")
    run = tmp_path / "run"
    _write_run(run, _samples(extra={
        "actual_dt_ms": [50.0, 50.1, 49.9],
        "psi_cpu_some10": [1.0, 2.0, 1.5],
        "cpu_freq_mean_mhz": [2800, 2800, 2800],
    }))
    out = P.timeseries(str(run))
    assert out.endswith("timeseries.png")
    assert (run / "timeseries.png").stat().st_size > 0


def test_timeseries_renders_old_artifact(tmp_path):
    pytest.importorskip("matplotlib")
    run = tmp_path / "old"
    _write_run(run, _samples())   # only the 4 core lanes present
    out = P.timeseries(str(run), out_png=str(tmp_path / "ts.png"))
    assert (tmp_path / "ts.png").stat().st_size > 0


# ---------------------------------------------------------------------------
# C3 · multi-run overlay + generalized violins
# ---------------------------------------------------------------------------

def test_resolve_labels_defaults_to_basename():
    assert P.resolve_labels(["/a/b/run_001", "/a/b/run_002"], None) == ["run_001", "run_002"]


def test_resolve_labels_uses_explicit():
    assert P.resolve_labels(["/x/r1", "/x/r2"], ["BF16", "NVFP4"]) == ["BF16", "NVFP4"]


def test_timeseries_overlay_renders(tmp_path):
    pytest.importorskip("matplotlib")
    a, b = tmp_path / "bf16", tmp_path / "nvfp4"
    _write_run(a, _samples())
    _write_run(b, _samples(extra={"power_w": [40.0, 40.0, 40.0]}))
    out = P.timeseries_overlay([str(a), str(b)], labels=["BF16", "NVFP4"],
                               out_png=str(tmp_path / "ov.png"))
    assert (tmp_path / "ov.png").stat().st_size > 0


def test_metric_violins_renders_multi_panel(tmp_path):
    pytest.importorskip("matplotlib")
    metrics = {
        "run time (s)": {"BF16": [1.0, 1.1, 0.9], "NVFP4": [0.5, 0.55, 0.48]},
        "J/token": {"BF16": [1.3, 1.32, 1.29], "NVFP4": [0.68, 0.69, 0.67]},
    }
    out = P.metric_violins(metrics, out_png=str(tmp_path / "v.png"))
    assert (tmp_path / "v.png").stat().st_size > 0


def test_runtime_violin_still_works(tmp_path):
    pytest.importorskip("matplotlib")
    out = P.runtime_violin({"A": [1.0, 1.1], "B": [2.0, 2.1]}, out_png=str(tmp_path / "rv.png"))
    assert (tmp_path / "rv.png").stat().st_size > 0
