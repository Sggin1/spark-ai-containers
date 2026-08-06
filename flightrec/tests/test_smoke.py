# File: test_smoke.py
# Location: tests/test_smoke.py
# Purpose: flightrec smoke — slice-and-kill driver, per-unit/ETA math, and heuristic verdicts.
# Dependencies: pandas, flightrec.smoke

"""The smoke arm: record a short slice, kill it, map the utilization vector onto a
verdict + recommendations + full-run ETA. Driver tested against a real subprocess;
the heuristic + math tested against synthetic frames so no NVML/GPU is required."""

import json
import sys

import pandas as pd

from flightrec import smoke as S

# ---------------------------------------------------------------------------
# Synthetic artifact (same schema as test_measure / test_sample helpers)
# ---------------------------------------------------------------------------


def _write_artifact(
    path,
    clock=2300,
    power=60.0,
    busy=90.0,
    mem_avail_mb=80000.0,
    e0=0.0,
    e1=5000.0,
    dur_ns=1_000_000_000,
):
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "t_ns": [0, dur_ns],
            "t_rel_s": [0.0, dur_ns / 1e9],
            "sm_clock_mhz": [clock, clock],
            "power_w": [power, power],
            "energy_mj": [e0, e1],
            "gpu_busy_pct": [busy, busy],
            "mem_avail_mb": [mem_avail_mb, mem_avail_mb],
            "n_active": [4, 4],
        }
    ).to_parquet(path / "samples.parquet")
    pd.DataFrame([{"phase": "smoke", "t0_ns": 0, "t1_ns": dur_ns}]).to_parquet(
        path / "phases.parquet"
    )
    (path / "header.json").write_text(json.dumps({"bandwidth_wall_gbps": 220.0}))


# ---------------------------------------------------------------------------
# run_until_marks — the slice-and-kill driver
# ---------------------------------------------------------------------------


def _emit_cmd(n=100, interval=0.02):
    """A subprocess that prints UNIT markers forever-ish (so we must kill it)."""
    prog = (
        f"import time\n"
        f"for i in range({n}):\n"
        f"    print('UNIT', i, flush=True)\n"
        f"    time.sleep({interval})\n"
    )
    return [sys.executable, "-c", prog]


def test_run_until_marks_kills_after_k():
    result = S.run_until_marks(_emit_cmd(), r"UNIT", k_units=3, grace=2.0)
    assert result["reached_units"] == 3
    assert result["killed"] is True
    assert len(result["mark_times"]) == 3


def test_run_until_marks_short_job_runs_to_exit():
    # Only 2 markers ever emitted, but we asked for 5 -> process exits on its own.
    result = S.run_until_marks(_emit_cmd(n=2, interval=0.0), r"UNIT", k_units=5, grace=2.0)
    assert result["reached_units"] == 2
    assert result["killed"] is False


# ---------------------------------------------------------------------------
# per-unit timing + ETA
# ---------------------------------------------------------------------------


def test_per_unit_seconds_drops_warmup():
    # gaps: 5 (warmup, dropped), 2, 2, 2 -> median 2.0
    marks = [0.0, 5.0, 7.0, 9.0, 11.0]
    assert S.per_unit_seconds(marks, settle=1) == 2.0


def test_per_unit_seconds_too_few_marks_is_none():
    assert S.per_unit_seconds([1.0], settle=1) is None


def test_eta_extrapolates():
    e = S.eta(per_unit_s=2.0, total_units=1800)
    assert e["eta_s"] == 3600.0
    assert e["eta_human"] == "1h00m"


def test_eta_none_without_total():
    e = S.eta(per_unit_s=2.0, total_units=None)
    assert e["eta_s"] is None
    assert e["per_unit_s"] == 2.0


# ---------------------------------------------------------------------------
# signals — load-scoped
# ---------------------------------------------------------------------------


def test_signals_scopes_to_loaded(tmp_path):
    run = tmp_path / "sig"
    _write_artifact(run, power=55.0, clock=2200, busy=90.0, mem_avail_mb=70000.0)
    samples = pd.read_parquet(run / "samples.parquet")
    phases = pd.read_parquet(run / "phases.parquet")
    sig = S.signals(samples, phases)
    assert sig["power_median_w"] == 55.0
    assert sig["sm_clock_median_mhz"] == 2200
    assert sig["mem_avail_min_mb"] == 70000.0
    assert sig["loaded_frac"] == 1.0


# ---------------------------------------------------------------------------
# diagnose — the heuristic mapping (first matching rule wins)
# ---------------------------------------------------------------------------


def _sig(power=60.0, clock=2300, busy=90.0, mem=80000.0, loaded=1.0):
    return {
        "power_median_w": power,
        "sm_clock_median_mhz": clock,
        "gpu_busy_median_pct": busy,
        "mem_avail_min_mb": mem,
        "loaded_frac": loaded,
    }


def test_diagnose_throttled_first():
    verdict, recs = S.diagnose({"valid": False}, _sig(power=13.0, clock=611))
    assert verdict == "THROTTLED"
    assert any("INVALID" in r for r in recs)


def test_diagnose_inconclusive_when_idle():
    verdict, _ = S.diagnose({"valid": True}, _sig(loaded=0.2))
    assert verdict == "INCONCLUSIVE"


def test_diagnose_gpu_starved():
    verdict, recs = S.diagnose({"valid": True}, _sig(power=15.0, busy=30.0))
    assert verdict == "GPU-STARVED"
    assert any("low_gpu_mem_usage=False" in r for r in recs)


def test_diagnose_near_oom():
    verdict, recs = S.diagnose({"valid": True}, _sig(power=60.0, busy=70.0, mem=4096.0))
    assert verdict == "NEAR-OOM"
    assert any("batch_size" in r for r in recs)


def test_diagnose_gpu_busy_with_headroom():
    verdict, recs = S.diagnose({"valid": True}, _sig(power=60.0, busy=92.0, mem=80000.0))
    assert verdict == "GPU-BUSY"
    assert any("raise batch_size" in r for r in recs)  # headroom hint appended
    assert any("NOT a confirmed compute-bound" in r for r in recs)  # honest about duty != compute


def test_diagnose_intermediate():
    verdict, _ = S.diagnose({"valid": True}, _sig(power=45.0, busy=55.0, mem=30000.0))
    assert verdict == "INTERMEDIATE"


# ---------------------------------------------------------------------------
# end-to-end smoke() — recorder + driver monkeypatched (no NVML, no real bench)
# ---------------------------------------------------------------------------


def test_smoke_end_to_end(tmp_path, monkeypatch):
    out = tmp_path / "slice"

    def _fake_recorder(out_dir, hz=20):
        class _Ctx:
            def __enter__(self_inner):
                _write_artifact(__import__("pathlib").Path(out_dir))
                return self_inner

            def __exit__(self_inner, *_):
                pass

            def phase(self_inner, _name):
                return _Ctx()

        return _Ctx()

    def _fake_driver(cmd, pattern, k_units, grace=5.0):
        return {
            "mark_times": [0.0, 5.0, 7.0, 9.0],
            "reached_units": 4,
            "killed": True,
            "exit_code": 0,
        }

    monkeypatch.setattr("flightrec.smoke.FlightRecorder", _fake_recorder)
    monkeypatch.setattr("flightrec.smoke.run_until_marks", _fake_driver)

    card = S.smoke(["fake-bench"], r"UNIT", str(out), units=4, total_units=100)
    assert card["verdict"] == "GPU-BUSY"  # high-duty synthetic artifact
    assert card["eta"]["eta_s"] == 200.0  # 2.0 s/unit × 100 units
    assert card["slice"]["reached_units"] == 4
    text = S.format_card(card)
    assert "PRE-FLIGHT" in text and "recommendations" in text
