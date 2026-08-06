# File: test_consistency.py
# Location: tests/test_consistency.py
# Purpose: Cross-check physical invariants in flightrec output — catch measurement artifacts
#          before they become bad data.  Each test encodes a constraint that, if violated,
#          indicates a real methodology problem (L2 residency, broken energy counter, etc.).

"""Physical consistency oracle for flightrec measurements.

Pattern: build a synthetic artifact that SHOULD trigger the anomaly, assert the flag fires,
then build a clean artifact, assert it doesn't.  When a new artifact is discovered in a real
run (above-wall BW, non-monotonic energy, tok_s divergence) add a test case here first so the
tool catches it on the next sweep.
"""

import json

import pandas as pd
import pytest

from flightrec.aggregate import aggregate
from flightrec.measure import measure_artifact, power_energy_consistency
from flightrec.roofline import roofline_point
from flightrec.validate import verdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path, clocks, powers, energies, busy=90.0, dur_ns=1_000_000_000,
           header_extra=None):
    """Write a multi-tick artifact; clocks/powers/energies are per-tick lists."""
    n = len(clocks)
    t_ns = [int(i * dur_ns / (n - 1)) for i in range(n)] if n > 1 else [0]
    t_rel = [t / 1e9 for t in t_ns]
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "t_ns": t_ns, "t_rel_s": t_rel,
        "sm_clock_mhz": clocks, "power_w": powers,
        "energy_mj": energies, "gpu_busy_pct": [busy] * n, "n_active": [4] * n,
    }).to_parquet(path / "samples.parquet")
    pd.DataFrame([{"phase": "bench", "t0_ns": 0, "t1_ns": dur_ns}]).to_parquet(
        path / "phases.parquet")
    hdr = {"bandwidth_wall_gbps": 220.0, "bandwidth_wall_source": "measured (test)"}
    if header_extra:
        hdr.update(header_extra)
    (path / "header.json").write_text(json.dumps(hdr))


def _healthy_run(path, energy_end_mj=5000.0):
    _write(path,
           clocks=[2300, 2300, 2300],
           powers=[60.0, 60.0, 60.0],
           energies=[0.0, 2500.0, energy_end_mj])


# ---------------------------------------------------------------------------
# 1. above_wall — L2 residency trap
# ---------------------------------------------------------------------------

class TestAboveWall:
    """achieved_gbps > wall_gbps is physically impossible from DRAM alone.
    Historically: decode M=1 benchmarks showed >100% of wall when weights were
    L2-resident.  Adding weight rotation (> L2 size) fixed it.
    """

    def test_above_wall_flag_fires(self):
        # 300 GB/s achieved vs 200 GB/s wall -> above_wall must be True
        pt = roofline_point(bytes_moved=30e9, flops=1, kernel_s=0.1, wall_gbps=200.0)
        assert pt["pct_of_wall"] == 150.0
        assert pt["above_wall"] is True

    def test_above_wall_flag_clear_when_below(self):
        # 180 GB/s vs 200 GB/s wall -> above_wall must be False
        pt = roofline_point(bytes_moved=18e9, flops=1, kernel_s=0.1, wall_gbps=200.0)
        assert pt["pct_of_wall"] == 90.0
        assert pt["above_wall"] is False

    def test_above_wall_at_exactly_100pct(self):
        # Exactly at the wall is NOT above (boundary belongs to the safe side)
        pt = roofline_point(bytes_moved=20e9, flops=1, kernel_s=0.1, wall_gbps=200.0)
        assert pt["pct_of_wall"] == 100.0
        assert pt["above_wall"] is False

    def test_above_wall_propagates_through_measure_artifact(self, tmp_path):
        """above_wall must surface from measure_artifact so callers see it."""
        _healthy_run(tmp_path / "run")
        # 220 GB wall; claim 330 GB moved in 1 s -> 330 GB/s -> above_wall
        vec = measure_artifact(str(tmp_path / "run"), bytes_moved=330e9)
        assert vec["above_wall"] is True
        assert vec["pct_of_wall"] > 100.0

    def test_above_wall_absent_when_no_bytes(self, tmp_path):
        """When bytes_moved is unknown, above_wall must be None (not False)."""
        _healthy_run(tmp_path / "run")
        vec = measure_artifact(str(tmp_path / "run"))
        assert vec["above_wall"] is None


# ---------------------------------------------------------------------------
# 2. Energy monotonicity — NVML integral must never decrease
# ---------------------------------------------------------------------------

class TestEnergyMonotonicity:
    """energy_mj is a cumulative NVML counter.  A decrease means a counter reset
    or sampling bug — j_token computed from such a run is silently wrong.
    """

    def test_monotonic_energy_is_valid(self):
        df = pd.DataFrame({
            "sm_clock_mhz": [2300, 2300, 2300],
            "power_w": [60.0, 60.0, 60.0],
            "energy_mj": [0.0, 2500.0, 5000.0],
            "gpu_busy_pct": [90.0, 90.0, 90.0],
        })
        result = verdict(df)
        assert result["energy_monotonic"] is True
        assert result["valid"] is True

    def test_non_monotonic_energy_invalidates_run(self):
        # energy drops at tick 2 -> counter reset or bug -> run must be INVALID
        df = pd.DataFrame({
            "sm_clock_mhz": [2300, 2300, 2300],
            "power_w": [60.0, 60.0, 60.0],
            "energy_mj": [0.0, 5000.0, 3000.0],   # 5000 -> 3000: decrease
            "gpu_busy_pct": [90.0, 90.0, 90.0],
        })
        result = verdict(df)
        assert result["energy_monotonic"] is False
        assert result["valid"] is False

    def test_no_energy_column_does_not_invalidate(self):
        # Some artifacts may lack energy (older recorders) — must not fail.
        df = pd.DataFrame({
            "sm_clock_mhz": [2300, 2300],
            "power_w": [60.0, 60.0],
            "gpu_busy_pct": [90.0, 90.0],
        })
        result = verdict(df)
        assert result["energy_monotonic"] is True
        assert result["valid"] is True

    def test_non_monotonic_energy_excluded_from_aggregate(self, tmp_path):
        """A run with decreasing energy must be dropped by aggregate() like a wedge."""
        good = tmp_path / "good"
        _write(good, clocks=[2300, 2300], powers=[60.0, 60.0], energies=[0.0, 5000.0])
        bad = tmp_path / "bad"
        _write(bad, clocks=[2300, 2300], powers=[60.0, 60.0], energies=[5000.0, 3000.0])

        result = aggregate([str(good), str(bad)], metric="energy_j")
        assert result["n_total"] == 2
        assert result["n_valid"] == 1
        assert result["n_invalid_dropped"] == 1


# ---------------------------------------------------------------------------
# 3. tok_s cross-check — computed vs benchy-reported must agree
# ---------------------------------------------------------------------------

class TestTokSCrossCheck:
    """When kernel_s gives us tokens/kernel_s AND benchy also reported tok_s
    independently, the two should agree within 20%.  A large divergence means
    the phase window is misaligned with what the bench actually timed.
    """

    def test_consistent_tok_s_flagged_ok(self, tmp_path):
        # kernel_s = 1.0 s, tokens = 100 -> computed tok_s = 100.0
        # benchy reports 98.0 -> 2% divergence -> consistent
        _healthy_run(tmp_path / "run")
        hdr = {"bandwidth_wall_gbps": 220.0,
               "throughput": {"tok_s": 98.0}}
        (tmp_path / "run" / "header.json").write_text(json.dumps(hdr))
        vec = measure_artifact(str(tmp_path / "run"), tokens=100)
        assert vec["tok_s_consistent"] is True
        assert vec["tok_s_divergence_pct"] < 5.0

    def test_divergent_tok_s_flagged(self, tmp_path):
        # kernel_s = 1.0 s, tokens = 100 -> computed tok_s = 100.0
        # benchy reports 50.0 -> 50% divergence -> inconsistent
        _healthy_run(tmp_path / "run")
        hdr = {"bandwidth_wall_gbps": 220.0,
               "throughput": {"tok_s": 50.0}}
        (tmp_path / "run" / "header.json").write_text(json.dumps(hdr))
        vec = measure_artifact(str(tmp_path / "run"), tokens=100)
        assert vec["tok_s_consistent"] is False
        assert vec["tok_s_divergence_pct"] > 20.0

    def test_no_cross_check_without_benchy(self, tmp_path):
        # No header_tok_s -> no cross-check fields emitted
        _healthy_run(tmp_path / "run")
        vec = measure_artifact(str(tmp_path / "run"), tokens=100)
        assert "tok_s_consistent" not in vec
        assert "tok_s_divergence_pct" not in vec


# ---------------------------------------------------------------------------
# 4. j_token identity: j_token * tokens must equal energy_j exactly
# ---------------------------------------------------------------------------

class TestJTokenIdentity:
    """j_token is defined as energy_j / tokens.  If the product doesn't round-trip,
    the formula or rounding diverged — a definitional error."""

    def test_j_token_equals_energy_over_tokens(self, tmp_path):
        _healthy_run(tmp_path / "run", energy_end_mj=5000.0)
        tokens = 128
        vec = measure_artifact(str(tmp_path / "run"), tokens=tokens)
        assert vec["j_token"] is not None
        assert vec["energy_j"] is not None
        # j_token is rounded to 4 decimal places; check it matches the exact ratio within
        # that precision (max error = 0.5e-4 J/token; not the round-trip j_token*tokens).
        assert abs(vec["j_token"] - vec["energy_j"] / tokens) < 0.5e-4

    def test_j_token_none_without_tokens(self, tmp_path):
        _healthy_run(tmp_path / "run")
        vec = measure_artifact(str(tmp_path / "run"))
        assert vec["j_token"] is None


# ---------------------------------------------------------------------------
# 6. power_w vs energy-rate — two independent measurements of average power
# ---------------------------------------------------------------------------

class TestPowerEnergyConsistency:
    """mean(power_w) (fast path) and d(energy_mj)/dt (slow path) measure the same
    average power; they must agree within 15%. Divergence flags an NVML energy
    glitch, the GB10 instantaneous-power smoothing lag, or a phase-boundary artefact.
    """

    def test_consistent_when_power_matches_energy_rate(self):
        # 60 W flat for 1 s -> 60 J -> energy_mj climbs 0..60000; rate == mean == 60 W.
        df = pd.DataFrame({
            "t_rel_s": [0.0, 0.5, 1.0],
            "power_w": [60.0, 60.0, 60.0],
            "energy_mj": [0.0, 30000.0, 60000.0],
        })
        out = power_energy_consistency(df)
        assert out["power_energy_consistent"] is True
        assert out["energy_rate_w"] == 60.0
        assert out["mean_power_w"] == 60.0
        assert out["power_energy_divergence_pct"] < 1.0

    def test_divergent_when_energy_counter_disagrees(self):
        # power reads 60 W but energy only climbs 30 J in 1 s (30 W rate) -> 50% off.
        df = pd.DataFrame({
            "t_rel_s": [0.0, 0.5, 1.0],
            "power_w": [60.0, 60.0, 60.0],
            "energy_mj": [0.0, 15000.0, 30000.0],
        })
        out = power_energy_consistency(df)
        assert out["power_energy_consistent"] is False
        assert out["power_energy_divergence_pct"] > 40.0

    def test_consistent_within_tolerance_band(self):
        # 60 W power vs ~52 W rate -> ~13% divergence -> still inside the 15% band.
        df = pd.DataFrame({
            "t_rel_s": [0.0, 1.0],
            "power_w": [60.0, 60.0],
            "energy_mj": [0.0, 52000.0],
        })
        out = power_energy_consistency(df)
        assert out["power_energy_consistent"] is True
        assert 10.0 < out["power_energy_divergence_pct"] < 15.0

    def test_idle_run_is_not_measured(self):
        # Real GB10 idle finding: rate 5.7 W vs mean 3.5 W = 38% relative because the
        # instantaneous-power floor diverges from the integral at idle. Load-scoping
        # (duty < LOAD_PCT here) drops it -> None, not a false alarm.
        df = pd.DataFrame({
            "t_rel_s": [0.0, 0.5, 1.0],
            "power_w": [3.5, 3.5, 3.5],
            "energy_mj": [0.0, 2750.0, 5500.0],
            "gpu_busy_pct": [2.0, 2.0, 2.0],   # idle: below LOAD_PCT
        })
        out = power_energy_consistency(df)
        assert out["power_energy_consistent"] is None

    def test_scopes_to_loaded_window_ignoring_idle_tail(self):
        # A loaded bench (60 W, consistent) followed by an idle tail (floor-divergent).
        # The check must score only the loaded portion -> consistent.
        df = pd.DataFrame({
            "t_rel_s": [0.0, 0.5, 1.0, 1.5, 2.0],
            "power_w": [60.0, 60.0, 60.0, 3.5, 3.5],
            "energy_mj": [0.0, 30000.0, 60000.0, 62000.0, 64000.0],
            "gpu_busy_pct": [90.0, 90.0, 90.0, 2.0, 2.0],
        })
        out = power_energy_consistency(df)
        assert out["power_energy_consistent"] is True
        assert out["mean_power_w"] == 60.0

    def test_field_present_but_none_without_energy(self):
        # No energy column -> key is still present (callers rely on it) but None.
        df = pd.DataFrame({"t_rel_s": [0.0, 1.0], "power_w": [60.0, 60.0]})
        out = power_energy_consistency(df)
        assert out["power_energy_consistent"] is None

    def test_surfaces_through_measure_artifact(self, tmp_path):
        _healthy_run(tmp_path / "run", energy_end_mj=60000.0)
        # _healthy_run: 60 W flat, energy 0..60000 over 1 s -> consistent.
        vec = measure_artifact(str(tmp_path / "run"))
        assert vec["power_energy_consistent"] is True
        assert vec["energy_rate_w"] == 60.0


# ---------------------------------------------------------------------------
# 5. Aggregate n invariant: n_valid + n_invalid_dropped == n_total always
# ---------------------------------------------------------------------------

class TestAggregateNInvariant:
    """The aggregate bookkeeping must account for every run — no silent drops."""

    def _make_run(self, path, clock, power, energies):
        _write(path, clocks=[clock] * 2, powers=[power] * 2,
               energies=energies, busy=90.0)

    def test_invariant_all_valid(self, tmp_path):
        dirs = []
        for i in range(3):
            d = tmp_path / f"r{i}"
            self._make_run(d, 2300, 60.0, [0.0, 5000.0])
            dirs.append(str(d))
        agg = aggregate(dirs, metric="energy_j")
        assert agg["n_valid"] + agg["n_invalid_dropped"] == agg["n_total"]
        assert agg["n_total"] == 3

    def test_invariant_mixed(self, tmp_path):
        dirs = []
        for i in range(4):
            d = tmp_path / f"r{i}"
            # every other run is a wedge
            if i % 2 == 0:
                self._make_run(d, 611, 13.0, [0.0, 5000.0])
            else:
                self._make_run(d, 2300, 60.0, [0.0, 5000.0])
            dirs.append(str(d))
        agg = aggregate(dirs, metric="energy_j")
        assert agg["n_valid"] + agg["n_invalid_dropped"] == agg["n_total"]
        assert agg["n_total"] == 4
        assert agg["n_valid"] == 2
