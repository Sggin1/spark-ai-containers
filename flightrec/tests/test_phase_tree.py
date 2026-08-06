# File: test_phase_tree.py
# Location: tests/test_phase_tree.py
# Purpose: flightrec phase-tree reconstruction — nesting from t0/t1 containment + per-node metrics.
# Dependencies: pandas, flightrec.phase_tree

"""C4: rebuild the nested phase tree from the flat phases frame. Containment encodes
nesting; the sort+stack pass must attach each phase to its smallest container and
roll up duration / energy / mean clock per node from the samples."""

import pandas as pd

from flightrec import phase_tree as PT


def _phases(rows):
    return pd.DataFrame(rows)


def _samples(clock=2300, e0=0.0, e1=10000.0, dur_ns=1_000_000_000, n=11):
    """A linear energy ramp over [0, dur_ns] so window energy is proportional to span."""
    t = [int(i * dur_ns / (n - 1)) for i in range(n)]
    energy = [e0 + (e1 - e0) * i / (n - 1) for i in range(n)]
    return pd.DataFrame({"t_ns": t, "t_rel_s": [x / 1e9 for x in t],
                         "sm_clock_mhz": [clock] * n, "energy_mj": energy})


# ---------------------------------------------------------------------------
# nesting from containment
# ---------------------------------------------------------------------------

def test_single_phase_is_a_root_leaf():
    tree = PT.build_tree(_phases([{"phase": "decode", "t0_ns": 0, "t1_ns": 1_000_000_000}]))
    assert len(tree) == 1
    assert tree[0]["phase"] == "decode"
    assert tree[0]["children"] == []


def test_child_nested_under_parent():
    # 'inner' (200..600) sits inside 'outer' (0..1000) -> outer.children == [inner]
    tree = PT.build_tree(_phases([
        {"phase": "outer", "t0_ns": 0, "t1_ns": 1_000_000_000},
        {"phase": "inner", "t0_ns": 200_000_000, "t1_ns": 600_000_000},
    ]))
    assert len(tree) == 1
    assert tree[0]["phase"] == "outer"
    assert [c["phase"] for c in tree[0]["children"]] == ["inner"]


def test_two_children_become_siblings():
    tree = PT.build_tree(_phases([
        {"phase": "outer", "t0_ns": 0, "t1_ns": 1_000_000_000},
        {"phase": "a", "t0_ns": 100_000_000, "t1_ns": 300_000_000},
        {"phase": "b", "t0_ns": 400_000_000, "t1_ns": 700_000_000},
    ]))
    assert [c["phase"] for c in tree[0]["children"]] == ["a", "b"]
    assert all(c["children"] == [] for c in tree[0]["children"])


def test_three_levels_deep():
    tree = PT.build_tree(_phases([
        {"phase": "L0", "t0_ns": 0, "t1_ns": 1_000_000_000},
        {"phase": "L1", "t0_ns": 100_000_000, "t1_ns": 900_000_000},
        {"phase": "L2", "t0_ns": 200_000_000, "t1_ns": 800_000_000},
    ]))
    l0 = tree[0]
    assert l0["phase"] == "L0"
    assert l0["children"][0]["phase"] == "L1"
    assert l0["children"][0]["children"][0]["phase"] == "L2"


def test_disjoint_phases_are_separate_roots():
    tree = PT.build_tree(_phases([
        {"phase": "p1", "t0_ns": 0, "t1_ns": 400_000_000},
        {"phase": "p2", "t0_ns": 500_000_000, "t1_ns": 900_000_000},
    ]))
    assert [r["phase"] for r in tree] == ["p1", "p2"]


def test_unordered_input_still_nests():
    # inner listed BEFORE outer -> the sort must still place inner under outer.
    tree = PT.build_tree(_phases([
        {"phase": "inner", "t0_ns": 200_000_000, "t1_ns": 600_000_000},
        {"phase": "outer", "t0_ns": 0, "t1_ns": 1_000_000_000},
    ]))
    assert tree[0]["phase"] == "outer"
    assert tree[0]["children"][0]["phase"] == "inner"


# ---------------------------------------------------------------------------
# empty / no-phase artifacts
# ---------------------------------------------------------------------------

def test_no_phase_row_yields_empty_tree():
    # recorder writes [{"phase": None}] when nothing was marked.
    assert PT.build_tree(pd.DataFrame([{"phase": None}])) == []


def test_empty_frame_yields_empty_tree():
    assert PT.build_tree(pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# per-node metrics from samples (inclusive of children)
# ---------------------------------------------------------------------------

def test_node_metrics_from_samples():
    # outer spans full ramp (10 J); inner spans 0.2..0.6 s = 40% -> ~4 J.
    samples = _samples(clock=2300, e1=10000.0)
    tree = PT.build_tree(_phases([
        {"phase": "outer", "t0_ns": 0, "t1_ns": 1_000_000_000},
        {"phase": "inner", "t0_ns": 200_000_000, "t1_ns": 600_000_000},
    ]), samples=samples)
    outer = tree[0]
    inner = outer["children"][0]
    assert outer["duration_s"] == 1.0
    assert outer["energy_j"] == 10.0
    assert outer["mean_clock_mhz"] == 2300
    assert abs(inner["energy_j"] - 4.0) < 0.5      # inclusive window energy ~ 40%


def test_metrics_none_without_samples():
    tree = PT.build_tree(_phases([{"phase": "x", "t0_ns": 0, "t1_ns": 1_000_000_000}]))
    assert tree[0]["energy_j"] is None
    assert tree[0]["mean_clock_mhz"] is None


# ---------------------------------------------------------------------------
# format_tree — indented drill-down
# ---------------------------------------------------------------------------

def test_format_tree_indents_children():
    tree = PT.build_tree(_phases([
        {"phase": "outer", "t0_ns": 0, "t1_ns": 1_000_000_000},
        {"phase": "inner", "t0_ns": 200_000_000, "t1_ns": 600_000_000},
    ]))
    text = PT.format_tree(tree)
    lines = text.splitlines()
    assert lines[0].startswith("outer")
    assert lines[1].startswith("  inner")   # child indented one level
