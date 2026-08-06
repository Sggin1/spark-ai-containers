# File: phase_tree.py
# Location: flightrec/phase_tree.py
# Purpose: Reconstruct the nested phase tree from the flat phases.parquet (t0/t1 containment).
# Dependencies: pandas (frames passed in)

"""Phase-tree reconstruction (the C4 arm).

``phases.parquet`` is a flat list of ``{phase, t0_ns, t1_ns}``. When phases are
nested — one ``rec.phase()`` inside another — the t0/t1 containment relationship
already encodes a tree; we just rebuild it. Each node carries its window's
duration + GPU energy + mean SM clock so a hierarchical breakdown (parent→child
drill-down) is available without any extra instrumentation.

Metrics are **inclusive** (a node's window totals include its children's), the
natural reading of "how much did this phase cost" — the same number ``measure``
would report for that span. Nesting uses the classic O(n log n) interval algorithm:
sort by ``(t0 asc, t1 desc)`` so a container always precedes what it contains, then
a stack assigns each node to the smallest open ancestor. Phases recorded via
context managers are always properly nested, so partial overlaps don't arise; if
one ever does, the non-contained node simply detaches to an ancestor / root.
"""

import pandas as pd


def build_tree(phases, samples=None):
    """Reconstruct the nested phase forest from a flat phases frame.

    Args:
        phases: The artifact's ``phases`` frame (flat ``{phase, t0_ns, t1_ns}`` rows).
        samples: Optional samples frame; when given, each node gets ``energy_j`` and
            ``mean_clock_mhz`` for its window (else both None).

    Returns:
        A list of root nodes; each node is a dict with ``phase``, ``t0_ns``,
        ``t1_ns``, ``duration_s``, ``energy_j``, ``mean_clock_mhz``, ``children``.
    """
    return _nest(_nodes(phases, samples))


def _nodes(phases, samples):
    """Build a flat list of metric-annotated nodes, skipping the empty/no-phase row."""
    if phases.empty or "t0_ns" not in phases.columns:
        return []
    built = (_node(row, samples) for _, row in phases.iterrows())
    return list(filter(None, built))


def _node(row, samples):
    """One metric-annotated node from a phase row, or None if it has no interval."""
    if pd.isna(row.get("t0_ns")) or pd.isna(row.get("t1_ns")):
        return None
    t0, t1 = int(row["t0_ns"]), int(row["t1_ns"])
    node = {"phase": row["phase"], "t0_ns": t0, "t1_ns": t1,
            "duration_s": round((t1 - t0) / 1e9, 6), "children": []}
    node.update(_metrics(samples, t0, t1))
    return node


def _nest(nodes):
    """Assign each node to its smallest containing ancestor via a sort+stack pass."""
    ordered = sorted(nodes, key=lambda n: (n["t0_ns"], -n["t1_ns"]))
    stack, roots = [], []
    for node in ordered:
        while stack and not _contains(stack[-1], node):
            stack.pop()
        (stack[-1]["children"] if stack else roots).append(node)
        stack.append(node)
    return roots


def _contains(outer, inner):
    """True when *outer*'s window fully covers *inner*'s."""
    return outer["t0_ns"] <= inner["t0_ns"] and inner["t1_ns"] <= outer["t1_ns"]


def _metrics(samples, t0, t1):
    """GPU energy (J) and mean SM clock (MHz) over [t0, t1], or Nones without samples."""
    if samples is None or "t_ns" not in getattr(samples, "columns", []):
        return {"energy_j": None, "mean_clock_mhz": None}
    window = samples[samples["t_ns"].between(t0, t1)]
    return {"energy_j": _energy_j(window), "mean_clock_mhz": _mean_clock(window)}


def _energy_j(window):
    if "energy_mj" not in window.columns or window["energy_mj"].dropna().empty:
        return None
    energy = window["energy_mj"].dropna()
    return round(float(energy.max() - energy.min()) / 1000.0, 3)


def _mean_clock(window):
    if "sm_clock_mhz" not in window.columns or window["sm_clock_mhz"].dropna().empty:
        return None
    return int(round(float(window["sm_clock_mhz"].dropna().mean())))


def format_tree(roots):
    """Render the phase forest as an indented parent→child drill-down."""
    lines = []
    for root in roots:
        _format_node(root, 0, lines)
    return "\n".join(lines)


def _format_node(node, depth, lines):
    pad = "  " * depth
    energy = f"{node['energy_j']} J" if node["energy_j"] is not None else "n/a"
    clock = f"{node['mean_clock_mhz']} MHz" if node["mean_clock_mhz"] is not None else "n/a"
    lines.append(f"{pad}{node['phase']}  ·  {node['duration_s']}s  ·  {energy}  ·  {clock}")
    for child in node["children"]:
        _format_node(child, depth + 1, lines)
