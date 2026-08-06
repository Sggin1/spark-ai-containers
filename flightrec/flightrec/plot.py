# File: plot.py
# Location: flightrec/plot.py
# Purpose: Statistical graphs from artifacts (time-series overlay, runtime violin).
# Dependencies: matplotlib (optional `plot` extra), pandas, flightrec.report

"""Graphs that each drive a decision, not decoration.

``timeseries`` is the "between 22.7-33.4 s only 3 cores active" picture: stacked
lanes on the shared monotonic timeline with phase bands. The lane set is
**adaptive** — every lane whose column is present in the parquet is drawn, so an
old 4-column artifact and a new one carrying recorder-jitter / PSI / CPU-freq both
render without empty axes. The GPU-power lane overlays the energy-integral RATE
(``d(energy_mj)/dt`` in W) on top of instantaneous ``power_w``: two independent
measurements of the same quantity, so visible disagreement = the C6 consistency
flag, seen. ``runtime_violin`` shows run-time distributions across configs so
median+IQR (the fact) replaces a single number (the anecdote).
"""

from pathlib import Path

from flightrec.report import load_run

# Lane spec: (column, label). Drawn in this order, filtered to columns present.
# Keep power_w first — it carries the energy-rate overlay.
_LANES = (
    ("power_w", "GPU power (W)"),
    ("sm_clock_mhz", "SM clock (MHz)"),
    ("n_active", "active CPU cores /20"),
    ("gpu_busy_pct", "GPU busy % (~6 Hz)"),
    ("actual_dt_ms", "recorder jitter (ms)"),
    ("psi_cpu_some10", "PSI CPU some-10 (%)"),
    ("cpu_freq_mean_mhz", "CPU freq (MHz)"),
)
_LABELS = dict(_LANES)

# Default lanes for a multi-run overlay — the comparison-worthy signals.
OVERLAY_COLUMNS = ("power_w", "sm_clock_mhz", "gpu_busy_pct")


def timeseries(run_dir, out_png=None):
    """Stacked overlay of every present lane on the shared monotonic timeline."""
    plt = _backend()
    samples, phases, _ = load_run(run_dir)
    origin = samples["t_ns"].iloc[0] - samples["t_rel_s"].iloc[0] * 1e9
    lanes = present_lanes(samples)
    fig, axes = plt.subplots(len(lanes), 1, sharex=True, figsize=(11, 1.9 * len(lanes)))
    axes = _as_list(axes)
    for axis, (column, label) in zip(axes, lanes):
        _line(axis, samples, column, label)
        if column == "power_w":
            _overlay_energy_rate(axis, samples)
    _mark_phases(axes, phases, origin)
    axes[-1].set_xlabel("t_rel_s")
    out = out_png or str(Path(run_dir) / "timeseries.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    return out


def present_lanes(samples):
    """The subset of ``_LANES`` whose column exists (with data) in this artifact."""
    return [(col, label) for col, label in _LANES if _has_data(samples, col)]


def energy_rate_series(samples):
    """``(t_rel_s, watts)`` of d(energy_mj)/dt at the energy-sample ticks, or None.

    Energy is the slow-path monotonic mJ integral (sparse — NaN on fast ticks), so
    the rate is differenced over the energy samples only and converted to watts
    (``ΔmJ / 1000 / Δs``). The leading point has no predecessor and is dropped.
    """
    if "energy_mj" not in samples.columns or "t_rel_s" not in samples.columns:
        return None
    frame = samples[["t_rel_s", "energy_mj"]].dropna()
    if len(frame) < 2:
        return None
    dt = frame["t_rel_s"].diff()
    watts = (frame["energy_mj"].diff() / 1000.0) / dt
    valid = dt > 0
    return frame["t_rel_s"][valid], watts[valid]


def timeseries_overlay(run_dirs, columns=None, labels=None, out_png="overlay.png"):
    """Overlay N runs' lanes on shared timelines — warmup vs steady-state, A vs B.

    Each run is one coloured line per lane on a common ``t_rel_s`` axis, so BF16 vs
    NVFP4 power profiles (or warmup vs steady state) are visible side-by-side instead
    of as two separate PNGs. Lanes default to ``OVERLAY_COLUMNS``; a run missing a
    column is simply skipped in that lane.

    Args:
        run_dirs: Artifact directories to overlay.
        columns: Lane columns to draw (default ``OVERLAY_COLUMNS``).
        labels: Per-run legend labels (default each dir's basename).
        out_png: Output path.
    """
    plt = _backend()
    columns = columns or OVERLAY_COLUMNS
    names = resolve_labels(run_dirs, labels)
    runs = [(name, load_run(d)[0]) for name, d in zip(names, run_dirs)]
    fig, axes = plt.subplots(len(columns), 1, sharex=True, figsize=(11, 2.2 * len(columns)))
    axes = _as_list(axes)
    for axis, column in zip(axes, columns):
        _overlay_lane(axis, runs, column)
    axes[-1].set_xlabel("t_rel_s")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    return out_png


def resolve_labels(run_dirs, labels):
    """Legend labels for an overlay: explicit *labels* or each dir's basename."""
    if labels is not None:
        return list(labels)
    return [Path(d).name for d in run_dirs]


def _overlay_lane(axis, runs, column):
    """Draw every run's *column* on one axis with a per-run legend entry."""
    for name, samples in runs:
        if column in samples.columns:
            series = samples[["t_rel_s", column]].dropna()
            axis.plot(series["t_rel_s"], series[column], linewidth=0.9, label=name)
    axis.set_ylabel(_LABELS.get(column, column), fontsize=9)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, loc="best")


def runtime_violin(durations_by_label, out_png):
    """Violin of run-time distributions across configs (thin wrapper over metric_violins)."""
    return metric_violins({"run time (s)": durations_by_label}, out_png)


def metric_violins(metrics_by_label, out_png):
    """One violin panel per metric — runtime + j_token + tok_s show the Pareto shape.

    Args:
        metrics_by_label: ``{metric_name: {config_label: [values]}}``. Each metric
            becomes a panel; within it, one violin per config with the median marked.
        out_png: Output path.
    """
    plt = _backend()
    metrics = list(metrics_by_label)
    fig, axes = plt.subplots(1, len(metrics), figsize=(max(5, 4.5 * len(metrics)), 5))
    for axis, metric in zip(_as_list(axes), metrics):
        _violin_panel(axis, metric, metrics_by_label[metric])
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    return out_png


def _violin_panel(axis, metric, by_label):
    labels = list(by_label)
    axis.violinplot([list(by_label[k]) for k in labels], showmedians=True)
    axis.set_xticks(range(1, len(labels) + 1))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_ylabel(metric)


def _backend():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _has_data(samples, column):
    return column in samples.columns and not samples[column].dropna().empty


def _as_list(axes):
    """matplotlib returns a bare Axes for a 1-row figure; normalise to a list."""
    return list(axes) if hasattr(axes, "__len__") else [axes]


def _line(axis, samples, column, label):
    if column not in samples.columns:
        return
    series = samples[["t_rel_s", column]].dropna()
    axis.plot(series["t_rel_s"], series[column], linewidth=0.8, label="measured")
    axis.set_ylabel(label, fontsize=9)
    axis.grid(alpha=0.3)


def _overlay_energy_rate(axis, samples):
    """Draw the energy-integral rate (W) over the power lane; agreement = healthy."""
    rate = energy_rate_series(samples)
    if rate is None:
        return
    times, watts = rate
    axis.plot(times, watts, linewidth=0.8, linestyle="--", color="orange",
              label="energy-rate ΔE/Δt")
    axis.legend(fontsize=7, loc="upper right")


def _mark_phases(axes, phases, origin):
    if phases.empty or "t0_ns" not in phases.columns:
        return
    for _, row in phases.iterrows():
        start = (row["t0_ns"] - origin) / 1e9
        end = (row["t1_ns"] - origin) / 1e9
        for axis in axes:
            axis.axvspan(start, end, alpha=0.07, color="green")
