# File: calibration.py
# Location: flightrec/calibration.py
# Purpose: Score predicted-vs-measured so the autotune sweep can PRUNE before measuring.
# Dependencies: numpy

"""Calibrate the loop's `predict` arm against what `measure_artifact` actually measured.

The discovery loop predicts each config's performance (roofline) then measures it. This compares the two
across the accumulated rows and answers the question the autotuner needs: *is prediction good enough to
prune?* — i.e. can it RANK configs reliably (so we skip measuring the obvious losers). It reports both the
absolute error (mae/bias, and a correction factor to de-bias predictions) and the rank correlation
(Spearman) that governs the prune decision. The loop becoming self-aware about when to trust itself.
"""

import numpy as np


def calibration_report(predicted, measured, tol_pct=15.0, prune_spearman=0.8):
    """Compare predicted vs measured for the same metric across configs.

    Args:
        predicted: Predicted metric per config (e.g. roofline kernel_s or %-of-peak).
        measured: Measured metric per config (from measure_artifact), same order/units.
        tol_pct: |rel error| within this counts as "on target".
        prune_spearman: rank-correlation threshold above which pruning is safe.

    Returns:
        n, mae_pct, bias_pct (<0 = predictions optimistic), within_tol_pct, spearman,
        correction_factor (multiply predictions by this to de-bias), and a `prunable` verdict.
    """
    pred = np.asarray(predicted, dtype=float)
    meas = np.asarray(measured, dtype=float)
    rel = (pred - meas) / meas
    spearman = _spearman(pred, meas) if pred.size >= 3 else None
    return {
        "n": int(pred.size),
        "mae_pct": round(100.0 * float(np.mean(np.abs(rel))), 1),
        "bias_pct": round(100.0 * float(np.mean(rel)), 1),
        "within_tol_pct": round(100.0 * float(np.mean(np.abs(rel) <= tol_pct / 100.0)), 1),
        "spearman": round(spearman, 3) if spearman is not None else None,
        "correction_factor": round(float(np.median(meas / pred)), 3),
        "prunable": bool(spearman is not None and spearman >= prune_spearman),
        "tol_pct": tol_pct,
    }


def _spearman(a, b):
    """Spearman rank correlation (Pearson of ranks); no scipy dependency."""
    rank_a = a.argsort().argsort().astype(float)
    rank_b = b.argsort().argsort().astype(float)
    if rank_a.std() == 0 or rank_b.std() == 0:
        return 0.0
    return float(np.corrcoef(rank_a, rank_b)[0, 1])
