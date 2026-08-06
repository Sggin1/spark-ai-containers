# File: compute_cal.py
# Location: flightrec/compute_cal.py
# Purpose: Compute-ceiling calibration — the FLOP-peak twin of roofline's bandwidth-wall calibrate.
# Dependencies: stdlib (json, subprocess, sys, os) + flightrec.roofline (calibration path/loader)

"""Per-box compute ceiling, persisted so the 2-axis ``regime`` runs hands-free.

``roofline.calibrate`` measures the bandwidth WALL; this measures the compute
PEAK (TFLOP/s) and writes it into the SAME per-box calibration file under
``compute_peak_tflops`` (a dtype->TFLOP/s map). ``measure`` then auto-reads it,
so ``--peak-tflops`` becomes optional.

Honesty about GB10: a dense bf16/fp16 matmul peak IS measurable from torch (a big
square matmul, best-of-N, CUDA-event timed). The NVFP4/FP8 peak is NOT — there is
no FP4 matmul in torch (that is what ``nvfp4bench`` exists for). So quant dtypes
are RECORDED from a known value (e.g. GB10 NVFP4 dense ~482 TF / sparse ~964 TF,
SASS-measured) with provenance ``recorded`` rather than ``matmul-bench``. Both
land in the same map; both are auto-read.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from flightrec.roofline import calibration_path, load_calibration

# dtypes whose peak a torch matmul can actually measure (a @ b path).
_TORCH_MEASURABLE = {"bf16": "torch.bfloat16", "fp16": "torch.float16"}

# Big square matmul; prints one JSON line. tflops = 2*n^3 / best_time.
_MATMUL_PROGRAM = r"""
import json, torch
n = {n}; reps = {reps}; dt = {dtype_expr}
a = torch.randn(n, n, device="cuda", dtype=torch.float32).to(dt)
b = torch.randn(n, n, device="cuda", dtype=torch.float32).to(dt)
for _ in range(10):
    c = a @ b
torch.cuda.synchronize()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
best_ms = float("inf")
for _ in range(reps):
    start.record(); c = a @ b; end.record(); torch.cuda.synchronize()
    best_ms = min(best_ms, start.elapsed_time(end))
tflops = (2 * n ** 3) / (best_ms / 1e3) / 1e12
print(json.dumps({{"compute_peak_tflops_measured": round(tflops, 1),
                   "n": n, "best_ms": round(best_ms, 4),
                   "backend": "torch " + torch.__version__}}))
"""


def measure_compute_peak(dtype, python=None, n=8192, reps=50):
    """Measure a dense matmul TFLOP/s peak in a torch python (bf16/fp16 only).

    Raises:
        ValueError: for a dtype no torch ``a @ b`` can express (fp8/fp4/int4) —
            record those with ``calibrate_compute(..., peak_tflops=...)`` instead.
        RuntimeError: if the torch python lacks torch+CUDA or the run fails.
    """
    key = dtype.lower()
    if key not in _TORCH_MEASURABLE:
        raise ValueError(
            f"matmul bench can't measure {dtype!r} (no torch FP4/FP8 matmul); "
            "record it with --peak-tflops from an external bench (e.g. nvfp4bench)."
        )
    interp = python or os.environ.get("FLIGHTREC_TORCH_PYTHON") or sys.executable
    program = _MATMUL_PROGRAM.format(n=n, reps=reps, dtype_expr=_TORCH_MEASURABLE[key])
    proc = subprocess.run([interp, "-c", program], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"compute-peak measurement failed under {interp!r}. Point --python at a "
            f"torch+CUDA interpreter.\n{proc.stderr.strip()}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def calibrate_compute(dtype, python=None, peak_tflops=None, n=8192, reps=50, host=None):
    """Measure (bf16/fp16) or record (peak_tflops given) a dtype's compute peak.

    Merges into the per-box calibration file's ``compute_peak_tflops`` map and
    returns the merged record. ``peak_tflops`` short-circuits the matmul bench —
    the path for quant dtypes whose peak is SASS-measured elsewhere.
    """
    if peak_tflops is not None:
        value, source = float(peak_tflops), "recorded"
    else:
        value = measure_compute_peak(dtype, python, n, reps)["compute_peak_tflops_measured"]
        source = "matmul-bench"
    return _persist_peak(dtype.lower(), value, source, host)


def _persist_peak(dtype, value, source, host):
    """Read-merge-write the compute peak into the per-box calibration file."""
    record = load_calibration(host) or {}
    peaks = record.get("compute_peak_tflops", {})
    peaks[dtype] = value
    record["compute_peak_tflops"] = peaks
    record.setdefault("compute_peak_source", {})[dtype] = source
    path = calibration_path(host)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {
        "dtype": dtype,
        "compute_peak_tflops": value,
        "source": source,
        "calibration_path": str(path),
    }


def compute_peak_for(dtype, host=None):
    """Persisted compute peak (TFLOP/s) for a dtype, or None if uncalibrated."""
    record = load_calibration(host) or {}
    return record.get("compute_peak_tflops", {}).get(dtype.lower())
