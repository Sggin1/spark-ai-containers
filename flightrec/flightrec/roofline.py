# File: roofline.py
# Location: flightrec/roofline.py
# Purpose: Roofline-as-code — measure the real GB10 BW wall and emit %-of-wall + AI as data.
# Dependencies: stdlib only (the GPU measurement shells out to a torch-enabled python).

"""Roofline-as-code for GB10.

Two jobs, no opinions baked in (the tool PROVIDES the numbers; interpretation is
downstream):

1. **Measure the bandwidth wall** instead of asserting it. GB10 exposes no live
   DRAM-bytes counter, so the *achievable* memory wall is found empirically with a
   STREAM-triad on the GPU (``a = b + s*c``), timed with CUDA events, best-of-N.
   The flightrec venv is deliberately torch-free, so the triad runs in a
   torch-enabled python via ``--python`` (or ``$FLIGHTREC_TORCH_PYTHON``) and
   returns JSON. The result is persisted per-box and read back by ``header.py``,
   replacing the old hard-coded ``195.0`` guess.

2. **Score a run against the wall.** ``roofline_point`` turns a measured kernel
   (bytes moved, FLOPs, kernel seconds) into the roofline coordinates —
   achieved GB/s, %-of-wall, arithmetic intensity, achieved GFLOP/s — the
   per-run data the baseline campaign mines. No conclusion is drawn here.
"""

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

GB10_SPEC_GBPS = 273.0  # LPDDR5X unified-memory spec peak (decimal GB)
MODELED_WALL_GBPS = 195.0  # legacy modeled fallback (~71% of spec) when uncalibrated

# Standalone STREAM-triad measured in a torch python; prints one JSON line to stdout.
# Best-of-N kernel time (CUDA events) -> max achieved bandwidth = the achievable wall.
_TRIAD_PROGRAM = r"""
import json, torch
n = {n}
reps = {reps}
dt = torch.float32
dev = "cuda"
a = torch.empty(n, device=dev, dtype=dt)
b = torch.randn(n, device=dev, dtype=dt)
c = torch.randn(n, device=dev, dtype=dt)
s = 0.42
for _ in range(10):  # warmup: prime clocks + caching allocator
    torch.add(b, c, alpha=s, out=a)
torch.cuda.synchronize()
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
best_ms = float("inf")
for _ in range(reps):
    start.record()
    torch.add(b, c, alpha=s, out=a)  # triad: read b, read c, write a = 3 arrays
    end.record()
    torch.cuda.synchronize()
    best_ms = min(best_ms, start.elapsed_time(end))
bytes_moved = 3 * n * a.element_size()
gbps = bytes_moved / (best_ms / 1e3) / 1e9
print(json.dumps({{
    "bandwidth_wall_gbps_measured": round(gbps, 1),
    "method": "STREAM-triad a=b+s*c, best-of-%d, fp32, CUDA-event timed" % reps,
    "backend": "torch " + torch.__version__,
    "gpu_name": torch.cuda.get_device_name(0),
    "n_elements": n,
    "best_kernel_ms": round(best_ms, 4),
    "bytes_per_rep": bytes_moved,
}}))
"""


def roofline_point(bytes_moved, flops, kernel_s, wall_gbps, peak_tflops=None):
    """Roofline coordinates for one measured kernel (pure data, no verdict).

    Args:
        bytes_moved: Bytes transferred to/from DRAM by the kernel (analytic).
        flops: Floating-point operations performed by the kernel.
        kernel_s: Kernel wall time in seconds (>0).
        wall_gbps: The achievable bandwidth wall in GB/s (from calibration).
        peak_tflops: Measured compute ceiling (TFLOP/s) for the kernel's dtype.
            When given, unlocks ``pct_of_peak`` and the TWO-axis ``regime`` that
            tells compute-bound from overhead-bound; omitted -> both None.

    Returns:
        Dict with achieved_gbps, achieved_gflops, arithmetic_intensity
        (FLOP/byte), pct_of_wall, pct_of_peak, the one-axis ``bandwidth_regime``
        (position vs the wall — blind to compute), and the two-axis ``regime``
        (None until ``peak_tflops`` is supplied).
    """
    if kernel_s <= 0 or bytes_moved <= 0:
        return {"error": "kernel_s and bytes_moved must be positive"}
    achieved_gbps = bytes_moved / kernel_s / 1e9
    achieved_gflops = flops / kernel_s / 1e9
    intensity = flops / bytes_moved
    pct_of_wall = 100.0 * achieved_gbps / wall_gbps if wall_gbps else None
    pct_of_peak = _pct(achieved_gflops / 1e3, peak_tflops)
    # above_wall: physically impossible from DRAM alone — indicates L2 residency artifact.
    # Caller should not treat pct_of_wall > 100 as a win; rotate/flush weights and re-run.
    above_wall = bool(achieved_gbps > wall_gbps) if wall_gbps else None
    return {
        "achieved_gbps": round(achieved_gbps, 2),
        "achieved_gflops": round(achieved_gflops, 2),
        "arithmetic_intensity": round(intensity, 4),
        "pct_of_wall": round(pct_of_wall, 1) if pct_of_wall is not None else None,
        "pct_of_peak": pct_of_peak,
        "above_wall": above_wall,
        "bandwidth_regime": _bandwidth_regime(pct_of_wall),
        "regime": _regime_2d(pct_of_wall, pct_of_peak),
    }


def roofline_predict(bytes_moved, flops, wall_gbps, peak_tflops):
    """Ideal-roofline prediction (lower-bound time) to PRUNE the autotune sweep before measuring.

    t = max(bytes_moved/wall, flops/peak). Real kernels achieve a fraction of this ideal;
    ``flightrec.calibration`` learns that fraction so predictions can rank/prune configs without
    measuring every one. Returns predicted kernel_s, %-of-wall, %-of-peak, and the binding bound.
    """
    t_bw = bytes_moved / (wall_gbps * 1e9) if wall_gbps else float("inf")
    t_cmp = flops / (peak_tflops * 1e12) if peak_tflops else float("inf")
    kernel_s = max(t_bw, t_cmp)
    if kernel_s in (0.0, float("inf")):
        return {"error": "need positive bytes_moved/flops and wall/peak"}
    return {
        "predicted_kernel_s": kernel_s,
        "predicted_pct_of_wall": _pct(bytes_moved / kernel_s / 1e9, wall_gbps),
        "predicted_pct_of_peak": _pct(flops / kernel_s / 1e12, peak_tflops),
        "predicted_bound": "compute" if t_cmp >= t_bw else "bandwidth",
    }


def _pct(achieved, ceiling):
    """Achieved ÷ ceiling as a %, or None when the ceiling is unknown."""
    return round(100.0 * achieved / ceiling, 1) if ceiling else None


def _bandwidth_regime(pct_of_wall):
    """Position on the BANDWIDTH axis only — a %-of-wall bin, NOT a full regime.

    This sees memory traffic vs the wall and nothing else, so it cannot tell a
    compute-bound kernel from a latency/overhead-bound one — both sit far below
    the wall. Only the near-wall case is a safe bound-claim (``bandwidth-bound``);
    the others are deliberately neutral ``*-wall`` positions, not verdicts. To
    resolve compute-vs-latency below the wall, compare achieved GFLOP/s to the
    measured peak (see ``roofline_predict``).
    """
    if pct_of_wall is None:
        return "unknown"
    if pct_of_wall >= 80.0:
        return "bandwidth-bound"
    if pct_of_wall <= 25.0:
        return "far-from-wall"
    return "mid-wall"


def _regime_2d(pct_of_wall, pct_of_peak):
    """Two-axis regime — needs BOTH ceilings; None when the compute peak is absent.

    The verdict ``_bandwidth_regime`` cannot give alone: a kernel far below the
    wall is COMPUTE-bound if it is near the FLOP peak, else OVERHEAD/LATENCY-bound
    (the 96%-busy-but-5%-of-peak case). Returns None (never a guess) when
    ``pct_of_peak`` is unknown, so callers never see a compute claim that was not
    measured.
    """
    if pct_of_wall is None or pct_of_peak is None:
        return None
    if pct_of_wall >= 80.0:
        return "bandwidth-bound"
    if pct_of_peak >= 80.0:
        return "compute-bound"
    return _below_wall_regime(pct_of_wall, pct_of_peak)


def _below_wall_regime(pct_of_wall, pct_of_peak):
    """Sub-wall AND sub-peak: overhead/latency-bound when BOTH are low (the GPU
    is busy but saturating neither ceiling), else a mixed/partial position."""
    if pct_of_wall < 40.0 and pct_of_peak < 40.0:
        return "overhead/latency-bound"
    return "mixed"


def calibration_path(host=None):
    """Per-box calibration file path (home is local to each Spark)."""
    host = host or socket.gethostname()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "flightrec"
    return base / f"calibration-{host}.json"


def load_calibration(host=None):
    """Read persisted calibration for this box, or None if uncalibrated."""
    path = calibration_path(host)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def measure_bw_wall(python=None, n_elements=1 << 26, reps=50):
    """Run the GPU STREAM-triad in a torch python and return its JSON result.

    Args:
        python: Path to a python with torch+CUDA. Defaults to
            ``$FLIGHTREC_TORCH_PYTHON`` then this interpreter.
        n_elements: Triad vector length (default 64Mi -> ~768 MB moved/rep).
        reps: Timed repetitions; the best (max bandwidth) is reported.

    Returns:
        The measurement dict from the triad program.

    Raises:
        RuntimeError: if the triad python lacks torch+CUDA or the run fails.
    """
    interp = python or os.environ.get("FLIGHTREC_TORCH_PYTHON") or sys.executable
    program = _TRIAD_PROGRAM.format(n=n_elements, reps=reps)
    proc = subprocess.run([interp, "-c", program], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"BW-wall measurement failed under {interp!r}. Point --python at a "
            f"torch+CUDA interpreter.\n{proc.stderr.strip()}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def calibrate(python=None, host=None, n_elements=1 << 26, reps=50):
    """Measure the BW wall and persist it for this box; return the record."""
    record = measure_bw_wall(python=python, n_elements=n_elements, reps=reps)
    record.update(
        {
            "host": host or socket.gethostname(),
            "measured_unix": time.time(),
            "spec_gbps": GB10_SPEC_GBPS,
            "pct_of_spec": round(
                100.0 * record["bandwidth_wall_gbps_measured"] / GB10_SPEC_GBPS, 1
            ),
        }
    )
    path = calibration_path(host)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    record["calibration_path"] = str(path)
    return record
