#!/usr/bin/env python3
"""
GB10 BF16 GEMM TFLOP/s sanity oracle (DUAL_SPARK_SETUP §0).

Healthy GB10: ~80-125 TFLOP/s burst, ~80-140 sustained.
Throttled GB10 (AC sag / undervolt — e.g. undersized UPS): ~8 burst, ~45 sustained.

Run inside a CUDA-equipped container:

  docker run --rm --gpus all --ipc=host \
    -v $PWD/gpu_stress.py:/work/gpu_stress.py \
    vllm-node-tf5 python3 -u /work/gpu_stress.py
"""
import time
import torch

DEV = "cuda"
DTYPE = torch.bfloat16
N = 16384            # GEMM size; 2 * N**3 = ~8.8 TFLOP per iter
WARMUP = 10
BURST_ITERS = 50      # short, eligible for cache + boost clocks
SUSTAINED_SEC = 20.0  # long, hits power envelope

assert torch.cuda.is_available(), "no CUDA device visible"
print(f"device: {torch.cuda.get_device_name(0)}")
print(f"GEMM size: {N}x{N} {DTYPE}, flops/iter = {2 * N**3 / 1e12:.2f} TFLOP")
print()

A = torch.randn(N, N, device=DEV, dtype=DTYPE)
B = torch.randn(N, N, device=DEV, dtype=DTYPE)
C = torch.empty(N, N, device=DEV, dtype=DTYPE)

# Warmup
for _ in range(WARMUP):
    torch.matmul(A, B, out=C)
torch.cuda.synchronize()

# Burst: best single-iter TFLOP/s
times = []
for _ in range(BURST_ITERS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.matmul(A, B, out=C)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
flops_per_iter = 2 * N ** 3
burst_tflops = flops_per_iter / min(times) / 1e12

# Sustained: total throughput over a fixed wall-clock window
deadline = time.perf_counter() + SUSTAINED_SEC
iters = 0
torch.cuda.synchronize()
t_start = time.perf_counter()
while time.perf_counter() < deadline:
    torch.matmul(A, B, out=C)
    iters += 1
torch.cuda.synchronize()
t_total = time.perf_counter() - t_start
sustained_tflops = (flops_per_iter * iters) / t_total / 1e12

print(f"burst    [{BURST_ITERS} iters, best-of]  : {burst_tflops:7.2f} TFLOP/s  (min {min(times)*1e3:.2f} ms / iter)")
print(f"sustained[{iters} iters / {t_total:.1f}s]: {sustained_tflops:7.2f} TFLOP/s")
print()

if burst_tflops < 60 or sustained_tflops < 60:
    print("VERDICT: UNHEALTHY (likely AC sag / undervolt — see DUAL_SPARK_SETUP §0)")
    raise SystemExit(2)
elif burst_tflops < 80 or sustained_tflops < 80:
    print("VERDICT: MARGINAL (below the documented healthy floor)")
    raise SystemExit(1)
else:
    print(f"VERDICT: HEALTHY (burst >= 80, sustained >= 80)")
