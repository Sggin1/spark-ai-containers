# FlashInfer JIT Compilation Issue on DGX Spark (sm_121)

## Problem

The NVIDIA vLLM containers (`nvcr.io/nvidia/vllm:25.12.post1-py3` and `26.01-py3`) do not ship
precompiled FlashInfer MoE CUTLASS kernels for sm_121a (DGX Spark GB10). On every cold start,
FlashInfer JIT-compiles 6+ CUDA kernels from source, causing a transient memory spike of 20-30 GB.

On DGX Spark's unified memory architecture, this spike competes with the model weights, KV cache,
and OS for the same 128 GB pool — unlike discrete GPU systems where host compilation and GPU
memory are separate.

## Root Cause

FlashInfer CUTLASS MoE GEMM kernels being compiled at runtime:
- `moe_gemm_kernels_fp8_uint4.cu` → compute_121a
- `moe_gemm_kernels_fp16_uint8.cu` → compute_121a
- `moe_gemm_kernels_bf16_uint8.cu` → compute_121a
- `moe_gemm_kernels_bf16_uint4.cu` → compute_121a
- `moe_gemm_kernels_fp16_uint4.cu` → compute_121a
- `moe_gemm_kernels_fp8_fp8.cu` → compute_121a

Each `cicc` (NVIDIA IR compiler) process consumes 1.5-6 GB RAM and they run in parallel.

## Why --enforce-eager and --compilation-config Don't Help

- `--enforce-eager` disables vLLM's CUDA graph capture — does NOT affect FlashInfer JIT
- `--compilation-config '{"level":0}'` disables torch.compile/inductor — does NOT affect FlashInfer JIT
- FlashInfer's JIT is independent of vLLM's compilation pipeline

## Memory Profile During Startup

| Phase | Duration | Memory Used | Notes |
|-------|----------|-------------|-------|
| Container init | ~5s | ~2 GB | Python, CUDA context |
| Model load | ~2min | ~22 GB | 19 GB weights + runtime |
| FlashInfer JIT | ~5-10min | **+20-30 GB spike** | 6 parallel cicc processes |
| JIT complete | — | spike freed | Back to ~22 GB |
| KV cache alloc | ~1s | +per util setting | 0.3=2GB, 0.5=26GB, 0.7=50GB |
| Steady state | — | 25-75 GB | Depends on gpu-memory-utilization |

## Workaround: Persistent FlashInfer Cache

Mount `/root/.cache/flashinfer` to a persistent host volume:

```bash
-v /mnt/ai/flashinfer-cache:/root/.cache/flashinfer
```

Cache path inside container:
```
/root/.cache/flashinfer/0.6.0rc2+.../121a/cached_ops/
```

After first successful compilation, the cache contains compiled `.so` files and `cicc` does not
need to run again. The spike becomes a one-time cost.

## First-run settings we used

For the initial JIT compilation run:

```bash
--gpu-memory-utilization 0.3   # Only 37 GB for vLLM, leaves 85 GB headroom
--max-model-len 32768          # Small KV pool
```

Peak during JIT: ~55 GB (model 19 + JIT 30 + runtime 6). Free during JIT: ~67 GB.

After first run, util can be raised to 0.5 or 0.7 since JIT spike will not recur.

## Tested Configurations

| Run | gpu-util | enforce-eager | compilation | Peak GB | Result |
|-----|:--------:|:-------------:|:-----------:|:-------:|--------|
| 1 | 0.9 | no | default | ~120 | Survived at 0.7 GB free |
| 2 | 0.5 | yes | default | ~65 | cicc still ran |
| 3 | 0.5 | yes | level=0 | ~60 | cicc still ran |
| 4 | 0.3 | no | default | 50+ climbing | Killed before completion |

## Alternative: eugr/spark-vllm-docker

The community image at https://github.com/eugr/spark-vllm-docker avoids the JIT spike:

- Builds FlashInfer from source at **image build time** with `FLASHINFER_CUDA_ARCH_LIST="12.1a"`
- Builds vLLM from source with `TORCH_CUDA_ARCH_LIST="12.1a"`
- Based on `nvidia/cuda:13.2.0-devel-ubuntu24.04`
- Ships prebuilt wheels for sm_121a — no JIT at runtime
- Uses `uv` throughout

Build: `cd spark-vllm-docker && ./build-and-copy.sh -j 4`

Runtime memory: ~19 GB model + KV cache + ~5 GB runtime = no spike observed.

## Root cause: NVIDIA container arch gap

The NVIDIA vLLM containers (25.12.post1-py3, 26.01-py3) ship FlashInfer with precompiled
binaries for sm_120 (data center Blackwell B100/B200), not sm_121 (DGX Spark GB10).

`cicc` is invoked at runtime because:
1. FlashInfer checks for precompiled `.so` matching the GPU arch
2. sm_121a has no match → falls back to JIT from CUDA source
3. CUTLASS MoE GEMM templates are large → each cicc instance uses 1.5-6 GB
4. 6 parallel compilations → 20-30 GB spike on unified memory

Specific to MoE architectures — non-MoE NVFP4 models (e.g. Nemotron-Nano-9B-v2) do not trigger these kernel compilations.

## Open Questions

- Will NVIDIA add sm_121a precompiled kernels to future container releases?
- Does the 26.01-py3 container behave differently? (Same FlashInfer version, likely same issue — confirmed no precompiled sm_121)

## DGX Spark Unified Memory Implications

This issue is specific to unified memory architectures where GPU compute, model weights,
KV cache, and host-side compilation share the same physical memory pool. On discrete GPU
systems (A100, H100, B200), cicc runs on host DRAM while the model runs on GPU VRAM —
they never compete.

Potential mitigations:
1. Shipping precompiled sm_121a kernels in the container
2. Detecting unified memory and serializing cicc processes
3. Documenting the JIT spike for DGX Spark users

---
*Documented March 25, 2026 — DGX Spark, CUDA 13.2, Driver 580.142*
