# Nemotron-3-Nano-30B-A3B-NVFP4 on DGX Spark — Research Status

## Date: March 25, 2026

## The Core Problem

A 19 GB NVFP4 model required >50 GB of system memory on DGX Spark (128 GB unified memory) with stock vLLM settings. Runtime overhead from the serving stack, not the model format.

Steady-state inference measured at 49.6 t/s.

## System

- NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory)
- CUDA 13.2.51, Driver 580.142
- Ubuntu 24.04, Kernel 6.17.0-1014-nvidia

## What We Tried

### 1. NVIDIA vLLM Container (nvcr.io/nvidia/vllm:25.12.post1-py3)

**Result:** Ran, but used 109+ GB at defaults, 50-65 GB with conservative settings.

Observations:
- Container ships sm_120 precompiled FlashInfer kernels, NOT sm_121
- FlashInfer JIT-compiles 6+ CUTLASS MoE GEMM kernels at runtime
- Each `cicc` compiler process uses 1.5-6 GB RAM (6 in parallel = 20-30 GB spike)
- JIT spike happens on unified memory, competing with model + OS
- `--enforce-eager` does NOT prevent FlashInfer JIT (only disables CUDA graphs)
- `--compilation-config '{"level":0}'` does NOT prevent FlashInfer JIT
- `--gpu-memory-utilization` only caps KV cache pool, not compilation overhead
- torch.compile/inductor adds additional memory on top of JIT

Tested configurations:

| gpu-util | enforce-eager | compilation | max-model-len | Peak GB Used |
|:--------:|:-------------:|:-----------:|:-------------:|:------------:|
| 0.9 | no | default | 262144 | ~120 |
| 0.5 | yes | default | 32768 | ~65 |
| 0.5 | yes | level=0 | 32768 | ~60 |
| 0.3 | no | default | 32768 | 50+ (killed) |

### 2. Community Container (eugr/spark-vllm-docker — vllm-node)

**Result:** No JIT spike (0 cicc processes), but still 50 GB at 0.3 util.

- Built from source with `FLASHINFER_CUDA_ARCH_LIST="12.1a"` and `TORCH_CUDA_ARCH_LIST="12.1a"`
- Ships prebuilt sm_121a FlashInfer wheels — no runtime compilation
- Build time: 3 minutes (uses prebuilt wheels from GitHub releases)
- BUT: the prebuilt FlashInfer ships `fused_moe_120` kernels (sm_120), not sm_121
- Non-fatal: falls back to working kernels, inference works
- torch.compile + CUDA graph capture still adds ~20 GB overhead
- At 0.3 util: model 19 GB + vLLM overhead = 50 GB

### 3. Non-vLLM Approaches (for comparison)

| Backend | Model Format | Total Memory | Speed | Stable? |
|---------|-------------|:---:|:---:|:---:|
| llama.cpp (sm_121 native) | GGUF Q8_0 (34 GB) | ~36 GB | 40.7 t/s | YES |
| Ollama | default quant (~24 GB) | ~26 GB | 48.7 t/s | YES |
| vLLM (any container) | NVFP4 (19 GB) | 50-120 GB | 49.6 t/s | NO (memory) |

## Gap

NVFP4 (19 GB, 49.6 t/s) is currently only loadable via vLLM, a multi-tenant serving engine that pre-allocates aggressively. No lightweight runtime (llama.cpp, Ollama) supports NVFP4 format as of this snapshot.

## Unsolved Questions

1. **Is there a vLLM configuration we're missing?** A flag to disable ALL pre-allocation, torch.compile, CUDA graphs, and run in minimal single-user mode?
2. **Can vLLM run without torch.compile?** The community container avoids FlashInfer JIT but torch.compile still adds ~15-20 GB.
3. **Is TensorRT-LLM leaner?** NVIDIA's own inference engine — does it handle NVFP4 on sm_121 with less overhead?
4. **Can the NVFP4 format be loaded by other runtimes?** ModelOpt FP4 → GGUF conversion? Direct loading via transformers?
5. **What are other DGX Spark users doing?** The 65 t/s forum post used eugr's container — did they see the same memory bloat?
6. **Is there a Python-only path?** Load the model with transformers + modelopt, run inference directly without a serving framework?

## Model Architecture Details

Nemotron-3-Nano-30B-A3B (NemotronH):
- 52 layers: 29 Mamba-2 (56%) + 23 Attention (44%)
- Attention: 32 query heads, 2 KV heads (GQA 16:1), 128 head dim
- Mamba: 64 heads, 64 head dim, SSM state 128
- MoE with shared experts
- NVFP4: ModelOpt FP4 weights + FP8 KV cache
- max_position_embeddings: 262144 (256K)

KV cache per sequence (FP8): 2.9 GB at 256K context
Mamba SSM state: 60 MB fixed (regardless of context)

## Expected vs Actual Memory

| Component | Expected | Actual |
|-----------|:--------:|:------:|
| Model weights | 19 GB | 19 GB |
| KV cache (1 seq, 256K) | 2.9 GB | 2.9 GB (but pool pre-allocated for many) |
| Mamba state | 0.06 GB | 0.06 GB |
| Runtime (Python, CUDA ctx) | ~3 GB | ~3 GB |
| torch.compile | 0 (should be optional) | ~10-15 GB |
| CUDA graphs | 0 (should be optional) | ~5-10 GB |
| FlashInfer JIT (NVIDIA container) | 0 (should be precompiled) | 20-30 GB transient |
| KV pool pre-allocation | 2.9 GB (single user) | 17-90 GB (multi-tenant) |
| **TOTAL** | **~26 GB** | **50-120 GB** |

## Target configuration

A runtime that:
1. Loads the 19 GB NVFP4 model
2. Allocates KV cache on demand (not pre-allocated pool)
3. No torch.compile overhead
4. No CUDA graph capture overhead
5. Precompiled sm_121 kernels
6. Total memory: ~25 GB idle, ~28 GB at 256K context

Would fit NVFP4 on 24 GB consumer GPUs.

## Files in This Project

- `BENCHMARK_RESULTS.md` — Full benchmarks with memory safety guide
- `FLASHINFER_JIT_ISSUE.md` — FlashInfer JIT compilation analysis
- `RESEARCH_STATUS.md` — This file
- `benchmark.py` — Benchmark tool (vLLM + Ollama)
- `benchmark_results.json` — Raw benchmark data
- `run_nvfp4.sh` — Launch script with safety notes
- `spark-vllm-docker/` — Community container (eugr)

---
*March 25, 2026 — DGX Spark, CUDA 13.2*
