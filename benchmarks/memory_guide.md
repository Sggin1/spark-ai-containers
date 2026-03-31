# DGX Spark Memory Issues — Root Causes and Solutions

Three distinct memory problems, often confused. Each has a different cause and fix.

## Issue 1: Memory CREEP (progressive growth over time)

**Symptom:** Memory climbs from 50 → 65 → 80 → 100+ GB over minutes/hours.
Eventually hits swap, system freezes.

**Root causes:**
- `torch.compile` + CUDA graph capture: 13-20 GB growing as code paths compile
- FlashInfer JIT compilation: 6 parallel `cicc` processes × 1.5-6 GB each = 20-30 GB spike
- CUTLASS FP4 kernel fallback: broken on SM121, triggers slow/memory-hungry codepaths

**Solutions (all applied in our config):**
- `--enforce-eager` — disables torch.compile and CUDA graphs
- `vllm-node` container (eugr) — prebuilt sm_121 FlashInfer wheels, zero JIT
- `VLLM_NVFP4_GEMM_BACKEND=marlin` — bypass broken CUTLASS FP4 kernels
- `VLLM_USE_FLASHINFER_MOE_FP4=0` — disable FlashInfer FP4 MoE path

**Status:** SOLVED. Zero creep confirmed across 8+ queries. Memory decreased 1.2 GB.

## Issue 2: KV Cache OVER-PROVISIONING (high baseline, but stable)

**Symptom:** Memory is stable but unnecessarily high (58 GB for single-user inference
on a 19 GB model).

**Root cause:** vLLM defaults to multi-tenant serving. `gpu_memory_utilization` controls
what fraction of GPU memory to fill with KV cache. On DGX Spark's 128 GB unified memory:
- 0.9 (default) = 89 GB KV cache = 6.2M tokens = 1247 concurrent 8K requests
- 0.35 = 23 GB KV cache = 1.6M tokens = 330 concurrent 8K requests
- 0.20 = 4.2 GB KV cache = 292K tokens = enough for single user

For single-user inference, you need maybe 8K-32K tokens of KV cache, not millions.

**Solution:** `--gpu-memory-utilization 0.2` for fp8, `0.25-0.35` for tq3 (TurboQuant
buffers need extra headroom).

**Caveat on unified memory:** `gpu_memory_utilization` is a hint based on
`torch.cuda.mem_get_info()`. On GB10 unified memory, the reported "total" and "free"
don't behave like discrete GPU VRAM. The actual allocation may differ from the target.

**Status:** Understood, tunable. Current tq3 config at 0.25 uses ~55 GB total.

## Issue 3: STARTUP SPIKE (transient, then settles)

**Symptom:** Memory spikes to 70-80 GB during model loading, then drops to steady state.

**Root cause:** Model loading reads safetensors from disk into CPU memory, then transfers
to GPU. On unified memory, both "CPU" and "GPU" allocations compete for the same 128 GB.
The spike is: model on disk (mmap) + model in transfer buffer + model in final location.

**Solution:** Flush page cache before starting: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`

Our `safe_run.sh` does this automatically.

**Status:** Expected behavior. Spike resolves within 60-90 seconds of startup.

## Memory Budget Summary

| Component | FP8 Baseline | TQ3 Current | Notes |
|-----------|:-----------:|:----------:|-------|
| Model weights (NVFP4) | 19 GB | 19 GB | Same |
| Runtime (Python, CUDA ctx) | ~3 GB | ~3 GB | Same |
| Marlin dequant buffers | ~2 GB | ~2 GB | Same |
| TurboQuant buffers | 0 | ~8-10 GB | Rotation matrices, centroids per layer |
| KV cache pool | 4.2 GB (0.2) | 23-25 GB (0.25-0.35) | Over-provisioned, tunable |
| **Total** | **~32 GB** | **~55-58 GB** | TQ overhead is ~25 GB over fp8 |

The TurboQuant overhead (~25 GB) is split between:
- Pre-allocated buffers per attention layer (~8-10 GB)
- Higher KV cache allocation needed to pass vLLM's minimum check (~15 GB)

The actual KV cache *per token* is smaller with tq3 (3 bits vs 8 bits = 62.5% savings),
but the minimum allocation threshold is higher due to the packed format.

## Quick Reference: Safe Configs

**FP8 baseline (32 GB, proven stable):**
```bash
--enforce-eager --max-num-seqs 1 --max-model-len 8192 \
--gpu-memory-utilization 0.2 --kv-cache-dtype fp8
```

**TurboQuant tq3 (55-58 GB, proven stable):**
```bash
--enforce-eager --max-num-seqs 1 --max-model-len 8192 \
--gpu-memory-utilization 0.25 --kv-cache-dtype tq3
```

Both require: `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_USE_FLASHINFER_MOE_FP4=0`,
`VLLM_TEST_FORCE_FP8_MARLIN=1`, and the eugr community container (or turboquant-patched).
