# NVFP4 on DGX Spark: Memory Footprint Investigation

> **⚠️ UPDATED 2026-06-05 — facts below are partly superseded.** This snapshot dates to **March 26 2026**, the week the sm_121 NVFP4 fixes landed. Since then: stock upstream **vLLM ≥ v0.19.0 (now v0.22.x) builds working sm_121 NVFP4 kernels — no eugr fork needed**; a **native SM120/121 CUTLASS NVFP4 GEMM** landed (vLLM PR #40082, merged 2026-05-20); the `VLLM_NVFP4_GEMM_BACKEND` / `VLLM_USE_FLASHINFER_MOE_FP4` env vars are **deprecated** (still functional, emit FutureWarning; replaced by `--linear-backend` / `--moe-backend`) and backend choice is now **per-model** — in current top recipes text Nemotron NVFP4 enables FlashInfer MoE while multimodal Omni still pins Marlin; **llama.cpp NVFP4 is now GPU-accelerated**. The memory-tuning findings (KV pre-allocation, `--enforce-eager`, util 0.2) still hold. Wrong/dangerous facts are corrected in place below; full diff in [`NVFP4_UPDATE_PLAN.md`](../NVFP4_UPDATE_PLAN.md). Current host baseline: **Driver 580.159.03 · CUDA 13.0.2** (DGX OS 7.5.0).

**Snapshot:** March 26, 2026 · CUDA 13.0 · Driver 580.142 · vLLM 0.18.1rc1 · FlashInfer 0.6.7
**Hardware:** DGX Spark (GB10, sm_121, 128 GB unified memory)
**Model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` (19 GB on disk)

> This is a point-in-time lab notebook, not a recommendation or standing claim. The CUDA 13 → 13.2 transition and weekly vLLM/FlashInfer releases mean findings may be stale within days. Dates and SHAs are included so anyone reproducing can compare against their own snapshot.

## TL;DR

- A 19 GB NVFP4 model uses **50–120 GB** on Spark with default vLLM settings. We reduced this to **~32 GB** with Marlin backend + `--enforce-eager` + `gpu-memory-utilization=0.2`, at **50 tok/s**.
- The memory bloat is not the NVFP4 format — it's vLLM's multi-tenant defaults (KV pool pre-allocation, torch.compile, CUDA graphs) plus FlashInfer JIT on Spark's unified memory.
- On SM121 specifically, CUTLASS FP4 kernels fall back (no `tcgen05`). Marlin works by dequantizing FP4 → BF16 at runtime on instructions that SM121 supports.
- Minimum working `gpu-memory-utilization` was `0.2`. Below that, model + runtime exceeds the allowed budget.

## Context

A 19 GB model fits a 24 GB consumer GPU in principle. The 50+ GB observed on a 128 GB box is an artifact of the serving stack, not the quantization format.

## The test matrix

All runs on eugr's `vllm-node:latest` container (prebuilt sm_121a FlashInfer wheels, no runtime JIT spike):

| Config | Peak Memory | Speed |
|---|---:|---:|
| Marlin + `--enforce-eager` + util 0.2 | **32 GB** | **50.0 t/s** |
| FlashInfer default + util 0.2 | 39 GB | 42.6 t/s |
| Marlin + CUDA graphs + util 0.3 | 46 GB | 51.6 t/s |
| vLLM default (util 0.9) | 117 GB | 49.2 t/s |

Container log confirms backend selection:
```
Using NvFp4LinearBackend.MARLIN for NVFP4 GEMM
```

FlashInfer CUTLASS path logs (when not forced off):
```
Failed to initialize cutlass TMA WS grouped gemm
```
Auto-tuner skips the broken tactics and falls back. Net cost: ~16% slower, ~7 GB more memory vs Marlin.

## Launch command used for the 32 GB config

```bash
# Clear page cache first — unified memory means buffer cache competes with GPU
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

VLLM_USE_FLASHINFER_MOE_FP4=0 \
VLLM_NVFP4_GEMM_BACKEND=marlin \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --enforce-eager \
  --max-num-seqs 1 \
  --max-model-len 8192 \
  --kv-cache-memory-bytes 2147483648 \
  --kv-cache-dtype fp8 \
  --trust-remote-code
```

Target: ~24 GB total (19 GB weights + 2 GB KV + 3 GB runtime). Measured 32 GB using `--gpu-memory-utilization 0.2` instead of `--kv-cache-memory-bytes`; the byte-exact flag is newer and we haven't fully re-run with it.

## Where the memory goes

| Component | Default vLLM | With minimal flags |
|---|---:|---:|
| Model weights (NVFP4) | 19 GB | 19 GB |
| KV cache pool | 17–90 GB | 1–3 GB |
| torch.compile / Inductor | 10–15 GB | 0 (`--enforce-eager`) |
| CUDA graphs | 5–10 GB | 0 (`--enforce-eager`) |
| FlashInfer JIT | 20–30 GB spike | 0 (prebuilt sm_121a wheels) |
| Python + CUDA runtime | 3 GB | 3 GB |
| **Total** | **50–120 GB** | **~23–32 GB** |

## The SM121 FP4 kernel situation

SM121 (GB10) lacks `tcgen05` — the 5th-gen tensor core instructions that SM100/SM110 datacenter Blackwell has. CUTLASS FP4 paths generate `cvt with .e2m1x2` PTX that isn't supported, so they fall back.

Tracked issues (as of March 26, 2026):
- CUTLASS #2947, #2800, #2802 — FP4 kernels hard-restricted to `sm_100a/sm_103a`
- CUTLASS #3038 — SM121-gated MXFP4 kernel wiring for MoE (in review)
- vLLM #37725 — preserve CUDA arch suffix for SM12x, fixes NVFP4 NaN (merged 2026-03-25; supersedes #35947, which was **closed unmerged**)
- vLLM #40082 — native SM120/121 CUTLASS NVFP4 GEMM + opt-in b12x kernel (merged 2026-05-20)
- vLLM #38126 — SM12x architecture suffix preservation, fixes NaN (merged)
- Flash-Attention #2329 / #2330 / #2333 — SM120 fwd/bwd/varlen support **merged 2026-03-12/13** (Blackwell GeForce / DGX Spark); draft #2222 closed unmerged

No public NVIDIA timeline for native SM121 `tcgen05` FP4 support.

## Open items

- `gpu-memory-utilization` below 0.2 failed: model + runtime exceeds the budget. `--kv-cache-memory-bytes` may bypass this but we haven't fully validated.
- Direct transformers + TorchAO path blocked: `KeyError: 'weight_scale'` on ModelOpt checkpoints (huggingface/transformers#44292).
- GGUF conversion exists; **UPDATE June 2026: llama.cpp now runs NVFP4 GGUF on GPU** (CUDA dp4a + native Blackwell FP4 covering sm_120/121, PRs #20644/#22196; also SYCL/Vulkan). The "CPU-only" limitation (PR #19769) was snapshot-era. Still a requantize, not a transcode.
- SGLang stable crashed on sm_121. Nightly builds (`lmsysorg/sglang:nightly-dev-cu13-*`) with `--attention-backend triton` ran.
- TensorRT-LLM: same class of multi-tenant engine as vLLM; forum-reported ~90 GB for Llama-3.3-70B-NVFP4 on Spark.

## Reproducing this

1. Pull eugr's container: https://github.com/eugr/spark-vllm-docker (was at 293 commits as of our test)
2. Drop page cache before launch (unified memory, buffer cache competes)
3. Use the minimal-memory command above
4. Watch `nvidia-smi` and `free -g` during warmup — the spike patterns are informative

## Credits and context

Builds on prior community work:
- eugr's `spark-vllm-docker` — the container we used; prebuilt sm_121a wheels avoid the JIT spike
- Forum post "We unlocked NVFP4 on the DGX Spark: 20% faster than AWQ" — source of the Marlin backend flag combination
- NVIDIA forum threads on PSA state of FP4/NVFP4 (3400+ views) — AWQ-vs-NVFP4 measurements
- vLLM RFC #20283 — `-O0` shortcut for no-compile, no-graphs startup

Full source list: see [../2026-03-26-landscape-snapshot/README.md](../2026-03-26-landscape-snapshot/README.md#10-sources)

## Scope disclaimer

- Single model tested (Nemotron-3-Nano 30B-A3B-NVFP4). Results may not generalize to other NVFP4 models.
- Single container/vLLM/FlashInfer snapshot. Behavior has likely shifted already given release cadence.
- On SM121 only. SM100/SM110 have working CUTLASS FP4 and a different optimization landscape.
- No quality evaluation (perplexity, lm-eval). Speed and memory only.

---

*Part of an ongoing investigation into NVFP4 inference on DGX Spark. Raw logs, earlier experiments, and follow-up tests will be added to this folder over time.*
