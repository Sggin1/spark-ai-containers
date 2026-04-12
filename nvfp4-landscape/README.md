# NVFP4 on DGX Spark: Landscape Snapshot

**Snapshot:** March 26, 2026 · CUDA 13.0 · Driver 580.142
**System:** NVIDIA DGX Spark (GB10, sm_121, 128 GB unified memory)
**Context:** Trying to run Nemotron-3-Nano-30B-A3B-NVFP4 (19 GB) in ~25 GB total memory instead of 50–120 GB

> This aggregates community reports, forum threads, and tracked PRs as of March 26, 2026. Links go stale fast during the CUDA 13 → 13.2 transition — verify PR merge status and flag names against current releases before acting on anything here. This is a literature snapshot, not a tutorial or recommendation.

---

## Table of Contents

1. [The SM121 FP4 Kernel Problem](#1-the-sm121-fp4-kernel-problem)
2. [Marlin Backend Workaround (community finding)](#2-marlin-backend-workaround-community-finding)
3. [vLLM Memory Optimization Deep Dive](#3-vllm-memory-optimization-deep-dive)
4. [SGLang as Alternative Runtime](#4-sglang-as-alternative-runtime)
5. [TensorRT-LLM Assessment](#5-tensorrt-llm-assessment)
6. [Direct Python Loading (Transformers + TorchAO)](#6-direct-python-loading-transformers--torchao)
7. [NVFP4 Format Lock-In Analysis](#7-nvfp4-format-lock-in-analysis)
8. [Community Memory Usage Reports](#8-community-memory-usage-reports)
9. [Configurations We Tested](#9-configurations-we-tested)
10. [Sources](#10-sources)

---

## 1. The SM121 FP4 Kernel Problem

### Root Cause

SM121 (DGX Spark GB10) **lacks `tcgen05`** (5th generation tensor core instructions) that datacenter Blackwell (SM100/SM110) has. This blocks native FP4 compute.

- CUTLASS Issue #2947: `tcgen05/FP4` hard-restricted to `sm_100a/sm_103a`, rejecting `sm_121`
- CUTLASS Issue #2800: `BlockScaledMmaOp` limits FP4 to SM100 only
- CUTLASS Issue #2802: Expects `sm_100a/sm_100f` but receives SM121a

The FlashInfer CUTLASS FP4 backend generates `cvt with .e2m1x2` PTX instructions that are **not supported on sm_121**. NVFP4 models on DGX Spark fall back to unoptimized codepaths; measured AWQ 4-bit is 32% faster than NVFP4 in this fallback path.

### Performance Impact (from PSA forum thread, 3400+ views)

| Metric | NVFP4 | AWQ 4-bit | Delta |
|--------|:-----:|:---------:|:-----:|
| Single request (tok/s) | 18.91 | 24.93 | AWQ 32% faster |
| 10 concurrent (tok/s) | 35.58 | 42.11 | AWQ 18% faster |
| Inter-token latency | 51.62ms | 39.01ms | AWQ 24% better |

### Active Fixes

| PR | Description | Status |
|----|-------------|--------|
| CUTLASS #3038 | SM121-gated MXFP4 kernel wiring for MoE | In review |
| vLLM #35947 | Software E2M1 conversion for SM12x NVFP4 activation | Merged |
| vLLM #38126 | SM12x architecture suffix preservation (fixes NaN) | Merged |
| Flash-Attention #2222 | SM12x support without tcgen05 | In progress |

**No timeline from NVIDIA for native sm_121 tcgen05 FP4 support.**

Source: [NVIDIA Forum: tcgen05 FP4 support](https://forums.developer.nvidia.com/t/dearest-cutlass-team-when-the-hell-are-you-going-to-properly-fix-tcgen05-fp4-support-for-dgx-spark-gb10-sm121/359598), [PSA: State of FP4/NVFP4](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069)

---

## 2. Marlin Backend Workaround (community finding)

### Source

Credit to the forum post [*"We unlocked NVFP4 on the DGX Spark: 20% faster than AWQ!"*](https://forums.developer.nvidia.com/t/we-unlocked-nvfp4-on-the-dgx-spark-20-faster-than-awq/361163) for identifying this flag combination.

The workaround bypasses the broken CUTLASS FP4 codepath entirely by using the **Marlin backend**, which dequantizes FP4 to BF16 on the fly using operations that work on sm_121.

### Flags

```bash
VLLM_USE_FLASHINFER_MOE_FP4=0 \
VLLM_NVFP4_GEMM_BACKEND=marlin \
VLLM_TEST_FORCE_FP8_MARLIN=1
```

### Reported results (forum post)

- 20% faster than AWQ on DGX Spark
- 60-110 tok/s with speculative decoding (Qwen3-Next-A3B-80B-Instruct-NVFP4)
- No illegal instruction errors
- No CUTLASS JIT compilation overhead

### Notes

- Marlin dequantizes FP4→BF16 at runtime; not using native FP4 tensor cores
- Being integrated into eugr community container

Source: [NVIDIA Forum: We unlocked NVFP4](https://forums.developer.nvidia.com/t/we-unlocked-nvfp4-on-the-dgx-spark-20-faster-than-awq/361163)

---

## 3. vLLM Memory Optimization Deep Dive

### Where the Memory Goes (19 GB model → 50-120 GB)

| Component | Expected | Default vLLM | With Tuning |
|-----------|:--------:|:------------:|:-----------:|
| Model weights (NVFP4) | 19 GB | 19 GB | 19 GB |
| KV cache pool | 2.9 GB (1 seq) | 17-90 GB | 1-3 GB |
| torch.compile / Inductor | 0 | 10-15 GB | 0 (enforce_eager) |
| CUDA graphs | 0 | 5-10 GB | 0 (enforce_eager) |
| FlashInfer JIT (sm_121) | 0 | 20-30 GB spike | 0 (prebuilt wheels) |
| Python + CUDA runtime | 3 GB | 3 GB | 3 GB |
| **Total** | **~25 GB** | **50-120 GB** | **~23-25 GB** |

### Flags that affected memory

**Disable torch.compile + CUDA graphs:**
```
--enforce-eager
```
Confirmed in vLLM 0.18.0 source: sets `CompilationMode.NONE` and `CUDAGraphMode.NONE`.

**Control KV cache precisely (bypass gpu_memory_utilization):**
```
--kv-cache-memory-bytes 2147483648   # Exactly 2 GB
```
New in recent vLLM — overrides `gpu_memory_utilization` entirely. Allows exact byte-level control.

**Alternative: Override block count directly:**
```
--num-gpu-blocks-override 128        # Exact number of KV blocks
```

**Minimize concurrent sequences:**
```
--max-num-seqs 1                     # Single-user mode
--max-model-len 8192                 # Reduce from 262144 default
```

**Optimization level shortcut:**
```
-O0    # Equivalent to --enforce-eager (no compile, no graphs)
```
From RFC #20283: `-O0` = "no compilation, no cudagraphs, just starting up immediately."

### Environment Variables

| Variable | Default | Value used | Purpose |
|----------|---------|-------------|---------|
| `VLLM_USE_FLASHINFER_MOE_FP4` | varies | `0` | Disable FP4 MoE kernels that fall back on sm_121 |
| `VLLM_NVFP4_GEMM_BACKEND` | auto | `marlin` | Marlin backend |
| `VLLM_TEST_FORCE_FP8_MARLIN` | `0` | `1` | Force Marlin for FP4/FP8 |
| `VLLM_DEEP_GEMM_WARMUP` | varies | `skip` | Skip DeepGemm JIT warmup |
| `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE` | 394 MB | reduced | FlashInfer workspace |
| `VLLM_DISABLE_COMPILE_CACHE` | `0` | `0` | Cache left enabled |

### Minimal-memory launch command

```bash
VLLM_USE_FLASHINFER_MOE_FP4=0 \
VLLM_NVFP4_GEMM_BACKEND=marlin \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --enforce-eager \
  --max-num-seqs 1 \
  --max-model-len 8192 \
  --kv-cache-memory-bytes 2147483648 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

Target: ~24 GB total (19 model + 2 KV + 3 runtime). Measured: 32 GB with `--gpu-memory-utilization 0.2` in place of `--kv-cache-memory-bytes`.

Source: [vLLM Engine Args](https://docs.vllm.ai/en/stable/configuration/engine_args/), [vLLM Cache Config](https://docs.vllm.ai/en/stable/api/vllm/config/cache/), [vLLM Conserving Memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)

---

## 4. SGLang as Alternative Runtime

### NVIDIA's Official Support

NVIDIA's own cookbook provides Nemotron-3-Nano NVFP4 on SGLang:

```bash
python3 -m sglang.launch_server \
  --model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --host 0.0.0.0 --port 5000 \
  --tp 1 --attention-backend flashinfer \
  --trust-remote-code \
  --tool-call-parser qwen3_coder \
  --reasoning-parser deepseek-r1
```

NVIDIA states: ≥20 GB VRAM for NVFP4 (vs ≥32 GB FP8, ≥64 GB BF16).

### Memory Control

- `--mem-fraction-static 0.75` — reserves 75% of available memory for KV cache + weights
- Lower values (e.g., `0.3`) reduce total memory footprint
- `--quantization modelopt_fp4` — required for some NVFP4 models (auto-detected for Nemotron)

### DGX Spark notes

- Stable SGLang images crashed on sm_121 with `ptxas` errors
- Nightly builds used: `lmsysorg/sglang:nightly-dev-cu13-*`
- Attention backend: `triton` avoided sm_121 attention bugs (or `flashinfer` if fixed in nightly)
- MoE backend: `--moe-runner-backend flashinfer_cutlass` for sparse MoE
- FP4 GEMM: `--fp4-gemm-backend flashinfer_cudnn`

### Performance (Mistral Small 4 119B NVFP4 on Spark — comparable MoE model)

| Concurrency | tok/s |
|:-----------:|:-----:|
| 1 | 26.78 |
| 8 | 59.9 |
| 16 | 79.4 |

Source: [NVIDIA Nemotron SGLang Cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano/sglang_cookbook.ipynb), [NVIDIA DGX Spark SGLang Playbook](https://build.nvidia.com/spark/sglang), [Forum: Mistral Small 4 on Spark](https://forums.developer.nvidia.com/t/running-mistral-small-4-119b-moe-on-dgx-spark-with-sglang-full-setup-benchmarks/364763)

---

## 5. TensorRT-LLM Assessment

### Memory behavior

TensorRT-LLM is the same class of multi-tenant serving engine as vLLM. It pre-allocates KV cache aggressively (default 90% of remaining GPU memory).

### DGX Spark results (from forum reports)

- Llama-3.3-70B-NVFP4: 5 tok/s (vs theoretical 7.8 tok/s)
- Same model: ~90 GB with TRT-LLM vs 43 GB with LM Studio (GGUF)
- Beta sm_121 support only, single-node configurations validated

For single-user minimal memory, TRT-LLM offered no advantage over vLLM in the measurements cited.

Source: [TRT-LLM slower than LM Studio on Spark](https://forums.developer.nvidia.com/t/trt-llm-for-inference-with-nvfp4-safetensors-slower-than-lm-studio-gguf-on-the-spark/348636), [TRT-LLM DGX Spark Issue #8474](https://github.com/NVIDIA/TensorRT-LLM/issues/8474)

---

## 6. Direct Python Loading (Transformers + TorchAO)

### Motivation

Zero serving-framework overhead — no KV pool pre-allocation, no CUDA graphs, no torch.compile:

```python
from transformers import AutoModelForCausalLM, TorchAoConfig
from torchao.prototype.mx_formats.inference_workflow import (
    NVFP4DynamicActivationNVFP4WeightConfig
)

quant_config = NVFP4DynamicActivationNVFP4WeightConfig(
    use_triton_kernel=True,
    use_dynamic_per_tensor_scale=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    quantization_config=TorchAoConfig(quant_type=quant_config),
    trust_remote_code=True,
)
```

### What we hit

1. **ModelOpt format loading fails in transformers** — `KeyError: 'weight_scale'` (huggingface/transformers#44292). The compressed-tensors decompression path does not handle ModelOpt's format.

2. **Without TorchAO**: `from_pretrained()` silently dequantizes FP4→BF16, inflating memory to ~60 GB.

3. **FPQuantConfig (ISTA-DASLab)**: Different quantization ecosystem — does not load ModelOpt-format checkpoints. Would require re-quantizing from BF16 base model.

4. **Nemotron hybrid architecture** (Mamba-2 + MoE + Attention) adds complexity for direct inference without a serving framework.

Projected ~22 GB footprint if loading worked. Blocked by format compatibility as of this snapshot.

Source: [HuggingFace Transformers Issue #44292](https://github.com/huggingface/transformers/issues/44292), [NVIDIA ModelOpt GitHub](https://github.com/NVIDIA/Model-Optimizer), [NVIDIA Blog: Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

---

## 7. NVFP4 Format Lock-In Analysis

### What's in the Checkpoint

`hf_quant_config.json`:
```json
{
    "producer": {"name": "modelopt", "version": "0.29.0"},
    "quantization": {
        "quant_algo": "NVFP4",
        "kv_cache_quant_algo": "FP8",
        "group_size": 16
    }
}
```

Safetensors (5 shards, ~19.4 GB): FP4 (E2M1) weights + FP8 per-group scales (1 per 16 values) + FP32 per-tensor scales. Two-level scaling is what differentiates NVFP4 from MXFP4.

### Format Conversion Options

| Target | Possible? | Status |
|--------|:---------:|--------|
| GGUF (llama.cpp) | YES | PR #19769 merged March 2026, but **CPU only, no GPU backend** |
| GPTQ | NO | Different algorithm, requires BF16 source |
| AWQ | NO | Different algorithm, requires BF16 source |
| MXFP4 | NO | Different scaling scheme |
| Re-quantize to BF16 | YES | But defeats purpose (~60 GB) |

### Format coupling

NVFP4 ModelOpt checkpoints target NVIDIA's serving stack (vLLM, SGLang, TRT-LLM). The 19 GB size only materializes with framework-specific FP4 CUDA kernels. No lightweight single-user runtime supported native NVFP4 GPU inference at the time of this snapshot.

---

## 8. Community Memory Usage Reports

### DGX Spark Models (from NVIDIA forums)

| Model | Quant | Weights | Total Used | Engine | tok/s |
|-------|:-----:|:-------:|:----------:|--------|:-----:|
| Nemotron-3-Nano-30B | NVFP4 | 19 GB | 50-65 GB | vLLM (default) | 49.6 |
| Nemotron-3-Nano-30B | GGUF Q8 | 34 GB | ~36 GB | llama.cpp | 40.7 |
| Nemotron-3-Nano-30B | default | ~24 GB | ~26 GB | Ollama | 48.7 |
| Nemotron-3-Super-120B | NVFP4 | 69.5 GB | ~103 GB | vLLM+Marlin | 16.6 |
| Mistral Small 4 119B | NVFP4 | ~66 GB | ~99 GB | SGLang | 26.78 |
| Qwen3.5-122B-A10B | NVFP4 | 75.6 GB | ~128 GB | vLLM | varies |

### Community notes (from forum threads)

1. Disabling GUI (`sudo systemctl set-default multi-user.target`) frees ~2-3 GB
2. Flushing buffer cache (`sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`) — unified memory means Linux buffer cache competes with GPU memory
3. System tuning used: `vm.swappiness=1`, `vm.dirty_bytes=268435456`
4. fastsafetensors with `gpu_memory_utilization > 0.76` caused system freezes on unified memory per forum reports
5. Driver 590 required for some newer NVFP4 models (fixes CUDA illegal instruction)
6. FlashInfer race condition causing silent memory corruption at high concurrency reported — forum authors used `flashinfer_cudnn` or Marlin

### eugr/spark-vllm-docker (Updated March 25, 2026)

- 293 commits, actively maintained
- Prebuilt vLLM 0.18.1rc1 + FlashInfer 0.6.7 wheels for sm_121
- Nemotron-3-Nano recipe uses Marlin backend
- `gpu-memory-utilization-gb` mod: specify memory in GiB instead of fraction
- Forum author's guidance: keep model weights under 105 GB for single Spark

---

## 9. Configurations We Tested

These are the flag combinations we exercised — each is a data point, not a recommendation. Results depend on vLLM / FlashInfer / driver snapshot.

### Minimal-memory target (~24 GB)

```bash
# Flush buffer cache first
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Launch vLLM with Marlin backend, minimal memory
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

### Higher-throughput variant

```bash
VLLM_USE_FLASHINFER_MOE_FP4=0 \
VLLM_NVFP4_GEMM_BACKEND=marlin \
VLLM_TEST_FORCE_FP8_MARLIN=1 \
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --load-format fastsafetensors \
  --trust-remote-code
```

### SGLang variant (NVIDIA's documented path)

```bash
python3 -m sglang.launch_server \
  --model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --tp 1 --attention-backend flashinfer \
  --mem-fraction-static 0.3 \
  --trust-remote-code
```

---

## 10. Sources

### NVIDIA Developer Forums
- [tcgen05 FP4 support for DGX Spark](https://forums.developer.nvidia.com/t/dearest-cutlass-team-when-the-hell-are-you-going-to-properly-fix-tcgen05-fp4-support-for-dgx-spark-gb10-sm121/359598)
- [PSA: State of FP4/NVFP4 in VLLM](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069)
- [We unlocked NVFP4: 20% faster than AWQ](https://forums.developer.nvidia.com/t/we-unlocked-nvfp4-on-the-dgx-spark-20-faster-than-awq/361163)
- [DGX Spark Nemotron3 65+ tps](https://forums.developer.nvidia.com/t/dgx-spark-nemotron3-and-nvfp4-getting-to-65-tps/355261)
- [SM121 native NVFP4 compute guidance](https://forums.developer.nvidia.com/t/sm121-gb10-native-nvfp4-compute-seeking-guidance-on-software-support/364607)
- [SM121 software support lacking](https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663)
- [Mistral Small 4 on Spark with SGLang](https://forums.developer.nvidia.com/t/running-mistral-small-4-119b-moe-on-dgx-spark-with-sglang-full-setup-benchmarks/364763)
- [TRT-LLM slower than LM Studio](https://forums.developer.nvidia.com/t/trt-llm-for-inference-with-nvfp4-safetensors-slower-than-lm-studio-gguf-on-the-spark/348636)

### NVIDIA Official
- [DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
- [Nemotron SGLang Cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Nano/sglang_cookbook.ipynb)
- [NVIDIA Blog: Software Optimizations Supercharge DGX Spark](https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark/)
- [NVIDIA Blog: Introducing NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer)
- [SGLang on DGX Spark](https://build.nvidia.com/spark/sglang)

### vLLM Documentation
- [Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)
- [Cache Config](https://docs.vllm.ai/en/stable/api/vllm/config/cache/)
- [Conserving Memory](https://docs.vllm.ai/en/latest/configuration/conserving_memory/)
- [RFC: Overhaul CompilationConfig and -O levels](https://github.com/vllm-project/vllm/issues/20283)

### Community
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
- [llama.cpp NVFP4 GGUF PR #19769](https://github.com/ggml-org/llama.cpp/pull/19769)
- [HuggingFace Transformers NVFP4 Issue #44292](https://github.com/huggingface/transformers/issues/44292)
- [SGLang FP4 Issue #11725](https://github.com/sgl-project/sglang/issues/11725)
- [SGLang-NVIDIA Q1 2026 Roadmap](https://github.com/sgl-project/sglang/issues/17130)

---

---

## 11. Our Test Results (March 26, 2026)

All tests validated on DGX Spark GB10, sm_121, using eugr's `vllm-node:latest` container (vLLM 0.18.1rc1, FlashInfer 0.6.7, PyTorch 2.12.0+cu130).

### Backend env vars that worked

```
VLLM_USE_FLASHINFER_MOE_FP4=0
VLLM_NVFP4_GEMM_BACKEND=marlin
VLLM_TEST_FORCE_FP8_MARLIN=1
```

Container log: `"Using NvFp4LinearBackend.MARLIN for NVFP4 GEMM"`.

### Memory results

| Config | Total Memory | tok/s |
|--------|:-:|:-:|
| Marlin + enforce_eager + 0.2 util | **32 GB** | **50.0** |
| FlashInfer default + 0.2 util | 39 GB | 42.6 |
| Marlin + CUDA graphs + 0.3 util | 46 GB | 51.6 |
| Default (0.9 util) | 117 GB | 49.2 |

### Minimum floor

`gpu_memory_utilization=0.2` was the minimum working value. Lower values fail because model (18 GB) + runtime (~5 GB) exceeds the total allowed memory budget.

### FlashInfer CUTLASS fallback observed

Container logs show: `"Failed to initialize cutlass TMA WS grouped gemm"` on sm_120 kernels. Auto-tuner skips failing tactics and falls back; measured 16% slower inference and 7 GB more memory vs Marlin.

*Snapshot compiled March 26, 2026 — DGX Spark, CUDA 13.0, Driver 580.142*
