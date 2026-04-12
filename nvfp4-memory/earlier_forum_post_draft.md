# Nemotron-3-Nano-30B NVFP4 on DGX Spark: Range from 120 GB to 32 GB — what happened?

**TL;DR:** A 19 GB NVFP4 model ran in 32 GB total memory at 50 tok/s on DGX Spark, down from 50-120 GB at vLLM defaults. Flag combination: Marlin backend + enforce_eager + gpu_memory_utilization 0.2.

---

## Observation

Nemotron-3-Nano-30B-A3B-NVFP4 is 19 GB on disk. Running it with vLLM on DGX Spark (GB10, sm_121, 128 GB unified memory) consumed 50-120 GB depending on configuration — 3-6x overhead.

## System

| Component | Version |
|-----------|---------|
| Hardware | NVIDIA DGX Spark (GB10, sm_121) |
| Memory | 128 GB unified (CPU+GPU shared) |
| Host CUDA Toolkit | 13.2.0 (`/usr/local/cuda`) |
| Driver | 580.142 (nvidia-smi reports CUDA 13.0 compat) |
| Container CUDA Toolkit | 13.2 (nvcc V13.2.51) |
| Container | eugr/spark-vllm-docker (`vllm-node:latest`) |
| vLLM | 0.18.1rc1 (March 25, 2026 build) |
| FlashInfer | 0.6.7 (prebuilt sm_121 wheels) |
| PyTorch | 2.12.0+cu130 (compiled against CUDA 13.0 runtime) |

## Contributing factors

Four independent factors observed:

### 1. CUTLASS FP4 kernels fall back on SM121 (+7 GB)

SM121 (DGX Spark GB10) lacks `tcgen05` tensor core instructions. The FlashInfer CUTLASS FP4 backend generates `cvt with .e2m1x2` PTX instructions not supported on sm_121. vLLM's auto-selection picks `FLASHINFER_CUTLASS` because sm_121 has capability ≥100; these kernels fail and fall back to slower, more memory-hungry codepaths.

vLLM source (`nvfp4_utils.py:59-64`):
```python
if current_platform.has_device_capability(100) and has_flashinfer():
    backend = NvFp4LinearBackend.FLASHINFER_CUTLASS  # falls back on sm_121
elif cutlass_fp4_supported():
    backend = NvFp4LinearBackend.VLLM_CUTLASS
elif is_fp4_marlin_supported():
    backend = NvFp4LinearBackend.MARLIN  # runs on sm_121
```

Container log:
```
[Autotuner]: Skipping tactic ... due to failure while profiling:
[TensorRT-LLM][ERROR] Failed to initialize cutlass TMA WS grouped gemm
```

### 2. KV cache pre-allocation (+70-90 GB)

vLLM defaults to `gpu_memory_utilization=0.9`, filling 90% of GPU memory with KV cache. On DGX Spark's 128 GB unified memory, this allocated 89 GB for KV cache — capacity for ~1247 concurrent 8K-token requests.

### 3. torch.compile + CUDA Graphs (+13-20 GB)

Without `--enforce-eager`, vLLM compiles the model with torch.compile and captures CUDA graphs, adding 13-20 GB. On unified memory this overhead competes with the OS.

### 4. FlashInfer JIT Compilation Spike (+20-30 GB transient)

The NVIDIA vLLM container ships sm_120 precompiled FlashInfer kernels, not sm_121. FlashInfer JIT-compiles 6+ CUTLASS MoE GEMM kernels at runtime; each `cicc` compiler process uses 1.5-6 GB RAM. The eugr community container avoids this via prebuilt sm_121 wheels.

## Configuration that worked

### Single-user config

```bash
docker run -d --runtime=nvidia \
    --name nemotron-nvfp4 \
    -v /path/to/hf-cache:/root/.cache/huggingface \
    -p 8000:8000 \
    -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
    -e VLLM_NVFP4_GEMM_BACKEND=marlin \
    -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
    vllm-node:latest \
    python3 -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 --port 8000 \
        --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
        --enforce-eager \
        --max-num-seqs 1 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.2 \
        --kv-cache-dtype fp8 \
        --trust-remote-code
```

### Flag effects measured

| Flag / Env Var | Purpose | Memory delta |
|----------------|---------|:-:|
| `VLLM_NVFP4_GEMM_BACKEND=marlin` | Bypass CUTLASS FP4 fallback | ~7 GB + 16% faster |
| `VLLM_USE_FLASHINFER_MOE_FP4=0` | Disable FlashInfer FP4 MoE path | avoids fallback overhead |
| `VLLM_TEST_FORCE_FP8_MARLIN=1` | Marlin for FP8 paths too | consistent backend |
| `--enforce-eager` | Disable torch.compile + CUDA graphs | ~13 GB |
| `--gpu-memory-utilization 0.2` | Smaller KV cache pre-allocation | ~85 GB (vs 0.9 default) |
| `--max-num-seqs 1` | Single-user | limits concurrent requests |
| `--max-model-len 8192` | Reduce max context (from 256K) | reduces per-request KV |
| `--kv-cache-dtype fp8` | FP8 KV cache (half BF16) | ~50% KV reduction |

## Benchmark Results

All tests on DGX Spark (GB10, sm_121), Nemotron-3-Nano-30B-A3B-NVFP4 (19 GB model), eugr container (`vllm-node:latest`), vLLM 0.18.1rc1.

### Memory Comparison

| Configuration | Loaded GB | Inference GB | Delta from Baseline | KV Cache |
|:---|:---:|:---:|:---:|:---|
| **Marlin + enforce_eager + 0.2 util** | **32.1** | **32.8** | **27.2 GB** | 4.2 GB (292K tokens) |
| FlashInfer default + 0.2 util | 39.0 | 38.6 | 33.0 GB | 3.3 GB (225K tokens) |
| Marlin + CUDA graphs + 0.3 util | 45.6 | 45.6 | 39.5 GB | 15.9 GB (1.1M tokens) |
| Marlin + 0.9 util (default) | 117.3 | 118.0 | 110.2 GB | 89.4 GB (6.2M tokens) |
| *Previous: NVIDIA container defaults* | *~120* | *~120* | *~113 GB* | *~90 GB* |

### Performance Comparison

| Configuration | Warmup (tok/s) | Steady State (tok/s) | Notes |
|:---|:---:|:---:|:---|
| **Marlin + enforce_eager + 0.2 util** | 8.6 | **50.0** | First request slow (model warmup) |
| FlashInfer default + 0.2 util | 8.4 | 42.6 | 16% slower, broken kernels fall back |
| Marlin + CUDA graphs + 0.3 util | 9.1 | 51.6 | +3% speed, +13 GB memory |
| Marlin + 0.9 util (default) | 8.6 | 49.2 | Same speed, 85 GB wasted on KV |

### Observations

1. Marlin backend 16% faster than FlashInfer default on SM121 (FlashInfer falls back to slow CUTLASS codepaths)
2. enforce_eager: 13 GB less memory, 3% slower than CUDA graphs
3. gpu_memory_utilization 0.2 was the minimum working value — 0.15 fails because model (18 GB) + runtime (~5 GB) exceeds 0.15 * 121 GB = 18 GB budget
4. KV cache pre-allocation dominates: default 0.9 allocates 89 GB for 6.2M tokens
5. First request always slow (~8-9 tok/s); model warmup

### Memory Floor Analysis

| gpu_memory_utilization | Total Allowed | KV Available | Status |
|:---:|:---:|:---:|:---|
| 0.01 | 1.2 GB | N/A | FAILED |
| 0.05 | 6.1 GB | -14.0 GB | FAILED |
| 0.15 | 18.2 GB | -1.8 GB | FAILED |
| **0.20** | **24.3 GB** | **4.2 GB** | **Minimum working** |
| 0.30 | 36.5 GB | 15.9 GB | Comfortable |
| 0.50 | 60.8 GB | 40+ GB | Overkill for single-user |
| 0.90 | 109.4 GB | 89.4 GB | Default (massive waste) |

### Comparison with other runtimes

| Runtime | Model Format | Total Memory | Speed | Stable |
|---------|:---:|:---:|:---:|:---:|
| vLLM (this config) | NVFP4 (19 GB) | 32 GB | 50 tok/s | yes |
| vLLM (defaults) | NVFP4 (19 GB) | 120 GB | 49 tok/s | yes |
| llama.cpp | GGUF Q8_0 (34 GB) | 36 GB | 41 tok/s | yes |
| Ollama | default quant (~24 GB) | 26 GB | 49 tok/s | yes |

## SM121 NVFP4 kernel status

As of March 2026, NVFP4 is not natively accelerated on SM121. The Marlin backend dequantizes FP4→BF16 at runtime; not using native FP4 tensor cores. Active PRs:

- CUTLASS #3038: SM121-gated MXFP4 kernel wiring
- vLLM #35947: Software E2M1 conversion for SM12x
- vLLM #38126: Architecture suffix preservation

Community thread with 3400+ views: [PSA: State of FP4/NVFP4 Support for DGX Spark in VLLM](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069)

## SGLang Alternative

NVIDIA's own cookbook supports Nemotron-3-Nano NVFP4 on SGLang:

```bash
python3 -m sglang.launch_server \
    --model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --tp 1 --attention-backend flashinfer \
    --mem-fraction-static 0.3 \
    --trust-remote-code
```

NVIDIA states ≥20 GB VRAM. Requires nightly SGLang for SM121 support.

## Notes for DGX Spark

1. Flush buffer caches before starting inference: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` — unified memory means Linux buffer cache competes with GPU memory
2. Disabling GUI on headless servers: `sudo systemctl set-default multi-user.target` frees ~2-3 GB
3. System tuning: `vm.swappiness=1`, `vm.dirty_bytes=268435456`
4. fastsafetensors with `gpu_memory_utilization > 0.76` caused system freezes in forum reports
5. eugr's prebuilt wheels avoid the FlashInfer JIT compilation spike
6. `/proc/meminfo` for memory — `nvidia-smi` does not report memory usage on GB10 unified memory

## Files in This Repository

| File | Description |
|------|-------------|
| `COMMUNITY_POST.md` | This document |
| `INTERNET_RESEARCH.md` | Comprehensive internet research findings |
| `RESEARCH_STATUS.md` | Original research status and problem statement |
| `FLASHINFER_JIT_ISSUE.md` | FlashInfer JIT compilation analysis |
| `BENCHMARK_RESULTS.md` | Full benchmark results with memory safety guide |
| `benchmark.py` | Benchmark tool (vLLM + Ollama) |
| `mem_monitor.py` | Memory monitoring script for DGX Spark |
| `test_all_paths.py` | Automated test runner for all configurations |
| `run_tests.sh` | Shell-based Docker test runner |
| `test_outputs/` | Raw test results (JSON + logs) |

## Acknowledgments

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — container with prebuilt SM121 wheels
- [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) — official setup guides
- [NVIDIA Forum: We unlocked NVFP4](https://forums.developer.nvidia.com/t/we-unlocked-nvfp4-on-the-dgx-spark-20-faster-than-awq/361163) — source of the Marlin backend flag combination
- DGX Spark community on NVIDIA Developer Forums

---

*Tested March 26, 2026 — DGX Spark GB10, Host CUDA 13.2, Container CUDA 13.2 (torch cu130), Driver 580.142*
*vLLM 0.18.1rc1 (eugr build), FlashInfer 0.6.7, PyTorch 2.12.0+cu130*
