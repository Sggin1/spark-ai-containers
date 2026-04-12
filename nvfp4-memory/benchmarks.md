# Nemotron-3-Nano-30B-A3B-NVFP4 — DGX Spark SM_121 Benchmarks

## System Configuration

| Component | Value |
|-----------|-------|
| **Hardware** | NVIDIA DGX Spark |
| **GPU** | NVIDIA GB10 (Grace Blackwell) |
| **Compute Capability** | sm_121 (12.1) |
| **Unified Memory** | 128 GB (121.6 GB available to userspace) |
| **CPU** | 20-core ARM Grace |
| **CUDA Toolkit** | 13.2.51 |
| **Driver** | 580.142 |
| **Kernel** | 6.17.0-1014-nvidia |
| **OS** | Ubuntu 24.04 |

---

## Memory behavior on DGX Spark

Note: `--gpu-memory-utilization` and `--max-model-len` defaults consume nearly all system memory on unified-memory systems.

### Observation

DGX Spark uses unified memory — GPU and CPU share the same 128 GB pool. Unlike discrete GPU systems (A100, H100) where vLLM only sees GPU VRAM, on DGX Spark vLLM sees the entire 121.6 GB as "GPU memory".

With defaults (`--gpu-memory-utilization 0.9`), vLLM claimed 109 GB at startup, leaving <1 GB for OS and other processes. Benchmark ran at 94% utilization (0.7 GB free).

### Where the Memory Actually Goes

The Nemotron-3-Nano model uses a **hybrid Mamba/Attention architecture** (NemotronH):

```
Layer pattern: M.E.M.E.M.M.E.M.E.M.E.M.M.E.M.E.M.E.M.M.E.M.E.M.E.M...
               29 Mamba (56%) + 23 Attention (44%) = 52 total layers
```

Only the 23 attention layers need KV cache. The 29 Mamba layers use a fixed-size SSM state (~60 MB total, regardless of context length). This is why the model can claim 256K context without enormous memory — but vLLM doesn't care about *your* context length, it pre-allocates for *concurrent serving*.

**KV cache per single sequence (FP8):**

| Context Length | KV Cache | Note |
|:-:|:-:|---|
| 4K | 0.02 GB | Tiny |
| 8K | 0.05 GB | Tiny |
| 32K | 0.4 GB | Reasonable |
| 64K | 0.7 GB | Still small |
| 128K | 1.4 GB | Moderate |
| 256K | 2.9 GB | Full context |

Small numbers due to the hybrid architecture (only 23 attention layers with GQA: 2 KV heads, 128 head dim). Memory bloat comes from vLLM's serving-side pre-allocation, not the model.

vLLM pre-allocates a pool for concurrent sequences at startup:

| `--gpu-memory-utilization` | vLLM Budget | KV Pool | ~Concurrent 256K Seqs | Free for OS |
|:-:|:-:|:-:|:-:|:-:|
| 0.9 (default) | 109 GB | ~74 GB | ~26 | 12 GB |
| 0.7 | 85 GB | ~50 GB | ~17 | 36 GB |
| 0.5 | 61 GB | ~26 GB | ~9 | 61 GB |
| 0.3 | 37 GB | ~2 GB | ~1 | 85 GB |

**Full memory breakdown at default settings:**

| Component | Size | Notes |
|-----------|:----:|-------|
| Model weights (NVFP4) | 19 GB | FP4 weights, loaded from 5 safetensor shards |
| KV cache pool (FP8) | ~74 GB | Pre-allocated for concurrent serving |
| CUDA graphs | ~8 GB | Compiled for 50+ batch sizes (one-time ~5min cost) |
| Torch compile cache | ~3 GB | Inductor-compiled kernels |
| Python/CUDA runtime | ~5 GB | vLLM process, CUDA context, libraries |
| **Total vLLM** | **~109 GB** | At `gpu-memory-utilization=0.9` |
| OS + desktop + apps | ~6 GB | What's left |

### Tested configurations

Single-user / light workload:

```bash
--gpu-memory-utilization 0.5 --max-model-len 32768
```

Measured with this config:
- ~61 GB free for OS and other processes
- 32K context
- ~9 concurrent request slots
- Throughput equal to or slightly better than 0.9 default

Dedicated inference server:

```bash
--gpu-memory-utilization 0.7 --max-model-len 131072
```

Defaults on DGX Spark leave <1 GB for the OS.

---

## NVFP4 Benchmark Results

**Container:** `nvcr.io/nvidia/vllm:25.12.post1-py3` (vLLM 0.12.0)
**Quantization:** NVFP4 (ModelOpt FP4 weights + FP8 KV cache)
**Model Size on Disk:** 19 GB
**Model Memory:** 18.6 GB loaded
**Environment:** `VLLM_USE_FLASHINFER_MOE_FP4=1`, `VLLM_FLASHINFER_MOE_BACKEND=throughput`
Note: results obtained with vLLM defaults (`gpu-memory-utilization=0.9`, `max-model-len=262144`).

### Steady-State Performance (Run 2, post-warmup)

| Prompt | Prompt Tokens | Response Tokens | Time (s) | Tokens/sec |
|--------|:------------:|:---------------:|:--------:|:----------:|
| Coding (Fibonacci) | 16 | 512 | 10.34 | **49.5** |
| Reasoning (Monopoly) | 18 | 223 | 4.54 | **49.1** |
| Knowledge (Biology) | 10 | 512 | 10.32 | **49.6** |
| Math (Odd Sum) | 17 | 512 | 10.24 | **50.0** |
| **Average** | — | — | — | **49.6** |

### First-Run Performance (includes CUDA graph compilation)

| Prompt | Tokens/sec | Note |
|--------|:----------:|------|
| Coding | 19.1 | Initial graph compilation penalty |
| Reasoning | 45.3 | Warming up |
| Knowledge | 50.4 | Steady state |
| Math | 50.2 | Steady state |
| **Average** | **41.3** | First run after cold server start |

First-run penalty ~8 minutes total (model load + CUDA graph capture for 50+ batch sizes). Cached — subsequent server restarts reuse the compile cache.

---

## Cross-Quantization Comparison (same prompts, same hardware)

| Model | Quantization | Backend | Avg t/s | Model Size |
|-------|:----------:|---------|:-------:|:----------:|
| Nano 30B-A3B | NVFP4 | vLLM 0.12 (docker) | 49.6 | 19 GB |
| Nano 30B-A3B | Q8_0 (GGUF) | llama.cpp sm121 | 40.7 | 34 GB |
| Nano (Ollama) | default | Ollama 0.18.2 | 48.7 | 24 GB |
| Super 120B | Q4_K (GGUF) | llama.cpp sm121 | 14.4 | 70 GB |

NVFP4 vs Q8_0 GGUF: 22% faster, 44% smaller on disk.

---

## Model Architecture Notes

Nemotron-3-Nano-30B-A3B is a hybrid **Mamba-2 / Attention** model (NemotronH):

- **52 layers total:** 29 Mamba-2 (56%) + 23 Attention (44%)
- **Attention config:** 32 query heads, 2 KV heads (GQA 16:1), 128 head dim
- **Mamba config:** 64 heads, 64 head dim, SSM state size 128, conv kernel 4
- **MoE:** Mixture-of-Experts with shared experts (separate CUDA stream)
- **NVFP4:** ModelOpt FP4 quantization with FlashInfer CUTLASS kernels for MoE GEMM

The hybrid architecture means:
- **Attention layers** scale linearly with context length (KV cache)
- **Mamba layers** have constant memory regardless of context (SSM state = ~60 MB fixed)
- This is why 256K context is feasible on 128 GB — most layers don't need KV cache

---

## Observations

1. NVFP4 ran on sm_121 with CUDA 13.2 using the NVIDIA vLLM container with FlashInfer CUTLASS kernels (CUDA 13.0 kernel incompatibility seen previously)
2. ~50 t/s steady-state; forum reports cite 65+ t/s with tuned `--gpu-memory-utilization` and `--max-model-len`
3. vLLM default `gpu-memory-utilization=0.9` claimed 109 GB on DGX Spark, leaving <1 GB for the OS — set explicitly on unified-memory systems
4. 19 GB for 30B parameters; 2.9 GB KV cache at full 256K context (hybrid Mamba/Attention architecture)
5. vLLM pre-allocates for concurrent serving — the ~90 GB "KV cache" is capacity for ~26 simultaneous 256K sessions
6. First-run penalty ~8min for CUDA graph compilation (50+ batch sizes), cached after first run
7. FP8 KV cache halves attention cache footprint

---

## Reproduction Steps

```bash
# 1. Requirements
# - DGX Spark with CUDA 13.2+ (apt install cuda-toolkit-13-2)
# - Driver 580.x+
# - Docker installed

# 2. Pull the container
docker pull nvcr.io/nvidia/vllm:25.12.post1-py3

# 3. Download the model
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4

# 4. Start the server (conservative settings for DGX Spark)
sudo docker run -d --name nemo3-nvfp4 \
  --gpus all --network host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_USE_FLASHINFER_MOE_FP4=1 \
  -e VLLM_FLASHINFER_MOE_BACKEND=throughput \
  -e HF_HUB_OFFLINE=1 \
  nvcr.io/nvidia/vllm:25.12.post1-py3 \
  vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 32768 \
    --port 8000

# 5. Wait ~8min for model load + CUDA graph compilation
#    Monitor with: docker logs -f nemo3-nvfp4
#    Ready when you see: "Application startup complete"

# 6. Test
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
       "prompt":"Hello, what is 2+2?","max_tokens":64,"temperature":0}'

# 7. Stop when done (frees ~60 GB)
docker stop nemo3-nvfp4 && docker rm nemo3-nvfp4
```

## Software Versions

- vLLM: 0.12.0+35a9f223.nv25.12.post1 (NVIDIA NGC container)
- Container CUDA: 13.1 (forward compat with host driver 580.142)
- Host CUDA Toolkit: 13.2.51
- Ollama: 0.18.2 (for GGUF comparison runs)
- llama.cpp: native sm_121 build (commit 463b6a963)

---
*Benchmarked on NVIDIA DGX Spark, March 2026*
