# DGX Spark: Running Nemotron-3-Super 120B with llama.cpp (sm_121 native build)

**TL;DR:** Built llama.cpp from source targeting `sm_121` (Blackwell) on a DGX Spark GB10. Generation speed ~14.4 t/s vs Ollama's ~14.2 t/s. The sm_121 build was not meaningfully faster. The ggml-org Q4_K GGUF used ~20 GB less memory than Ollama's Q4_K_M, relevant on shared unified memory. Build recipe, benchmarks, and OOM workarounds below.

---

## Setup

- **Hardware:** NVIDIA DGX Spark — GB10 Grace Blackwell, 128 GB unified VRAM, 20-core ARM
- **OS:** Ubuntu 24.04, kernel 6.17.0-1008-nvidia
- **CUDA:** 13.0 / Driver 580.126.09
- **Model:** [ggml-org/nemotron-3-super-120b-GGUF](https://huggingface.co/ggml-org/nemotron-3-super-120b-GGUF) (Q4_K, 66 GB)
- **llama.cpp commit:** `463b6a963`

## Why build llama.cpp from source?

The GB10 has compute capability **12.1** (Blackwell). Ollama ships llama.cpp compiled for `sm_120` via its `cuda_v12` library. Building from source with `-DCMAKE_CUDA_ARCHITECTURES="121"` targets the exact hardware.

The performance difference we measured was within noise. For a 120B MoE model, generation is memory-bandwidth bound, not compute bound. Both sm_120 and sm_121 hit the same memory wall. Reasons to do a native build:

1. Use the upstream GGUF (Q4_K, 66 GB) instead of Ollama's Q4_K_M (86 GB) — saves 20 GB
2. Control over context length, flash attention, and server configuration
3. When CUDA 13.2+ arrives, sm_121 native builds may enable NVFP4 dispatch

## Results

| | Ollama (Q4_K_M) | llama.cpp (Q4_K) |
|---|---|---|
| CUDA arch | sm_120 | sm_121 |
| Model size on disk | ~86 GB | ~66 GB |
| Runtime memory | ~86 GB | ~73 GB |
| Avg generation | **14.18 t/s** | **14.43 t/s** |

~2% difference, within measurement noise. No 40% speedup from sm_121 was observed. An earlier draft of this post made that claim based on `llama-bench` prompt-processing numbers, which is misleading for interactive use.

### Per-task generation speeds (120B Super)

| Task | Ollama (t/s) | llama.cpp (t/s) |
|------|---:|---:|
| Coding (Fibonacci) | 5.43* | 14.51 |
| Reasoning (hotel puzzle) | 16.88 | 13.93 |
| Knowledge (photosynthesis) | 17.21 | 14.60 |
| Math (odd sum 1-100) | 17.21 | 14.66 |

*The Ollama coding result appears to be an outlier (possibly extended thinking output). The other three Ollama tasks average ~17 t/s vs llama.cpp's ~14.4 t/s. Different quantization, different context windows (4096 vs 2048), and different response lengths make direct comparison imperfect.

### Comparison caveat

This is not a clean sm_120 vs sm_121 comparison. The two runs used different GGUF files (Q4_K_M vs Q4_K) and different context lengths. Isolating the architecture effect would require the same GGUF on both Ollama and the native build. Both approaches produced similar interactive speeds for this model in our measurements.

## Other models on DGX Spark

| Model | Params (active) | Quant | Backend | Avg t/s |
|-------|---:|---|---|---:|
| Qwen 2.5-Coder | 1.5B | default | Ollama | 135.88 |
| Nemotron-3-Nano | 30B (3B) | Q8_0 | Ollama | 48.69 |
| Nemotron-3-Nano | 30B (3B) | Q8_0 | llama.cpp sm_121 | 40.74 |
| Nemotron-3-Super | 120B (12B) | Q4_K | llama.cpp sm_121 | 14.43 |
| Nemotron-3-Super | 120B (12B) | Q4_K_M | Ollama | 14.18 |

The 120B Super has 512 experts with 22 active per token (12B active params), which keeps it runnable on 128 GB.

## OOM on Model Load

The 66 GB model needed ~73 GB at runtime. On a 128 GB system Linux caches file data aggressively; after downloading or copying large files, page cache consumed 50+ GB, leaving too little for the model.

**Workaround:**

```bash
# Drop page cache before loading
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Stop Ollama if running (it holds GPU memory)
sudo systemctl stop ollama

# Then start llama-server
./build/bin/llama-server \
  -m /path/to/models/Nemotron-3-Super-120B-Q4_K.gguf \
  -ngl 99 -c 32768 -fa on \
  --host 0.0.0.0 --port 8090
```

Note: `-fa on` (not bare `-fa`) is required in recent llama.cpp builds.

## Ollama GGUF is not compatible with upstream llama.cpp

Loading Ollama's 81 GB Q4_K_M blob with our llama.cpp build failed:

```
check_tensor_dims: tensor 'blk.1.ffn_down_exps.weight' has wrong shape;
expected 2688, 4096, 512, got 2688, 1024, 512, 1
```

Ollama packs MoE expert weights differently (groups of 4, dim=1024) than upstream llama.cpp expects (concatenated, dim=4096). These are structurally incompatible despite both being valid GGUF v3.

This also means a true apples-to-apples sm_120 vs sm_121 speed comparison isn't possible without either:
- Building llama.cpp twice (sm_120 and sm_121) using the same ggml-org GGUF, or
- Waiting for Ollama to adopt the upstream tensor layout

Download the [ggml-org GGUF](https://huggingface.co/ggml-org/nemotron-3-super-120b-GGUF) (66 GB) separately.

## Build instructions

Full recipe with all error solutions: [BUILD_RECIPE.md](BUILD_RECIPE.md)

Quick version:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="121" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
# ~3 minutes on 20-core Grace ARM
```

## Possible next speedups

- NVFP4 native dispatch — requires CUDA 13.2+ (we have 13.0). Would cut model size roughly in half and help bandwidth-bound generation.
- Ollama shipping sm_121 — native arch without manual builds.
- vLLM on Spark — continuous batching and PagedAttention for multi-user serving. Community work at [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker).

## Files in this package

- [BUILD_RECIPE.md](BUILD_RECIPE.md) — Full build instructions with error solutions
- [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) — Detailed benchmark data for all models
- [run_server.sh](run_server.sh) — Ready-to-use server launch script
- [nemotron_super.modelfile](nemotron_super.modelfile) — Ollama Modelfile for custom configuration

## Related threads

- [llama.cpp Nemotron benchmark (DGX Spark)](https://github.com/ggml-org/llama.cpp/blob/master/benches/nemotron/nemotron-dgx-spark.md)
- [vLLM sm_121 support #36821](https://github.com/vllm-project/vllm/issues/36821)
- [spark-vllm-docker PR #93](https://github.com/eugr/spark-vllm-docker/pull/93)

---

*Tested on DGX Spark (GB10, Ubuntu 24.04, CUDA 13.0, Driver 580.126.09), llama.cpp commit 463b6a963, March 2026.*
