# Follow-up: GGUF vs NVFP4 — We Tried Running Both on DGX Spark

Following up on my [earlier post](forum_post.md) about running Nemotron-3-Super 120B with a native sm_121 llama.cpp build. Several people asked about the official NVFP4 checkpoint — can you run it on Spark? How does quality compare?

**Short answer:** The GGUF is the path we got working on Spark. The NVFP4 checkpoint did not load — blocked at the kernel level, not just config.

---

## GGUF vs NVFP4 on DGX Spark

The GGUF path requires building llama.cpp from source (see [original post](forum_post.md) for the full recipe). Steps once the binary is built:

1. Build llama.cpp targeting sm_121 (~3 min compile, one-time)
2. Download one file — the [ggml-org Q4_K GGUF](https://huggingface.co/ggml-org/nemotron-3-super-120b-GGUF) (66 GB)
3. Run `llama-server -m model.gguf`

The NVFP4 path requires a compatible vLLM build with CUDA 13 support, the right Docker image, correct quantization config handling, and NemotronH-specific MoE kernels. As of March 2026, none of these pieces fully exist for sm_121. We hit three separate layers of failure — details below.

## What Happened When We Tried NVFP4

### Layer 1: Config validation (fixable)

The community `avarok/vllm-dgx-spark` Docker image (vLLM 0.14, built Jan 2026) doesn't recognize the NVFP4 model's `MIXED_PRECISION` quantization config. The model was released March 11 — two months after the image was built. The model uses a mix of NVFP4 (40,961 layers) and FP8 (139 layers), which the older vLLM doesn't expect.

```
ValueError: ModelOpt currently only supports: ['FP8', 'FP8_PER_CHANNEL_PER_TOKEN',
'FP8_PB_WO', 'NVFP4'] quantizations in vLLM.
```

We patched the ModelOpt quantization module to accept `MIXED_PRECISION` and route it to the NVFP4 handler. Config validation passed.

### Layer 2: MoE kernel incompatibility (not fixable today)

```
NotImplementedError: No NvFp4 MoE backend supports the deployment configuration.
```

All 5 available NVFP4 MoE backends (FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL, FLASHINFER_CUTLASS, VLLM_CUTLASS, MARLIN) require fused `act_and_mul` MLP layers. NemotronH uses `relu2` activation with separate projections in its LatentMoE — a different architecture than standard MoE models like Mixtral or DeepSeek. We force-selected Marlin specifically and got:

```
ValueError: NvFp4 MoE backend 'MARLIN' does not support the deployment
configuration since kernel does not support no act_and_mul MLP layer.
```

This is a kernel-level limitation. No config change or patch fixes it.

### Layer 3: No pip escape hatch either

Tried installing vLLM 0.18.0 natively (the version the model README recommends). The pip wheels are compiled against CUDA 12 — Spark runs CUDA 13. Immediate import failure on `libcudart.so.12`.

### What would unblock NVFP4

- vLLM 0.18+ built for CUDA 13 + sm_121 with NemotronH-compatible MoE kernels
- CUDA 13.2+ (may bring native NVFP4 dispatch for Blackwell)
- An updated `avarok/vllm-dgx-spark` image with these fixes

## GGUF output spot-check

Caveat: not a rigorous benchmark. 8 prompts; responses checked for basic correctness (keyword presence, correct answers). No BF16 or NVFP4 baseline for quality comparison, so no statement here about Q4_K quantization loss. Only whether responses were correct and coherent.

Settings: temperature=1.0 and top_p=0.95 (NVIDIA's documented settings for the NVFP4 model; same settings applied to the GGUF for consistency).

| Task | Tokens | Speed | Result |
|------|-------:|------:|--------|
| Code Generation (Fibonacci w/ memoization) | 882 | 16.9 t/s | Correct implementation with input validation and complexity analysis |
| Lateral Thinking (Monopoly riddle) | 189 | 16.7 t/s | Correct answer (Monopoly) with explanation |
| General Knowledge (photosynthesis vs respiration) | 1,605 | 17.2 t/s | Accurate, well-organized comparison |
| Math (sum of odds 1-100) | 337 | 17.0 t/s | Correct answer (2500) with step-by-step arithmetic sequence formula |
| Instruction Following (5 Asian countries with 'I') | 224 | 16.9 t/s | Exactly 5 items, numbered as requested |
| Creative (haiku about GPU) | 340 | 17.0 t/s | Valid 5-7-5 syllable structure |
| Technical Analysis (TCP vs UDP) | 1,669 | 17.3 t/s | Accurate comparison with use-case recommendations |
| Multi-step Reasoning ("all but 9" trick) | 290 | 16.9 t/s | Correct answer (9) with reasoning about the phrasing |

**Average: 16.99 t/s across 5,536 total tokens generated.**

All responses were factually correct for the prompts tested. The model handled a reasoning trick question ("all but 9") and a lateral thinking puzzle (Monopoly). Harder benchmarks (MMLU-Pro, LiveCodeBench, etc.) not run. NVIDIA's model card shows NVFP4 scores ~0.4% below BF16 on MMLU-Pro (83.33 vs 83.73); GGUF Q4_K not measured.

## How to Reproduce

### Prerequisites

- DGX Spark with llama.cpp built for sm_121 (see [original post](forum_post.md) for build recipe — requires cmake, CUDA toolkit 13.0)
- The ggml-org GGUF: [ggml-org/nemotron-3-super-120b-GGUF](https://huggingface.co/ggml-org/nemotron-3-super-120b-GGUF) (Q4_K, 66 GB)

### Step 1: Clear the system

The 66 GB model needs ~71 GB at runtime.

```bash
# Stop anything holding GPU memory
sudo systemctl stop ollama
pkill -f llama-server

# Drop page cache
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Verify you have headroom
free -g   # check the "available" column, should show 110+ GB
```

### Step 2: Start the server

```bash
llama-server \
  -m /path/to/Nemotron-3-Super-120B-Q4_K.gguf \
  --port 8090 \
  --host 0.0.0.0 \
  -ngl 99 \
  -fa on \
  -c 8192 \
  --metrics
```

Model load takes ~6 minutes (mmap'ing 66 GB). Wait until you see `HTTP server listening` in the output.

### Step 3: Test it

```bash
curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the sum of all odd numbers from 1 to 100?"}],
    "max_tokens": 2048,
    "temperature": 1.0,
    "top_p": 0.95
  }' | python3 -m json.tool
```

Or use the benchmark script which runs all 8 prompts and captures full responses + speed metrics:

```bash
git clone https://gist.github.com/<gist-id> nemo3_super_benchmarks
cd nemo3_super_benchmarks
python3 quality_compare.py gguf
```

### Memory breakdown

| Component | Size |
|-----------|-----:|
| Model weights (mmap) | 66 GB |
| KV cache (8192 context) | ~4 GB |
| CUDA context + buffers | ~1 GB |
| **Total runtime** | **~71 GB** |

`free` may show low "free" but check the "available" column — Linux page cache is reclaimable. ~50 GB remained available for other processes in our runs.

## Speed comparison across our tests

| Date | Build | max_tokens | Prompts | Avg t/s |
|------|-------|--------:|--------:|--------:|
| March 22 | llama.cpp build 924 (01d8eaa) | 2048 | 8 | **16.99** |
| March 14 | llama.cpp 463b6a963 | 512 | 4 | 14.43 |
| March 13 | Ollama (sm_120) | default | 4 | 14.18 |

The March 22 test used a newer llama.cpp build, higher max_tokens, and more prompts than the March 14 test — multiple variables changed, so the speed difference can't be attributed to a single factor. Observed range: ~14-17 t/s on Spark depending on build and settings.

## Summary

GGUF was the path we got working for Nemotron-3-Super 120B on DGX Spark. Requires a one-time llama.cpp build, a 66 GB download, and a single command to serve. Responses were correct across the prompts tested; generation at ~17 t/s.

NVFP4 checkpoint blocked by missing MoE kernel support in vLLM for NemotronH's architecture. Kernels would need updating for the LatentMoE layer structure.

---

*Tested on DGX Spark (GB10, Ubuntu 24.04, CUDA 13.0, Driver 580.126.09), llama.cpp build 924 (01d8eaa), March 2026.*
