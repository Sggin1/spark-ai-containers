# Benchmark Results: Nemotron-3-Super 120B on DGX Spark (GB10)

**Date:** 2026-03-14
**Hardware:** DGX Spark — GB10 Grace Blackwell, 128 GB unified memory, 20-core ARM Grace
**OS:** Ubuntu 24.04, kernel 6.17.0-1008-nvidia, CUDA 13.0, Driver 580.126.09

## Head-to-Head: Ollama vs llama.cpp Native Build

| | Ollama | llama.cpp (sm_121) |
|---|---|---|
| **Quantization** | Q4_K_M | Q4_K |
| **File size** | ~86 GB | ~66 GB |
| **Runtime memory** | ~86 GB | ~73 GB |
| **CUDA arch** | sm_120 (cuda_v12 lib) | sm_121 (native build) |
| **Context window** | 4096 | 2048 |
| **llama.cpp commit** | Ollama bundled | 463b6a963 |
| **Avg generation speed** | **14.18 t/s** | **14.43 t/s** |

### Assessment

Speeds nearly identical (14.18 vs 14.43 t/s, ~2% difference within measurement noise). The earlier claim of a "40% speedup" from sm_121 was incorrect.

Generation for a 120B MoE model on this hardware is memory-bandwidth bound, not compute bound. Whether CUDA kernels target sm_120 or sm_121 makes little difference when the GPU is memory-bound. A clean same-GGUF, same-context comparison was not run.

Memory delta: ggml-org Q4_K GGUF is 20 GB smaller than Ollama's Q4_K_M (66 GB vs 86 GB). Quantization difference (Q4_K vs Q4_K_M), not an sm_121 effect.

### Same-GGUF comparison not possible

We tried loading Ollama's 81 GB Q4_K_M blob directly with our sm_121 llama.cpp build. It failed:

```
check_tensor_dims: tensor 'blk.1.ffn_down_exps.weight' has wrong shape;
expected 2688, 4096, 512, got 2688, 1024, 512, 1
```

Ollama's GGUF packs MoE expert weights into groups of 4 per shard (dim=1024); upstream llama.cpp expects them concatenated (dim=4096). Structurally incompatible files despite both being valid GGUF. Implications:

- Ollama's blob cannot be used with upstream llama.cpp (or vice versa)
- Same-file sm_120 vs sm_121 comparison would require building llama.cpp twice (once with each arch) using the ggml-org GGUF
- The Ollama vs llama.cpp comparison here conflates quantization (Q4_K_M vs Q4_K), tensor layout, context length, and CUDA arch — not a controlled experiment

### Other Caveats

- Context window differed: 4096 (Ollama) vs 2048 (llama.cpp). This affects prompt processing time but has minimal impact on generation speed.
- Different quantization levels make this not a pure sm_120 vs sm_121 comparison.
- Ollama's coding result (5.43 t/s) appears to be an outlier, possibly due to the model generating a very long response (715 tokens with extended thinking). The other three Ollama tasks averaged ~17 t/s.

## Per-Task Breakdown: 120B Super

### Ollama (Q4_K_M, sm_120, ctx=4096)

| Task | Tokens | Time (s) | Speed (t/s) |
|------|-------:|---------:|------------:|
| Coding (Fibonacci) | 715 | 131.73 | 5.43 |
| Reasoning (hotel puzzle) | 262 | 15.52 | 16.88 |
| Knowledge (photosynthesis) | 1,648 | 95.75 | 17.21 |
| Math (odd sum 1-100) | 944 | 54.86 | 17.21 |
| **Total / Average** | **3,569** | **297.86** | **14.18** |

### llama.cpp Native (Q4_K, sm_121, ctx=2048)

| Task | Tokens | Time (s) | Speed (t/s) |
|------|-------:|---------:|------------:|
| Coding (Fibonacci) | 512 | 35.30 | 14.51 |
| Reasoning (hotel puzzle) | 184 | 13.21 | 13.93 |
| Knowledge (photosynthesis) | 512 | 35.08 | 14.60 |
| Math (odd sum 1-100) | 246 | 16.78 | 14.66 |
| **Total / Average** | **1,454** | **100.37** | **14.43** |

Note: llama.cpp results are more consistent across tasks (13.9-14.7 t/s range vs Ollama's 5.4-17.2 t/s range). The Ollama coding outlier drags its average down.

## Other Models Tested

### Nemotron-3-Nano 30B (A3B) — Q8_0

| Backend | Avg Speed |
|---------|----------:|
| Ollama (sm_120) | 48.69 t/s |
| llama.cpp native (sm_121) | 40.74 t/s |

Nano is faster due to 3B active parameters (vs 12B for Super). Ollama measured faster here; possible factors: Q8_0 at 30B total params is small enough that the bundled Ollama runtime has no disadvantage, and Ollama may apply additional optimizations.

#### Nano Per-Task (llama.cpp native, sm_121)

| Task | Tokens | Time (s) | Speed (t/s) |
|------|-------:|---------:|------------:|
| Coding | 512 | 12.49 | 40.98 |
| Reasoning | 364 | 8.93 | 40.76 |
| Knowledge | 512 | 12.18 | 42.04 |
| Math | 319 | 8.14 | 39.19 |

### Qwen 2.5-Coder 1.5B (Ollama)

| Task | Tokens | Time (s) | Speed (t/s) |
|------|-------:|---------:|------------:|
| Coding | 294 | 2.10 | 140.29 |
| Reasoning | 69 | 0.60 | 115.35 |
| Knowledge | 209 | 1.47 | 142.62 |
| Math | 250 | 1.72 | 145.26 |
| **Average** | | | **135.88** |

Reference point for small-model throughput on this hardware.

## Summary Table: All Models

| Model | Params (active) | Quant | Backend | Avg t/s |
|-------|----------------:|-------|---------|--------:|
| Qwen 2.5-Coder | 1.5B | default | Ollama | 135.88 |
| Nemotron-3-Nano | 30B (3B) | Q8_0 | Ollama | 48.69 |
| Nemotron-3-Nano | 30B (3B) | Q8_0 | llama.cpp sm_121 | 40.74 |
| Nemotron-3-Super | 120B (12B) | Q4_K | llama.cpp sm_121 | 14.43 |
| Nemotron-3-Super | 120B (12B) | Q4_K_M | Ollama | 14.18 |

## Notes

- NVFP4 dispatch requires CUDA 13.2+ (we have 13.0).
- All benchmarks ran on the same DGX Spark system with no other GPU workloads.
- Ollama was stopped before llama.cpp benchmarks to free unified memory.
- The model architecture is MoE: 120B total parameters, 12B active, 512 experts with 22 active per token.
