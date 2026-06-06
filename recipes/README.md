# Verified DGX Spark serving recipes (captured from Spark Arena)

**Captured:** 2026-06-05 from the [Spark Arena leaderboard](https://spark-arena.com/leaderboard)
(community DGX Spark benchmark, `recipeType: spark-vllm-docker`). These are *live, benchmarked*
launch recipes — the practical counterpart to the dated lab notes in [`../nvfp4-guide`](../nvfp4-guide/),
[`../nvfp4-landscape`](../nvfp4-landscape/), [`../nvfp4-memory`](../nvfp4-memory/).

**Ranking basis:** test `tg128 @ d16384`, concurrency 2, single-node decode, sorted by tokens/sec.
The ranks below are positions in that specific view (99 entries total). Other test types /
concurrencies re-rank the table — treat these as *a* slice, not an absolute ordering.

> Live source of truth: each model row on the leaderboard is clickable → recipe modal (full YAML).
> Bulk data: `https://spark-arena.com/static/snapshot` (all entries). Per-benchmark metrics:
> `https://spark-arena.com/api/benchmarks/<uuid>/raw`. Canonical recipe files for *some* models:
> [`eugr/spark-vllm-docker/recipes/`](https://github.com/eugr/spark-vllm-docker/tree/main/recipes)
> (NVFP4 Qwen3.6 and the AutoRound/Omni variants are Arena submissions, not in that repo).

## The four locked-in recipes

| Rank | Model | Quant | tok/s | Container | Backend story |
|:----:|-------|:-----:|:-----:|-----------|---------------|
| [6](rank06-qwen3.6-35b-a3b-int4-autoround.yaml) | Intel/Qwen3.6-35B-A3B-int4-AutoRound | INT4 (gptq) | 131.36 | `vllm-node` | INT4 + DFlash spec-decode (k=8) |
| [10](rank10-qwen3.6-35b-a3b-nvfp4.yaml) | RedHatAI/Qwen3.6-35B-A3B-NVFP4 | NVFP4 (compressed-tensors) | 111.92 | `vllm-node-tf5` | auto backend + DFlash (k=6), CUDA graphs ON |
| [13](rank13-nemotron-3-nano-30b-a3b-nvfp4.yaml) | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | NVFP4 (modelopt) | 104.74 | **official** `vllm/vllm-openai:v0.20.0-aarch64-cu130-ubuntu2404` | **FlashInfer MoE FP4 = 1** |
| [22](rank22-nemotron-3-nano-omni-30b-a3b-nvfp4.yaml) | nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 | NVFP4 (multimodal) | 84.67 | **official** `vllm/vllm-openai:v0.20.0-aarch64-cu130-ubuntu2404` | **Marlin** (`--moe-backend marlin`, FP4 MoE = 0) |

## Measured locally (optimize project, single GB10) — not Arena scrapes

These two are *our own* benchmarks on this box (vLLM v0.22.1rc1.dev124, TP=1, same `tg128 @ d16384`
test), added from the [optimize](../../optimize/) deep-dive. They use `nvidia/*` NVFP4 (modelopt),
distinct from the Arena entries' `RedHatAI/*` (compressed-tensors).

| Recipe | Quant | conc=1 solo | conc=2 | Note |
|--------|:-----:|:-----------:|:------:|------|
| [nvfp4-marlin](measured-qwen3.6-35b-a3b-nvfp4-marlin.yaml) | NVFP4 (modelopt) | 66.8 | 94.8 | marlin = the sm_121 NVFP4 ceiling in this image |
| [nvfp4-dflash **k=8**](measured-qwen3.6-35b-a3b-nvfp4-dflash.yaml) | NVFP4 + DFlash | 85.8 | **112.5** (full matrix) | **edges board #10 NVFP4 (111.9)**; #6 model now 404 |

**Findings:** (1) On sm_121 in this image, FlashInfer/TRT-LLM NVFP4-MoE kernels are *hardware-gated*
(`kernel does not support current device cuda`) and `flashinfer_b12x` loads but is slower — **marlin
wins** at the bandwidth-bound conc=1–2 regime. (2) **`num_speculative_tokens` is the decisive DFlash knob.**
At k=15, DFlash "washes out" at conc=2 (94.9 ≈ plain NVFP4); at **k≈6–8** it lifts the rank cell to **112.5
(full official matrix)**, edging board #10's NVFP4 — on lighter NVFP4 with a *simpler* config (no throughput
flags, no RedHatAI checkpoint). Found via a fast restart+short-probe autotune sweep. (3) `--performance-mode
throughput` / `-O3` were *not* levers here. (Isolated single-cell runs read ~121–123; the full 28-cell matrix
— the submission methodology — settles ~112.5. Use the matrix number.) See [`../TUNING.md`](../TUNING.md).

## Why these matter for the nvfp4-* docs

Rank #13 is the **exact model the docs were built around**, so it's a direct before/after:

| | nvfp4-* docs (March 2026) | Rank #13 recipe (June 2026) |
|---|---|---|
| Container | eugr fork `vllm-node:0.18.1rc1` | **official** `vllm/vllm-openai:v0.20.0-aarch64-cu130` |
| FlashInfer MoE FP4 | `VLLM_USE_FLASHINFER_MOE_FP4=0` (disable) | **`=1` (enable)** |
| GEMM backend | force `VLLM_NVFP4_GEMM_BACKEND=marlin` | not set (auto) |
| CUDA graphs | `--enforce-eager` (off) | on (default) |
| gpu-mem-util | 0.2 (memory floor) | 0.8 (throughput) |
| max-num-seqs / ctx | 1 / 8192 | 20 / 262144 |

**Key takeaways**
1. **Stock upstream vLLM, no fork** — confirmed by two recipes running the official aarch64/cu130 image.
2. **The "always force Marlin / disable FlashInfer" rule is now per-model.** Text Nemotron-Nano (#13)
   enables FlashInfer MoE FP4; the multimodal Omni (#22) still pins Marlin. Both on the official image.
3. **NVFP4 format split:** modelopt (nvidia/*) vs compressed-tensors (RedHatAI/*) — the launch flag
   differs (`--quantization fp4` vs `compressed-tensors`). Docs only cover modelopt.
4. **Speculative decoding (z-lab DFlash) is the throughput lever** behind the top Qwen3.6 entries —
   absent from the docs entirely.
5. Defaults trend: `util 0.8`, full long context, prefix caching + chunked prefill, `kv-cache-dtype fp8`.

## Note on Holo (rank #5, the top NVFP4 entry — appeared "out of nowhere")

`Hcompany/Holo-3.1-35B-A3B-NVFP4` is **H Company's** (French startup) Holo 3.1 — a **vision-language
"computer use" agent** family (browser/desktop/mobile automation + native function calling), sizes
0.8B → 35B-A3B. The 35B-A3B ships FP8, Q4 GGUF, and **NVFP4 (ModelOpt, W4A16)**; FP8 and NVFP4 match
on OSWorld, ~2 pts under BF16. H Company reports **NVFP4 W4A16 = 1.41× FP8 / 1.74× BF16 throughput on
DGX Spark**. On the leaderboard it's a **2-node TP=2 (Ray over CX-7)** recipe — see the dual-Spark note.
Not an LLM-chat model; it's an agentic VLM, which is why it looked unfamiliar.
Refs: [HF model](https://huggingface.co/Hcompany/Holo-3.1-35B-A3B) ·
[Holo3.1 blog](https://huggingface.co/blog/Hcompany/holo31) · [hcompany.ai/holo3.1](https://hcompany.ai/holo3.1)
