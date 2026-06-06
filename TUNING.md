# Tuning levers — DGX Spark (GB10, sm_121)

Brief, chart-first reference for what actually moves serving throughput on this box, so new models can be
tuned the same way and dropped in as a row. Method + raw data: [optimize project](../optimize/) ·
recipes: [`recipes/`](recipes/). Board metric throughout = **llama-benchy `tg128 @ d16384`, conc=2, single GB10, TP=1**.

## Levers, ranked (Qwen3.6-35B-A3B, NVFP4)

| Lever | Effect on conc=2 | Verdict |
|-------|------------------|---------|
| **Quant: FP8 → NVFP4** | 61 → 95 (**+55%**) | ✅ dominant — smaller weights win (bandwidth-bound) |
| **DFlash spec-decode, `num_speculative_tokens=6`** | 95 → **~120 (+35%)** | ✅✅ the win — beats board #10 NVFP4 (111.9) |
| `num_speculative_tokens` 15 (default) | 95 → 95 | ❌ wasted verify-compute; "washes out" |
| MoE backend `marlin` vs `flashinfer_*` | marlin best | ✅ flashinfer/trtllm hardware-gated on sm_121; b12x loads but slower |
| `--performance-mode throughput` | no mean gain, +variance | ❌ not a lever here |
| `--optimization-level 3` | no mean gain, +variance | ❌ not a lever here |
| `max_num_seqs` 8 / 16 / 32 | 109 / 136 / 129 (short-probe) | ~ minor; 32 fine |

**Two big levers: the quant (FP8→NVFP4) and the spec-decode token count (k=6).** Everything else was noise
on this model. The k=6 finding corrected an earlier wrong call that "DFlash washes out at conc=2" — that was
purely a k=15 artifact.

## The `num_speculative_tokens` (k) curve — the key knob

_(short-probe ranking: tg128 @ d2048, conc=2; absolute > board-d16384 but ranks hold)_

| k | conc=2 (short-probe d2048) | conc=1 | notes |
|---|--------------------------:|-------:|-------|
| 4  | 99.9 **±56** | 96.5 | too few — collapses, unstable |
| **6**  | **145.0 ±10** | 104.6 | stable peak |
| 7  | 123.0 ±16 | 92.8 | dip |
| **8**  | **160.4 ±25** | 113.7 | highest mean, noisy (board #6 INT4 used k=8) |
| 10 | 115.6 ±13 | 86.8 | |
| 12 | 127.7 ±3  | 111.9 | |

Sharply peaked, bumpy: **k6 (stable)** and **k8 (high but noisy)** are the two candidates → both validated; tied (k6 121.8, k8 123.3 isolated rank cell).

**Submission-grade (full official matrix, k8 @ ctx 131072):** rank cell `tg128 @ d16384 c2` = **112.5 ± 6**,
edging board #10 NVFP4 (111.9). _Isolated single-cell runs read ~121–123 (fresh-cache best case); the full
28-cell matrix — the actual submission methodology — settles at ~112.5 due to thermal/cache accumulation. Use
the full-matrix number for any board claim._ Strongest cell: d16384 **c5 = 154**. Full grid: `optimize/results/submit_matrix_k8.md`.
Rule: every model needs its own k-sweep (acceptance-rate-dependent); k≈6–8 is the A3B/DFlash sweet spot.

## Why k matters (rule of thumb for new models)
At draft acceptance rate `a`, each step verifies `k` draft tokens but only ~`a·k` are kept. Large `k` with
modest `a` (~25–40% here) wastes verify-compute and *adds* latency at concurrency. **Start `k` near
`1/(1-a)` ≈ 6–8, sweep ±3.** Board leaders independently landed on k=6 (NVFP4) / k=8 (INT4).

## Method (reusable)
`optimize/bench/autotune.sh` — restart per config + **short probe** (tg128@d2048, conc 1+2, ~90s) to rank
fast, then full board-bench only the winner. Cost is restart-bound (~3–4 min/config), so short probes keep
sweeps cheap. ~25 min for a 6-point k-curve.

## Per-model results (add a row per model as tuned)

| Model | Quant | Best config | conc=2 | vs board | Notes |
|-------|:-----:|-------------|:------:|----------|-------|
| Qwen3.6-35B-A3B | NVFP4 | marlin + DFlash **k=8** | **112.5** (full matrix) | edges #10 (111.9); #6 model is 404 | [recipe](recipes/measured-qwen3.6-35b-a3b-nvfp4-dflash.yaml) |
| Qwen3.6-35B-A3B | NVFP4 | marlin (plain, batched) | 94.8 | board #18-ish | [recipe](recipes/measured-qwen3.6-35b-a3b-nvfp4-marlin.yaml) |
| _next model…_ | | | | | |
