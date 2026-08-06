# NVFP4 on DGX Spark

**Folder status:** sole top-level `nv*` topic in this repo. Combined 2026-08-06 from `nvfp4-guide/`, `nvfp4-landscape/`, and `nvfp4-memory/`; root stubs for those paths removed 2026-08-06.  
**Primary snapshot (friction era):** 2026-03-25 → 2026-03-26  
**Last narrative update:** 2026-08-06 (this README)  
**Host baseline noted June 2026:** Driver **580.159.03** · CUDA **13.0.2** · DGX OS 7.5.0  
**Earlier lab baseline (March):** Driver 580.142 · CUDA 13.0/13.2 containers · vLLM 0.18.1rc1

One topic, one place: how NVFP4 went from “barely works / burns 50–120 GB” on GB10 to **usable stock upstream**, what still matters for memory, and where the dated lab notes live.

---

## TL;DR (as of mid‑2026)

| Then (March 2026) | Now (June+ 2026) |
|---|---|
| Often need community **eugr** `vllm-node` + prebuilt sm_121 FlashInfer | **Official** `vllm/vllm-openai:…-aarch64-cu130` works for several NVFP4 recipes |
| Force **Marlin**, disable FlashInfer FP4 MoE env vars | Backend is **per-model**; text Nemotron may enable FlashInfer MoE; Omni still often pins Marlin |
| “Native sm_121 NVFP4 is broken / missing” | **CUTLASS SM120/121 NVFP4 GEMM** in vLLM (PR #40082, May 2026); arch-suffix NaN fixes earlier |
| Goal: 19 GB model in **~32 GB** at ~50 tok/s (memory floor) | Recipes also chase **throughput** (util 0.8, long ctx, DFlash, graphs on) — see [`../recipes/`](../recipes/) |
| llama.cpp NVFP4 weak / CPU-ish on Spark | **GPU-accelerated** NVFP4 path landed (see June notes) |

**Still true regardless of kernel progress:**

- Unified memory + multi-tenant vLLM defaults ⇒ **KV over-provision** and **compile/graph** bloat
- `/proc/meminfo` beats `nvidia-smi` for real pressure on GB10
- Drop caches before tight memory experiments; treat `gpu_memory_utilization` as a deliberate knob

**Do this today:** use verified launch YAMLs in [`../recipes/`](../recipes/).  
**Silicon deep dive:** [`../fp4_SASS/`](../fp4_SASS/) (consumer Blackwell FP4 vs marketing `tcgen05`).

---

## The arc: friction → working

### 1. Friction (≈ March 2026)

Nemotron-3-Nano-30B-A3B-**NVFP4** is ~**19 GB** on disk. On DGX Spark (128 GB unified), stock-ish vLLM often landed at **50–120 GB** and/or slow fallbacks.

Four independent problems stacked:

1. **Broken / missing sm_121 CUTLASS FP4 paths** — capability checks said “Blackwell-ish,” silicon lacked the datacenter `tcgen05` pipe; tactics failed or fell back  
2. **KV cache over-allocation** — `gpu_memory_utilization≈0.9` on a 128 GB pool  
3. **torch.compile + CUDA graphs** — tens of GB on unified memory  
4. **FlashInfer JIT** — NVIDIA images with sm_120 wheels → multi-GB `cicc` spikes on first use  

Workaround era: **Marlin** (FP4→BF16 dequant on instructions that *do* work) + **`--enforce-eager`** + **util 0.2** + community containers with **prebuilt sm_121** wheels → **~32 GB @ ~50 tok/s**. That story is preserved in [`guide.md`](guide.md) and the [`archive/`](archive/) lab notes.

### 2. Software catch-up (late March → May 2026)

Upstream and ecosystem moved unusually fast for a new SKU:

| Area | Progress (high level) |
|---|---|
| **vLLM** | Arch-suffix preservation (NaN / sm_12x correctness); **native SM120/121 CUTLASS NVFP4 GEMM** (#40082, 2026-05-20); releases **≥ 0.19** increasingly “just work” on Spark without a private fork |
| **Backend knobs** | Old `VLLM_NVFP4_GEMM_BACKEND` / `VLLM_USE_FLASHINFER_MOE_FP4` → **deprecated**; prefer `--linear-backend` / `--moe-backend` (and accept **per-model** choices) |
| **Containers** | Official **aarch64 + cu130** images usable; eugr still useful for some community/Arena recipes but no longer the only path |
| **llama.cpp** | NVFP4 GPU acceleration landed (removes an earlier “only GGUF Qn” mental model for some models) |
| **CUTLASS / FA** | Ongoing sm_12x wiring; consumer FP4 vs datacenter FP4 remains a **silicon** distinction — see `fp4_SASS` |

### 3. Firmware / host stack

Lab notes and host baselines also moved (not just Python wheels):

| When | Driver / CUDA (notes) |
|---|---|
| March lab | Driver **580.142**, CUDA 13.0 reported / 13.2 toolkit in places |
| June accuracy pass | Driver **580.159.03**, CUDA **13.0.2**, DGX OS **7.5.0** |

Driver + DGX OS updates matter for NVML/clock behavior, container/runtime compatibility, and “did my recipe stop working after an OTA?” debugging. Always stamp **driver · CUDA · vLLM image tag · model ID** on any new number you publish.

### 4. Working (June 2026+)

Spark Arena and local recipes show NVFP4 as a **first-class serving format** on GB10:

- Official image recipes for Nemotron NVFP4 (including **FlashInfer MoE FP4 enabled** on text Nano in rank #13-style configs)  
- Marlin still wins in some bandwidth-bound / multimodal / kernel-gated cases  
- Throughput stack adds **DFlash**, higher util, long context, prefix cache — different objective than the March “memory floor” campaign  

Compare March vs June flags side-by-side in [`../recipes/README.md`](../recipes/README.md) (“Why these matter for the nvfp4 docs”).

---

## What to read (by need)

| Need | Doc |
|---|---|
| Current launch configs | [`../recipes/`](../recipes/) |
| Memory problem taxonomy (creep / KV / spike) | [`memory.md`](memory.md) — **still useful**; flag names partly stale |
| Full March 120 GB → 32 GB narrative + tables | [`guide.md`](guide.md) — **historical how-we-got-there**, not default install |
| Literature / PR / runtime survey (Mar 26) | [`archive/landscape-2026-03-26.md`](archive/landscape-2026-03-26.md) |
| Flag matrix, JIT notes, research status | [`archive/`](archive/) index |
| FP4 silicon truth (tcgen05 vs warp MMA) | [`../fp4_SASS/`](../fp4_SASS/) |
| Long-context KV compression on top of NVFP4 stack | [`../turboquant/`](../turboquant/) |

---

## What still holds vs what to re-verify

### Still holds (re-check numbers, keep the *ideas*)

- Serving-stack overhead ≠ NVFP4 weight size  
- KV util and max length dominate steady-state footprint  
- Compile/graph and JIT can spike or creep on unified memory  
- Backend choice is load-bearing on sm_121 (Marlin vs FlashInfer vs CUTLASS paths)  
- Single-user memory floor ≠ multi-tenant throughput config  

### Treat as dated unless re-measured

- Exact GB and tok/s tables from March (vLLM **0.18.1rc1**, eugr-only story)  
- “Always force Marlin / always disable FlashInfer MoE” as a universal rule  
- “Must use eugr; official image cannot do NVFP4”  
- Any claim that native sm_121 NVFP4 GEMM does not exist  

### Follow-ups (not done in this combine)

Worth a later pass when you have cycles — **not** blocking this folder merge:

1. Re-run Nemotron Nano NVFP4 memory floor on **current** official image; refresh one table in `guide.md` or a new `measurements-YYYY-MM.md`  
2. Align `memory.md` examples to `--linear-backend` / `--moe-backend` only (drop env-var-first examples)  
3. One-page “backend decision tree” (text Nano vs Omni vs Qwen modelopt vs compressed-tensors)  
4. Link or note dual-node NVFP4 (Holo-style TP=2) vs single-box recipes  
5. Confirm llama.cpp NVFP4 GPU path on *this* host with a stamped command + tok/s  
6. Root monorepo hardware table still mentions older driver in places — refresh when convenient  

---

## Layout

```text
nvfp4/
  README.md                 ← you are here (current narrative, 2026-08-06)
  guide.md                  ← March 2026 120→32 guide (+ June inline corrections)
  memory.md                 ← memory taxonomy (creep / KV / spike)
  archive/                  ← date-stamped lab snapshots (frozen)
    README.md
    landscape-2026-03-26.md
    memory-footprint-2026-03-26.md
    benchmarks-2026-03-26.md
    flashinfer-jit-2026-03-26.md
    research-status-2026-03-25.md
```

**Removed on combine:** `nvfp4-memory/earlier_forum_post_draft.md` (near-duplicate of the guide).  
**Removed later:** root redirect stubs `nvfp4-guide/`, `nvfp4-landscape/`, `nvfp4-memory/` — use this folder only.

---

## Related

- [`../recipes/`](../recipes/) — verified Arena + local launch YAMLs  
- [`../TUNING.md`](../TUNING.md) — decode/tuning notes  
- [`../fp4_SASS/`](../fp4_SASS/) — FP4 instruction-level evidence  
- [`../turboquant/`](../turboquant/) — 3-bit KV on the NVFP4 serving path  
- [`../dual-spark/`](../dual-spark/) — two-box fabric (some Arena NVFP4 entries are multi-node)

---

*Combine + narrative refresh: 2026-08-06. Underlying March lab work: 2026-03-25/26. June accuracy banners: 2026-06-05.*
