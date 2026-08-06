# DGX-SPARK

DGX Spark research and tests — containers, benchmarks, and investigation notes for running large models on the NVIDIA DGX Spark (GB10, SM 12.1, 128 GB unified memory).

Entries address compatibility issues with CUDA 13.x, aarch64, and SM121 that aren't covered by upstream containers or documentation. Each folder is a self-contained topic; dates and environment details live inside each folder's README.

## Contents

### 1. [nvfp4/](nvfp4/) — NVFP4 on DGX Spark (friction → working)

Single home for the NVFP4 story on GB10: March 2026 memory-floor campaign (19 GB model, 50–120 GB → ~32 GB @ ~50 tok/s), then the **software + driver/OS catch-up** that made stock upstream viable. Combines the old `nvfp4-guide/`, `nvfp4-landscape/`, and `nvfp4-memory/` trees (stubs remain at those paths).

- Current narrative + what still holds: [`nvfp4/README.md`](nvfp4/README.md)
- Living launch YAMLs: [`recipes/`](recipes/) (not re-copied here)
- Silicon deep dive: [`fp4_SASS/`](fp4_SASS/)
- Dated lab notes under [`nvfp4/archive/`](nvfp4/archive/) (March 2026 stamps)

### 2. [turboquant/](turboquant/) — TurboQuant 3-bit KV Cache Compression

Patches vLLM with TurboQuant KV cache compression ([PR #38479](https://github.com/vllm-project/vllm/pull/38479)). Builds on the NVFP4 stack — same model family, compressed KV cache.

- 240K token context at 64 GB memory, zero memory creep
- Faster than fp8 at long context (7.4 vs 5.0 tok/s at 64K)
- Needle-in-haystack recall tested across 1K–240K tokens
- Based on [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) with prebuilt SM121 FlashInfer

### 3. [mamba-dev/](mamba-dev/) — mamba-ssm for aarch64

Working mamba-ssm + causal-conv1d build for DGX Spark. `pip install mamba-ssm` from PyPI is broken on aarch64 (x86_64 binaries + PyTorch ABI mismatch). This container builds from source.

- Loads NemotronH hybrid models (Mamba-2 + Attention) via transformers
- Tested with Nemotron-Nano-12B-v2-VL-BF16 (13.2B params on GPU)

### 4. [nemo3-super-gguf/](nemo3-super-gguf/) — Nemotron-3-Super 120B via sm_121 llama.cpp

Native sm_121 llama.cpp build for Nemotron-3-Super 120B MoE. GGUF path documented because the NVFP4 checkpoint was blocked on vLLM — NemotronH LatentMoE uses relu² with separate projections and no fused `act_and_mul` MoE backend supports it.

- ~17 tok/s at Q4_K (66 GB weights, ~71 GB runtime)
- Ollama GGUF not compatible with upstream llama.cpp (different MoE tensor layout)
- NVFP4 attempt documented: three layers of failure (config, kernel, pip wheels)

### 5. [dual-spark/](dual-spark/) — Pairing two GX10 over a single CX-7 cable (200 Gb/s RDMA)

Field-tested runbook for pairing two NVIDIA DGX Sparks (ASUS Ascent GX10) over a single CX-7 direct-link cable. Covers the two failure classes that burn the most forum hours: power-path throttling (measure with `gpu_stress.py` — AC sag can clamp P8 under load; not all UPS bad) and dual-node connection issues (UFW, NCCL TCP-fallback, GID-index, half-bandwidth twin misconfig). Username models A (same user), B (two users + SSH map), and B→A conversion are documented.

- Symptom → fix table at the top of [`dual-spark/DUAL_SPARK_SETUP.md`](dual-spark/DUAL_SPARK_SETUP.md)
- Visual cable / twin / PCI map: [`dual-spark/TOPOLOGY.md`](dual-spark/TOPOLOGY.md) (one cable ≈ 200 Gb/s total)
- Verified: ~195–197 Gb/s RDMA aggregate (full single-cable budget)
- Patches eugr's launcher with NCCL multi-rail + RoCEv2 GID overrides

### 6. [atlas/](atlas/) — Atlas Inference Engine: head-to-head vs vLLM

Single-host benchmark of [Atlas](https://github.com/Avarok-Cybersecurity/atlas), a pure Rust+CUDA inference engine for GB10. Reproduced their published `~88 tok/s` Nemotron-3-Nano-30B-A3B-NVFP4 number (88.34 max, 84.36 median, c=1). Same model, same prompts: Atlas single-stream beats vLLM by +63%, vLLM wins concurrency at c=4 (continuous batching). Atlas cold-start with cached weights: 20s vs vLLM 171s (8.5×). Includes reproducer (`bench.py`), raw JSON results, and the `serve --help` flag dump.

### 7. [fp4_SASS/](fp4_SASS/) — FP4 on GB10: silicon vs marketing

Evidence-first probe of what FP4 tensor-core paths actually work on `sm_121a`. NVIDIA's own `ptxas` refuses `tcgen05.mma` for `sm_121a` — the "1 PFLOP FP4" pipe is a silicon gap on GB10, not a software switch. Consumer warp-level FP4 (`mma.sync.aligned.kind::mxf4.block_scale...`) compiles cleanly and emits the same SASS opcode (`OMMA.SF.16864.F32.E2M1.E2M1.E8`) as the RTX 5060 Ti (`sm_120a`) — Spark and the 5060 Ti share the consumer-Blackwell FP4 silicon at the instruction level.

- `findings.md` — full evidence trail with PTX/SASS citations
- `kernel-patterns.md` — patterns for integrating custom low-bit formats with consumer-Blackwell FP4, distilled from llama.cpp MMQ
- `reference/fp4_mma_reference.cu` — working MXFP4/NVFP4 inline-PTX MMA, compiles cleanly on sm_120a + sm_121a
- CuTeDSL 4.5.1+ now supports `sm_121a` as a first-class JIT target (earlier 4.4.x "missing kernel images" reports no longer apply)

### 8. [comfyui_spark_notes/](comfyui_spark_notes/) — ComfyUI wheel-shadowing gotchas on Spark

Two real pip-shadowing risks observed while running ComfyUI on Spark with the [Triplany/comfyui-dgx-spark](https://github.com/Triplany/comfyui-dgx-spark) kit applied: PyPI's `onnxruntime` silently overwrites Jay0515's `sm_121` GPU wheel (same `top_level` import path, different distribution names, no pip conflict detection), and `pip install sageattention` overwrites the local rebuild against current torch/CUDA.

- Shadowing mechanism traced to shared `top_level.txt` declaration across distinct distributions
- `get_available_providers()` is the reliable detector — startup-log GPU-discovery warnings are misleading
- Both findings verified against a live ComfyUI install on 2026-05-22
- Cross-references `fp4_SASS/` for the sm_120 / sm_121 SASS-equivalence basis behind the `sm_120` rebuild target

### 9. [recipes/](recipes/) — Verified serving launch YAMLs

Spark Arena + local measured recipes (NVFP4 and related). Living “run this” counterpart to [`nvfp4/`](nvfp4/).

## Hardware

| | March 2026 lab notes | June 2026 baseline (nvfp4 pass) |
|---|---|---|
| System | NVIDIA DGX Spark (GB10) | same |
| GPU | SM 12.1, 128 GB unified | same |
| CPU | 20-core ARM Grace (aarch64) | same |
| Driver / CUDA | 580.142 · 13.0/13.2 toolkit | **580.159.03 · 13.0.2** (DGX OS 7.5.0) |

Stamp driver · CUDA · image tag · model ID on any new measurement.

## Status

This is work on a single hardware configuration mostly. Results may not generalize to other setups. The TurboQuant container patches an unmerged vLLM PR — the API may change. Sharing what worked in case it helps others with similar hardware.

Related: [llm-latent-bridge-negative-results](https://github.com/Sggin1/llm-latent-bridge-negative-results) — negative results on bridging LLM hidden states at shallow layers, measured on this hardware.

## Acknowledgments

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — Community vLLM container with prebuilt SM121 wheels
- [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479) — TurboQuant attention backend by vibhavagarwal5
- [TurboQuant](https://arxiv.org/abs/2504.19874) — Zandieh et al., Google Research, ICLR 2026
- [turboquant-torch](https://pypi.org/project/turboquant-torch/) — Community PyTorch reimplementation
- [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
- The DGX Spark community on NVIDIA Developer Forums
- [Avarok-Cybersecurity/atlas](https://github.com/Avarok-Cybersecurity/atlas) — Pure Rust+CUDA inference engine, AGPL-3.0
- Atlas Discord community — collaborative benchmarking + maintainer responsiveness
- [eugr's spark-vllm-docker networking deep-wiki](https://deepwiki.com/eugr/spark-vllm-docker/7-dgx-spark-networking) — basis for the dual-spark §2 NCCL config

---

*Topics span March–August 2026 — DGX Spark GB10 / SM121. See each folder for stamped baselines.*
