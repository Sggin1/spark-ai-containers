# DGX-SPARK

DGX Spark research and tests — containers, benchmarks, and investigation notes for running large models on the NVIDIA DGX Spark (GB10, SM 12.1, 128 GB unified memory).

Entries address compatibility issues with CUDA 13.x, aarch64, and SM121 that aren't covered by upstream containers or documentation. Each folder is a self-contained topic; dates and environment details live inside each folder's README.

## Contents

### 1. [nvfp4-guide/](nvfp4-guide/) — NVFP4 on DGX Spark: 120 GB → 32 GB

Root cause analysis and fix for vLLM memory bloat with Nemotron-3-Nano-30B NVFP4 on SM121. Four independent issues identified (broken CUTLASS FP4, KV cache over-allocation, torch.compile overhead, FlashInfer JIT spike), each with a specific fix.

- 19 GB model running at 32 GB total, 50 tok/s
- Marlin backend bypasses broken CUTLASS FP4 kernels on SM121
- Full benchmark data across configurations

### 2. [turboquant/](turboquant/) — TurboQuant 3-bit KV Cache Compression

Patches vLLM with TurboQuant KV cache compression ([PR #38479](https://github.com/vllm-project/vllm/pull/38479)). Builds on the NVFP4 guide — same model, same Marlin backend, adds compressed KV cache.

- 240K token context at 64 GB memory, zero memory creep
- Faster than fp8 at long context (7.4 vs 5.0 tok/s at 64K)
- Needle-in-haystack recall tested across 1K–240K tokens
- Based on [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) with prebuilt SM121 FlashInfer

### 3. [mamba-dev/](mamba-dev/) — mamba-ssm for aarch64

Working mamba-ssm + causal-conv1d build for DGX Spark. `pip install mamba-ssm` from PyPI is broken on aarch64 (x86_64 binaries + PyTorch ABI mismatch). This container builds from source.

- Loads NemotronH hybrid models (Mamba-2 + Attention) via transformers
- Tested with Nemotron-Nano-12B-v2-VL-BF16 (13.2B params on GPU)

### 4. [nvfp4-landscape/](nvfp4-landscape/) — NVFP4 on DGX Spark: Landscape Snapshot (March 2026)

Literature synthesis of community findings, tracked PRs, and the SM121 FP4 kernel situation as of March 26, 2026. Aggregates forum threads, vLLM docs, and NVIDIA cookbook links into a single reference.

- Why CUTLASS FP4 falls back on SM121 (no `tcgen05`)
- Tracked PRs: CUTLASS #3038, vLLM #35947 / #38126, Flash-Attention #2222
- Memory component breakdown: 50–120 GB default → ~25 GB with minimal flags

### 5. [nvfp4-memory/](nvfp4-memory/) — NVFP4 Memory Footprint (supporting data)

Flag-by-flag memory snapshot for Nemotron-3-Nano-30B-A3B-NVFP4 on vLLM 0.18.1rc1. Deeper cut supporting `nvfp4-guide/` — raw research notes, FlashInfer JIT analysis, earlier forum draft.

- Config table: 117 GB default → 32 GB with Marlin + `--enforce-eager` + util 0.2
- `research_status.md`, `flashinfer_jit.md`, `benchmarks.md` as supporting files

### 6. [nemo3-super-gguf/](nemo3-super-gguf/) — Nemotron-3-Super 120B via sm_121 llama.cpp

Native sm_121 llama.cpp build for Nemotron-3-Super 120B MoE. GGUF path documented because the NVFP4 checkpoint was blocked on vLLM — NemotronH LatentMoE uses relu² with separate projections and no fused `act_and_mul` MoE backend supports it.

- ~17 tok/s at Q4_K (66 GB weights, ~71 GB runtime)
- Ollama GGUF not compatible with upstream llama.cpp (different MoE tensor layout)
- NVFP4 attempt documented: three layers of failure (config, kernel, pip wheels)

## Hardware

| | |
|---|---|
| System | NVIDIA DGX Spark |
| GPU | GB10 Blackwell, SM 12.1, 128 GB unified memory |
| CPU | 20-core ARM Grace (aarch64) |
| CUDA | 13.2, Driver 580.142 |

## Status

This is hobbyist work on a single hardware configuration. Results may not generalize to other setups. The TurboQuant container patches an unmerged vLLM PR — the API may change. Sharing what worked in case it helps others with similar hardware.

## Acknowledgments

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — Community vLLM container with prebuilt SM121 wheels
- [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479) — TurboQuant attention backend by vibhavagarwal5
- [TurboQuant](https://arxiv.org/abs/2504.19874) — Zandieh et al., Google Research, ICLR 2026
- [turboquant-torch](https://pypi.org/project/turboquant-torch/) — Community PyTorch reimplementation
- [NVIDIA DGX Spark Playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
- The DGX Spark community on NVIDIA Developer Forums

---

*Tested March 2026 — DGX Spark GB10, SM121, CUDA 13.2, Driver 580.142*
