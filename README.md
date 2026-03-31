# spark-ai-containers

Docker containers and guides for running AI models on the NVIDIA DGX Spark (GB10, SM121, 128 GB unified memory).

These address specific compatibility issues with CUDA 13.2, aarch64, and SM121 that aren't covered by upstream containers or documentation.

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
