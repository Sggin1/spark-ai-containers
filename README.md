# spark-ai-containers

Docker containers for running AI models on the NVIDIA DGX Spark (GB10, SM121, 128 GB unified memory).

These solve specific compatibility issues with CUDA 13.2, aarch64, and SM121 that aren't addressed by upstream containers.

## Containers

### [turboquant/](turboquant/) — TurboQuant KV Cache Compression for vLLM

Patches vLLM with TurboQuant 3-bit KV cache compression ([PR #38479](https://github.com/vllm-project/vllm/pull/38479)). Tested on Nemotron-3-Nano-30B-A3B-NVFP4.

- 240K token context at 64 GB memory
- Faster than fp8 at long context (7.4 vs 5.0 tok/s at 64K)
- Zero memory creep across all tests
- Based on [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (prebuilt SM121 FlashInfer)

### [mamba-dev/](mamba-dev/) — mamba-ssm for aarch64

Working mamba-ssm + causal-conv1d build for DGX Spark. `pip install mamba-ssm` from PyPI is broken on aarch64 (ships x86_64 binaries, ABI mismatch with PyTorch 2.10+). This container builds from source with the correct flags.

- Loads NemotronH hybrid models (Mamba-2 + Attention) via transformers
- Tested with Nemotron-Nano-12B-v2-VL-BF16 (13.2B params)

## Hardware

| | |
|---|---|
| System | NVIDIA DGX Spark |
| GPU | GB10 Blackwell, SM 12.1, 128 GB unified memory |
| CPU | 20-core ARM Grace (aarch64) |
| CUDA | 13.2, Driver 580.142 |

## Acknowledgments

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — Community vLLM container with prebuilt SM121 wheels
- [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479) — TurboQuant attention backend
- [TurboQuant paper](https://arxiv.org/abs/2504.19874) (Google, ICLR 2026)
- [turboquant-torch](https://pypi.org/project/turboquant-torch/) — Community PyTorch reimplementation
- The DGX Spark community on NVIDIA Developer Forums
