# Nemotron-3-Super 120B on DGX Spark — llama.cpp sm_121 Build Package

Running NVIDIA's Nemotron-3-Super 120B (MoE, 12B active) on the DGX Spark GB10 with a native sm_121 llama.cpp build.

## Quick Start

```bash
# 1. Build llama.cpp for sm_121 (Blackwell)
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="121" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# 2. Download the model (~66 GB)
pip install huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('ggml-org/nemotron-3-super-120b-GGUF', 'Nemotron-3-Super-120B-Q4_K.gguf', local_dir='/path/to/models/')
"

# 3. Free memory and serve
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
sudo systemctl stop ollama  # if running
./run_server.sh
```

## What's Here

| File | Description |
|------|-------------|
| [README.md](README.md) | This file |
| [BUILD_RECIPE.md](BUILD_RECIPE.md) | Step-by-step build instructions with environment details, error solutions, and OOM workaround |
| [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) | Full benchmark data: 5 model/backend combinations, per-task breakdowns, honest analysis |
| [forum_post.md](forum_post.md) | Community write-up summarizing findings |
| [forum_post_followup_nvfp4_attempt.md](forum_post_followup_nvfp4_attempt.md) | Follow-up documenting why the NVFP4 path was blocked |
| [run_server.sh](run_server.sh) | Launch script for llama-server (port 8090, flash attention, OpenAI-compatible API) |
| [nemotron_super.modelfile](nemotron_super.modelfile) | Ollama Modelfile with tuned parameters and reasoning system prompt |

## Observations

- Speed: llama.cpp sm_121 native build ran at 14.43 t/s, Ollama (sm_120) at 14.18 t/s
- Memory: ggml-org Q4_K GGUF (66 GB) used ~20 GB less than Ollama's Q4_K_M (86 GB)
- Compatibility: Ollama's GGUF not compatible with upstream llama.cpp (different MoE tensor layout)
- OOM pitfall: dropped Linux page cache before loading to avoid OOM kills

## Hardware

| | |
|---|---|
| System | NVIDIA DGX Spark |
| GPU | GB10 Blackwell, 128 GB unified VRAM, compute capability 12.1 |
| CPU | 20-core ARM Grace |
| OS | Ubuntu 24.04, CUDA 13.0, Driver 580.126.09 |

## Model Architecture

Nemotron-3-Super is a Mixture-of-Experts (MoE) model:
- 120B total parameters, 12B active per forward pass
- 512 experts, 22 active per token
- Mamba-Transformer hybrid with LatentMoE

Only the active experts need to be computed per token, though all weights must be in memory.

## License

These configuration files and documentation are provided as-is for the DGX Spark community. The Nemotron-3-Super model is subject to NVIDIA's license terms.
