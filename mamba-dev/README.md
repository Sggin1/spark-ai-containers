# mamba-ssm for DGX Spark (aarch64) — Early Results

Working mamba-ssm + causal-conv1d container for DGX Spark. Enables loading NemotronH hybrid models (Mamba-2 + Attention) via transformers on aarch64.

## The Problem

`pip install mamba-ssm` from PyPI is broken on DGX Spark for two independent reasons:

1. **Wrong architecture.** The PyPI wheel ships `selective_scan_cuda.cpython-312-x86_64-linux-gnu.so`. DGX Spark is aarch64. The `.so` can't load.

2. **ABI mismatch.** Even with `MAMBA_FORCE_BUILD=TRUE`, the PyPI release (v2.3.1) compiles CUDA extensions referencing `c10::cuda::CUDAStream::query()`, a symbol removed in PyTorch 2.10+. Same issue affects `causal-conv1d`.

The fix is building both packages from the GitHub main branch, which has the PyTorch 2.10+ ABI fixes.

## Quick Start

### Build

```bash
git clone https://github.com/Sggin1/spark-ai-containers.git
cd spark-ai-containers/mamba-dev
docker build -t spark-mamba-dev:latest .
```

Build takes ~15 minutes (compiling CUDA extensions for SM121).

### Run

```bash
docker run --runtime=nvidia --gpus all -it \
    -v /path/to/hf-cache:/root/.cache/huggingface \
    spark-mamba-dev:latest
```

### Test

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda",
)

inputs = tokenizer("What is the capital of France? Answer in one word:", return_tensors="pt").to("cuda")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
# Paris
```

## Manual Install (without Docker)

Inside `nvcr.io/nvidia/pytorch:25.11-py3` or similar:

```bash
export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="12.1"

pip install --no-cache-dir --no-build-isolation \
    "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git"

pip install --no-cache-dir --no-build-isolation \
    "mamba-ssm @ git+https://github.com/state-spaces/mamba.git"

# For Nemotron-Nano VL models:
pip install accelerate timm open_clip_torch
```

Key flags:
- `MAMBA_FORCE_BUILD=TRUE` — forces native compilation instead of using broken PyPI wheel
- `--no-build-isolation` — uses the container's existing PyTorch for ABI compatibility
- `TORCH_CUDA_ARCH_LIST="12.1"` — targets SM121 (GB10)

## What This Enables

- Loading any NemotronH model (Mamba-2 + Attention hybrid) via `transformers`
- Direct access to model weights, KV cache, hidden states
- Research on hybrid SSM/Attention architectures on consumer Blackwell hardware
- KV cache manipulation (capture, compress, inject) for memory research

## Tested Models

| Model | Params | from_pretrained | generate() | Notes |
|-------|:------:|:---------------:|:----------:|-------|
| NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 | 13.2B | Yes | Yes | BF16, ~7 tok/s steady state |

**What's been verified:**
- CUDA extensions compile and import (`selective_scan_cuda`, `causal_conv1d_cuda`, mamba-ssm 2.3.1)
- Model loads to GPU via `AutoModelForCausalLM.from_pretrained()` (13.2B params, 7 shards)
- `model.generate()` produces correct outputs (see results below)
- Weight extraction works (used for KV projection analysis in [turboquant](../turboquant/))

**What has NOT been verified:**
- Text-only NemotronH variants
- Nemotron-3-Nano-30B (served via vLLM in a separate container, not tested here)

## Results

### Build Observations

| Metric | Value |
|--------|:-----:|
| Build time | ~12 min |
| Memory during build | ~12 GB init → ~30 GB spike (CUDA compilation) → ~7-8 GB settled |
| mamba-ssm version | 2.3.1 |
| causal-conv1d | compiled from main branch |

### Model Loading

| Metric | Value |
|--------|:-----:|
| Download time | ~3 min (25 GB, 7 shards) |
| Load to GPU | ~6 min (374s) |
| Parameters | 13.2B |
| Architecture | NemotronH (55 Mamba-2 + 7 Attention layers) |

### Generation Quality

| Test | Tokens | tok/s | Result |
|------|:------:|:-----:|--------|
| factual ("capital of France") | 3 | 0.2* | **PASS** — "Paris" |
| math (17 * 24) | 6 | 8.1 | **PASS** — "408" |
| code (is_prime) | 100 | 7.1 | **PASS** — correct description, hit token limit |

*First request warmup — same pattern observed in vLLM serving.

## Caveats

- **Build from source required.** The upstream PyPI packages don't work on aarch64 + PyTorch 2.10+. This may change in future releases.
- **Vision model dependencies.** The VL variant needs `timm` and `open_clip_torch`. Text-only NemotronH models may need fewer dependencies.
- **Build time.** Compiling CUDA extensions takes ~15 minutes. Pre-built images avoid this.
- **Container CUDA version.** The base container uses CUDA 13.0, while the DGX Spark host runs CUDA 13.2. This hasn't caused issues but is worth noting.

## Known Limitations & Next Steps

- **Slow throughput needs investigation.** Steady-state generation is ~7-8 tok/s via direct transformers, compared to ~50 tok/s for the same model family via vLLM (NVFP4 quantized, Marlin backend). Possible causes: BF16 vs NVFP4 (2x model size in memory), no KV cache optimization, no batched prefill, no CUDA graphs, transformers' generate() overhead. Needs profiling to identify the bottleneck.
- **First-request warmup.** First generate() call runs at ~0.2 tok/s (15.6s for 3 tokens). Subsequent calls reach 7-8 tok/s. This matches vLLM behavior and is likely CUDA kernel warmup, not a mamba-ssm issue.
- **Unpinned source builds.** Both mamba-ssm and causal-conv1d build from `main` branch with no pinned commit. A future breaking change upstream could break the build. Consider pinning once a known-good commit is identified.
- **Single model tested.** Only Nemotron-Nano-12B-v2-VL-BF16 has been loaded. Other NemotronH models (text-only 12B, 30B) need testing.
- **Build memory spike.** CUDA extension compilation spikes to ~30 GB during build. Not an issue on the 128 GB Spark, but would matter on smaller systems.
- **No comparison to vLLM path.** For serving, vLLM handles mamba-ssm internally with optimized kernels. This container is for direct transformers access and research — a fair throughput comparison would need the same model format (BF16) on both paths.
- **Deprecation warning.** `torch_dtype` parameter is deprecated in transformers 5.x — use `dtype` instead.

## References

- [mamba-ssm](https://github.com/state-spaces/mamba) — Mamba: Linear-Time Sequence Modeling
- [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) — Fast 1D causal convolution
- [NVIDIA Nemotron-Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16)

---

*Build tested March 31, 2026 — DGX Spark GB10, nvcr.io/nvidia/pytorch:25.11-py3, CUDA 13.0 (container), PyTorch 2.10, mamba-ssm 2.3.1*
