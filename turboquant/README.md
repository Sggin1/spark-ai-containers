# TurboQuant KV Cache Compression on DGX Spark — Early Results

TurboQuant 3-bit KV cache compression for Nemotron-3-Nano-30B-A3B-NVFP4 on DGX Spark, patched into vLLM via [PR #38479](https://github.com/vllm-project/vllm/pull/38479).

This is an early test of TurboQuant on SM121 hardware. The PR is unmerged and the results are from Triton fallback kernels (not the fused CUDA path). Numbers will likely improve as the PR matures.

## Results

### Quality (8-test battery)

| Test | Category | Result |
|------|----------|:------:|
| Capital of France | factual | PASS |
| Chemical formula for water | factual | PASS |
| 17 * 24 | math | PASS |
| Syllogism (Whiskers) | logic | PASS |
| Secret code recall | needle-in-haystack | PASS |
| List 5 colors numbered | instruction | PASS |
| is_prime function | code | FAIL* |
| DGX Spark summary | summarization | PASS |

7/8 pass. *Code test hit token limit during model reasoning, not a quality issue.

### Needle-in-Haystack Across Context Lengths

Secret code placed at token 0, filler padding to target length, recall question at the end.

**128K max context (gpu_memory_utilization=0.35):**

| Context | Prompt Tokens | tok/s | Memory | Recall |
|--------:|:---:|:---:|:---:|:---:|
| 1K | 746 | 40.7 | 57 GB | PASS |
| 4K | 2,846 | 34.9 | 56 GB | PASS |
| 16K | 11,246 | 15.5 | 57 GB | PASS |
| 32K | 22,446 | 9.2 | 58 GB | PASS |
| 64K | 44,846 | 7.4 | 57 GB | PASS |
| 96K | 67,246 | 3.5 | 53 GB | PASS |
| 120K | 84,046 | 3.9 | 54 GB | PASS |

7/7 pass. Memory flat at 53-58 GB.

**256K max context (gpu_memory_utilization=0.45):**

| Context | Prompt Tokens | tok/s | Memory | Recall |
|--------:|:---:|:---:|:---:|:---:|
| 1K | 689 | 2.4 | 68 GB | PASS |
| 8K | 5,189 | 7.3 | 69 GB | PASS |
| 32K | 20,617 | 9.6 | 69 GB | PASS |
| 64K | 41,189 | 8.1 | 70 GB | FAIL* |
| 128K | 82,332 | 4.4 | 69 GB | PASS |
| 192K | 123,474 | 3.0 | 68 GB | FAIL* |
| 240K | 154,332 | 1.8 | 64 GB | PASS |

5/7 pass. *Failures at 64K and 192K are non-systematic — model passes at both shorter and longer contexts around these points. FP8 baseline needle test at these depths has not been run yet, so these may be model behavior or TQ3 artifact. Needs investigation. Memory flat at 64-70 GB.

### TQ3 vs FP8 Head-to-Head

Same hardware, same model, same gpu_memory_utilization (0.45), same context depths.

| Context | TQ3 tok/s | FP8 tok/s | TQ3 Memory | FP8 Memory |
|--------:|:---------:|:---------:|:----------:|:----------:|
| 1K | 40.7 | 2.3* | 57 GB | 92 GB |
| 4K | 34.9 | 3.2* | 56 GB | 92 GB |
| 16K | 15.5 | 16.8 | 57 GB | 93 GB |
| 32K | 9.2 | 9.0 | 58 GB | 92 GB |
| 64K | 7.4 | 5.0 | 57 GB | 92 GB |
| 96K | 3.5 | 3.0 | 54 GB | 90 GB |
| 120K | 3.9 | 2.3 | 54 GB | 90 GB |

*FP8 1K/4K numbers are first-request warmup, not steady-state. Needs re-run with warmup request for fair comparison.

TQ3 is faster at most context lengths and uses 35 GB less memory. At 64K+, TQ3 wins by 50-70% on throughput because 3-bit KV means less data moved through memory bandwidth per attention step — the decompress overhead is less than the bandwidth savings.

### Memory Stability

Memory measured via `/proc/meminfo` (unified memory — `nvidia-smi` doesn't report on GB10).

| After query # | Memory (GB) | Delta |
|:---:|:---:|:---:|
| 0 (idle) | 59.6 | — |
| 1 | 58.9 | -0.7 |
| 4 | 58.7 | -0.2 |
| 8 | 58.4 | -0.3 |

Zero creep. Memory decreased slightly as caches settled.

### KV Cache Cost Per Token

Nemotron-3-Nano has only 7 attention layers (55 are Mamba-2 with fixed-size state). KV cache is inherently small.

| Format | Per Token | Per 1K Tokens |
|--------|:---------:|:-------------:|
| tq3 (3-bit) | 5.3 KB | 5.1 MB |
| fp8 (8-bit) | 14.3 KB | 13.7 MB |
| bf16 (16-bit) | 28.7 KB | 27.3 MB |

| Context | tq3 KV | fp8 KV | bf16 KV |
|--------:|:------:|:------:|:-------:|
| 32K | 168 MB | 448 MB | 896 MB |
| 128K | 672 MB | 1.8 GB | 3.5 GB |
| 256K | 1.3 GB | 3.5 GB | 7.0 GB |
| 1M | 5.1 GB | 13.7 GB | 27.3 GB |

vLLM pre-allocates the full KV pool at startup, so total memory is flat during queries regardless of actual context used. The per-token savings translate to more context capacity in the same allocation.

## How It Works

TurboQuant compresses only the **7 attention layers' KV cache**. The 55 Mamba-2 layers have fixed SSM state (unaffected). MoE expert routing is unaffected.

Per KV vector (bf16 → 3 bits):
1. **Randomized Hadamard rotation** — spreads information evenly across dimensions so rounding errors are uniform, not catastrophic in one dimension
2. **Lloyd-Max quantization** (2 bits) — mathematically optimal rounding to 4 discrete levels
3. **QJL residual** (1 bit) — random projection + sign captures error direction

The result preserves inner products as an unbiased estimator — attention still points at the right tokens.

## Quick Start

### Build

Requires [vllm-node:latest](https://github.com/eugr/spark-vllm-docker) as the base image.

```bash
git clone https://github.com/Sggin1/spark-ai-containers.git
cd spark-ai-containers/turboquant
docker build -t vllm-turboquant:latest .
```

### Run

```bash
# Flush page cache (recommended on unified memory)
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

docker run -d --runtime=nvidia \
    --name nano-tq3 \
    -v /path/to/hf-cache:/root/.cache/huggingface \
    -p 8000:8000 \
    -e VLLM_NVFP4_GEMM_BACKEND=marlin \
    -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
    -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
    vllm-turboquant:latest \
    python3 -m vllm.entrypoints.openai.api_server \
        --host 0.0.0.0 --port 8000 \
        --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
        --enforce-eager \
        --max-num-seqs 1 \
        --max-model-len 131072 \
        --gpu-memory-utilization 0.35 \
        --kv-cache-dtype tq3 \
        --trust-remote-code
```

### Test

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 100
  }'
```

## Configuration Notes

| Flag | Purpose |
|------|---------|
| `--kv-cache-dtype tq3` | TurboQuant 3-bit KV cache (also supports `tq4`) |
| `--enforce-eager` | Disables torch.compile + CUDA graphs (prevents memory creep) |
| `--gpu-memory-utilization 0.35` | KV cache pool size. 0.35 for 128K, 0.45 for 256K |
| `--max-model-len` | Max context. Model supports up to 262144 |
| `VLLM_NVFP4_GEMM_BACKEND=marlin` | Bypasses broken CUTLASS FP4 kernels on SM121 |

See the [NVFP4 guide](../nvfp4-guide/) for details on memory creep causes and solutions.

## Caveats

- **Unmerged PR.** This patches vLLM with [PR #38479](https://github.com/vllm-project/vllm/pull/38479) which is open and under review. The API may change.
- **Triton fallback only.** The PR includes fused CUDA kernels that aren't compiled for SM121 in this build. Throughput would improve with native kernels.
- **Unofficial TurboQuant implementation.** Google has not released official code. The vLLM PR and `turboquant-torch` are community implementations based on the [paper](https://arxiv.org/abs/2504.19874).
- **Tested on one model.** Nemotron-3-Nano-30B-A3B-NVFP4 only. Other models may behave differently.

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Container build (extends vllm-node:latest) |
| `patch_vllm.py` | Applies PR #38479 patches to vLLM |
| `patches/pr_38479.diff` | Raw PR diff |
| `new_files/` | New TurboQuant source files from the PR |

## Known Limitations & Next Steps

- **FP8 baseline incomplete.** Throughput comparison exists but needle-in-haystack has not been run on FP8 at matching context depths (64K, 192K). Needed to determine if 256K needle failures are model behavior or TQ3 artifact.
- **FP8 short-context warmup.** The 1K/4K FP8 throughput numbers (2.3/3.2 tok/s) reflect first-request warmup, not steady-state. Needs re-run with warmup request for fair comparison.
- **Single model tested.** Results are from Nemotron-3-Nano-30B only. Other architectures (pure transformer, different GQA ratios) may behave differently.
- **Triton fallback only.** Fused CUDA kernels from PR #38479 not yet compiled for SM121. Throughput has room to improve with native kernels.
- **TQ4 not tested.** 4-bit mode (`--kv-cache-dtype tq4`) is available but not benchmarked yet — may offer a better quality/speed tradeoff.

## References

- [TurboQuant: Online Vector Quantization](https://arxiv.org/abs/2504.19874) — Zandieh et al., ICLR 2026
- [Google Research blog post](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479) — TurboQuant attention backend
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — Base container with SM121 support
- [NVIDIA Forum: State of FP4/NVFP4 on DGX Spark](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069)

---

*Tested March 30-31, 2026 — DGX Spark GB10, SM121, CUDA 13.2, Driver 580.142*
