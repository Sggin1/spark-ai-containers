# Atlas Inference Engine — Benchmark Results

Single-host benchmark of the [Atlas inference engine](https://github.com/Avarok-Cybersecurity/atlas) on an ASUS Ascent GX10 (NVIDIA GB10 / DGX Spark class), with vLLM A/B on the same model + prompt set for context.

Date: 2026-05-07.

---

## Hardware

- ASUS Ascent GX10 (NVIDIA GB10, SM121)
- BIOS `GX10DGX.0104.2026.0326.1657`
- Driver `580.142`, CUDA `13.0`, VBIOS `9A.0B.1E.00.00`
- EC firmware `0x02000005`, UEFI/SoC `0x03000006` (current per LVFS as of 2026-04-08 / 2026-04-02)
- Direct wall outlet (no UPS — see [`../dual-spark/`](../dual-spark/) §0 for why this matters)

## GPU health gate (pre-bench)

BF16 GEMM `16384²`, 50-iter burst + 20s sustained.

- **Burst:** 96.67 TFLOP/s
- **Sustained:** 85.85 TFLOP/s
- Verdict: HEALTHY (≥ 80 TFLOP/s floor for GB10)

## Engine versions

- Atlas: `avarok/atlas-gb10@sha256:9943e54a92b41e2f20673964056084f619d942dc4479b658672cd68acd72fbe1`
- vLLM: `vllm/vllm-openai:cu130-nightly`

## Model

`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` — 30B total / 3B active, Mamba-2 + attention + MoE, NVFP4 weights.

## Bench parameters

- 3 prompt classes (short ~50–100 tok out · medium ~256–320 · long ~768–845)
- 3 iterations per class
- 2 concurrency levels (c=1, c=4)
- `temperature=0.0`, `max-seq-len=32768`, `gpu-memory-utilization=0.85`
- Reproducer: [`bench.py`](./bench.py) — pure-stdlib OpenAI-compat client

## Throughput

### Single-stream (c=1, 9 runs)

| Engine | Median tok/s | Max tok/s |
|---|---|---|
| **Atlas** | **84.36** | **88.34** |
| vLLM | 51.84 | 52.07 |

Atlas single-stream is **+62.7%** over vLLM on the same hardware, model, prompts.

### Concurrent (c=4, 9 runs)

| Engine | Per-request median | Aggregate (× 4) |
|---|---|---|
| Atlas | 22.14 tok/s | **88.56 tok/s** |
| vLLM  | 33.89 tok/s | **135.56 tok/s** |

vLLM aggregate at c=4 is **+53%** over Atlas — vLLM's continuous batching wins under parallel load.

### Atlas saturation curve (c=1,2,4,8, 6 runs each)

| Concurrency | Per-req median | Aggregate |
|---|---|---|
| 1 | 76.21 | 76.21 |
| 2 | 39.88 | 79.76 |
| 4 | 19.73 | 78.90 |
| 8 | 12.83 | **102.60** |

Atlas plateaus around 80 tok/s aggregate at c=2–4, then climbs to 102 tok/s at c=8 — long-prompt requests in the c=8 batch saw 22–31 tok/s individually vs 19 for short, suggesting prefill amortization improves with depth.

## Cold start (image cached, weights cached)

| Engine | Time |
|---|---|
| Atlas | **20 s** |
| vLLM | 171 s |

Atlas is **8.5×** faster cold-start with weights already on disk. (First-ever cold start with weight pull would add the model-download time on top.)

## Forum-claim reproducibility

The forum thread quotes "Nemotron-3 Nano 30B (FP8): ~88 tok/s" — measured **88.34 tok/s max / 84.36 median** on NVFP4 with default flags. **Reproduces.**

## Findings (doc-gaps + behavior worth noting)

1. **`--quantization` flag has been renamed to `--mtp-quantization`.** README example still uses the old name. Forum thread already noted this.
2. **README `docker run` example uses `--network host`,** which works because `spark serve` defaults to `--bind 127.0.0.1` and the container's loopback equals the host's loopback in that mode. If a user swaps to `-p 127.0.0.1:8888:8888` (defense-in-depth), the port-forward never reaches the listener — must add `--bind 0.0.0.0`. Worth a README clarification.
3. **`--ngram-speculative` started cleanly (`/v1/models` responded) but generation hung** on Nemotron-H. The `--help` text already warns that `--speculative` is "currently slower than regular decode for hybrid SSM models" — looks like that caveat extends to ngram on this architecture. Not regression; worth a help-text edit if intentional.
4. **Per-response perf telemetry in the OpenAI-compat `usage` block** — `time_to_first_token_ms` and `response_token/s`. Nice touch; vLLM doesn't have these on this image.
5. **Prefix caching is dramatic.** TTFT dropped from ~270–330 ms (cold) to ~20 ms (warm prefix) on identical repeated prompts — `--enable-prefix-caching` paying off.
6. **Multi-node flags exist but undocumented:** `--rank`, `--world-size`, `--tp-size`, `--ep-size`, `--master-addr`, `--master-port`, `--bind`, `--require-auth`. Not exercised in this run.

## Caveats

- Single-host (`gx10` only); no TP=2 attempt yet.
- vLLM's first request had a ~24 s warmup (JIT / autotune); subsequent requests stable at ~52 tok/s.
- vLLM does not surface TTFT in the `usage` field on this image.
- vLLM run required `--trust-remote-code` for Nemotron-H's custom `configuration_nemotron_h.py`.
- KV-cache dtype: Atlas `fp8` (default); vLLM auto/bf16. Not strictly normalized.
- `temperature=0.0`, but Atlas's reasoning content (separate `reasoning_content` field) was not suppressed.

## Reproduce

```bash
# Atlas
docker run -d --name atlas-bench \
  -p 127.0.0.1:8888:8888 --gpus all --ipc=host \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  avarok/atlas-gb10:latest \
  serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
    --port 8888 --bind 0.0.0.0 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching

# vLLM
docker run -d --name vllm-bench \
  -p 127.0.0.1:8001:8000 --gpus all --ipc=host \
  -v /path/to/hf-cache:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:cu130-nightly \
  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --port 8000 --trust-remote-code

# Bench
python3 bench.py --url http://127.0.0.1:8888 --runs 3 --concurrency 1 4 \
  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
  --out results/atlas.json
```

## Files

| File | Contents |
|---|---|
| [`README.md`](./README.md) | This page |
| [`bench.py`](./bench.py) | Reproducer (Python stdlib only) |
| [`serve-help.txt`](./serve-help.txt) | `spark serve --help` output (286 lines — fuller flag list than the Atlas README) |
| [`results/atlas-bench.json`](./results/atlas-bench.json) | Raw Atlas c=1 + c=4 results |
| [`results/vllm-bench.json`](./results/vllm-bench.json) | Raw vLLM c=1 + c=4 results |
| [`results/atlas-saturation.json`](./results/atlas-saturation.json) | Atlas c=1,2,4,8 sweep |

## Open follow-ups (could run on request)

- Atlas TP=2 across two CX-7-paired Sparks (no public Atlas multi-node numbers exist anywhere)
- `--self-speculative` on Nemotron-H (different from `--ngram-speculative`)
- `--dflash` block-diffusion speculative (needs Z-Lab DFlash drafter)
- Concurrency extension (c=16, c=32)
- Multi-model A/B (Qwen3.5-35B headline `130 tok/s` claim, MiniMax M2.7, Gemma-4-26B-A4B)
