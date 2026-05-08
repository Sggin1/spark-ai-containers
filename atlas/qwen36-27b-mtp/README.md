# Qwen3.6-27B-FP8 — Atlas ± MTP vs vLLM (3-way head-to-head)

Maintainer-requested test (Atlas Discord, 2026-05-07).
Single-host, ASUS Ascent GX10 (NVIDIA GB10, SM121), post-firmware-update
(BIOS 0104.2026.0326.1657, EC 0x02000005, UEFI/SoC 0x03000006).

## Setup

- **Model:** `Qwen/Qwen3.6-27B-FP8` (27B dense, FP8 weights)
- **Atlas image:** `avarok/atlas-gb10:latest`
- **vLLM image:** `vllm/vllm-openai:cu130-nightly`
- **Atlas flags (per maintainer):** `--max-seq-len 65536 --kv-cache-dtype fp8 --kv-high-precision-layers auto --gpu-memory-utilization 0.90 --scheduling-policy slai --tool-call-parser qwen3_coder --enable-prefix-caching --disable-thinking`
- **MTP run additionally:** `--speculative` (built-in MTP heads, no separate drafter)
- **vLLM flags:** `--max-model-len 65536 --gpu-memory-utilization 0.90 --trust-remote-code` (KV-dtype: vLLM default = bf16 / auto)
- **Bench:** 3 prompts × 3 runs × concurrency {1, 4}, `temperature=0`

## Throughput (median tok/s, 9 runs per concurrency)

| Engine             | c=1 median | c=1 max | c=4 per-req median | c=4 aggregate |
|---|---|---|---|---|
| **Atlas (no MTP)** | 4.38 | 5.57 | 1.33 | 5.32 |
| **Atlas (MTP)**    | 4.64 | 5.12 | 1.17 | 4.68 |
| **vLLM**           | 4.0    | 4.02    | 3.86    | 15.44 |

## Cold start (post-cache, container start → /v1/models 200)

| Engine             | seconds |
|---|---|
| Atlas (no MTP)     | 247 |
| Atlas (MTP)        | 233 |
| vLLM               | 911 |

## MTP speedup vs no-MTP (Atlas)

```
c=1: +5.9% (4.38 -> 4.64 tok/s)
c=4 aggregate: +-12.0% (5.32 -> 4.68 tok/s)
```

## Atlas vs vLLM (single-stream)

```
no-MTP: Atlas +9.5% over vLLM (4.38 vs 4.00 tok/s)
MTP:    Atlas +16.0% over vLLM (4.64 vs 4.00 tok/s)
```

## Files

```
results/atlas-no-mtp/bench.json
results/atlas-mtp/bench.json
results/vllm/bench.json
results/*/cold-start.seconds
logs/*.log              (full bench-script output per run)
```

## Reproduction

```bash
bash run-mtp.sh
```
