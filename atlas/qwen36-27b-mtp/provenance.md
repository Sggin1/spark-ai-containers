# Run provenance — qwen3.6-27B-FP8 ± MTP vs vLLM

Captured: 2026-05-07T20:29:48Z
Run dir: `/home/gx10/projects/atlas/runs/qwen36-27b-mtp-2026-05-07`

## Hardware

| | |
|---|---|
| System | NVIDIA DGX Spark (ASUS Ascent GX10) |
| GPU | NVIDIA GB10 (GB10 Blackwell, SM 12.1) |
| CPU | Cortex-X925 |
| RAM | 121Gi unified |
| Kernel | 6.17.0-1014-nvidia |
| NVIDIA driver | 580.142 |
| CUDA runtime | 13.0 |

Firmware-current (post the 2026-05-07 BIOS/CX-7 flash); GPU was idle (P8) at start of run.

## Software / images

| Component | Tag | Image ID | Created |
|---|---|---|---|
| Atlas | `avarok/atlas-gb10:latest` | `sha256:f116b3bea58dc134ffdd600ac4422f8483f136aaa8c1675b2ef3fb43c4c14d98` | 2026-05-06T15:46:13.753726759-04:00 |
| vLLM | `vllm/vllm-openai:cu130-nightly` | `sha256:9c55c58b17f45b27ed5f716e6f8f79e793f0def4bb0a4427cd6a8a90583f28f1` | 2026-03-31T05:22:34.151991538Z |

## Model

| | |
|---|---|
| HF repo | `Qwen/Qwen3.6-27B-FP8` |
| Local cache | `/mnt/ai/cache/hub/hub/models--Qwen--Qwen3.6-27B-FP8` |
| Snapshot SHA | `e89b16ebf1988b3d6befa7de50abc2d76f26eb09` |
| On-disk | 29 GB |

## Atlas serve flags (Runs 1 + 2)

Verbatim from `run-mtp.sh`. Run 2 differs only by the addition of `--speculative`.

```
serve Qwen/Qwen3.6-27B-FP8 \
    --port 8888 --bind 0.0.0.0 \
    --max-seq-len 65536 \
    --kv-cache-dtype fp8 \
    --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.90 \
    --scheduling-policy slai \
    --tool-call-parser qwen3_coder \
    --enable-prefix-caching \
    [--speculative]              # Run 2 only
    --disable-thinking
```

Per maintainer's Discord message (2026-05-07): _"It's Qwen/Qwen3.6-27B-FP8 the same parameters should work that you have for 35-A3B, with/without speculative!"_ — these are the 35-A3B README example flags.

## vLLM serve flags (Run 3 — control)

```
--model Qwen/Qwen3.6-27B-FP8 \
--max-model-len 65536 \
--gpu-memory-utilization 0.90 \
--port 8000 --trust-remote-code \
--served-model-name Qwen/Qwen3.6-27B-FP8
```

Both engines: same model, same max context (65,536), same GPU mem util (0.90).

## Bench harness (`bench.py`)

Reused verbatim from the morning's full-bench-2026-05-07/phase2 run. Copied into this dir for self-contained reproduction.

| Param | Value |
|---|---|
| Runs per prompt | 3 |
| Concurrency levels | 1, 4 |
| Temperature | 0.0 (deterministic) |
| API | OpenAI-compatible `/v1/chat/completions` |

Prompts (verbatim from `bench.py:12-15`):

| Name | max_tokens | Prompt |
|---|---|---|
| short | 96 | "Reply with a single short sentence about cats." |
| medium | 256 | "Explain in ~150 words how transformers compute attention." |
| long | 768 | "Write a detailed 500-word summary of the impact of LLMs on software engineering, with examples." |

## Cold-start times

| Run | Seconds |
|---|---|
| Atlas no-MTP | 247 |
| Atlas MTP | 233 |
| vLLM | (in progress) |

Cold start = wall seconds from `docker run` to `/v1/models` returning 200.

## Reproduction

Single command (after the model is in HF cache and both images are pulled):

```bash
cd qwen36-27b-mtp/        # this dir, post-clone from GitHub
bash run-mtp.sh           # ~75-90 min on GX10
cat summary.md
```

The script is idempotent across containers (uses fixed names `atlas-bench` / `vllm-bench`, force-removes between runs) and writes raw `bench.json` per engine plus the aggregated `summary.md`.
