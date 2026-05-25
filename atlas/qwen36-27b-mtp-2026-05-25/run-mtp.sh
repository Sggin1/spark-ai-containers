#!/usr/bin/env bash
# Qwen3.6-27B-FP8 throughput, Atlas with/without MTP, vs vLLM.
# 2026-05-25 rerun against latest avarok/atlas-gb10:latest (post-#63 + #71).
#
# Methodology: identical to 2026-05-07 run except:
#   - bench.py extended to validate PR #71 invariant Σ token_ids == usage.completion_tokens
#   - vLLM image pinned to locally cached tag (cu130-nightly tag may have rotated)
#   - Atlas image digest captured at start for provenance
#
# Usage: bash run-mtp.sh
# Output: results/{atlas-no-mtp,atlas-mtp,vllm}/bench.json + summary.md

set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_PY="$DIR/bench.py"
MODEL=Qwen/Qwen3.6-27B-FP8
HF_CACHE_HOST=/mnt/ai/cache/hub
ATLAS_IMAGE=avarok/atlas-gb10:latest
VLLM_IMAGE=vllm/vllm-openai:v0.20.0-aarch64-cu130-ubuntu2404

mkdir -p "$DIR"/{results/atlas-no-mtp,results/atlas-mtp,results/vllm,logs}

# Capture image provenance
docker inspect "$ATLAS_IMAGE" --format '{{.Id}} {{.Created}}' > "$DIR/atlas-image.id" 2>/dev/null || true
docker inspect "$VLLM_IMAGE" --format '{{.Id}} {{.Created}}' > "$DIR/vllm-image.id" 2>/dev/null || true

stop_all() { docker rm -f atlas-bench vllm-bench 2>/dev/null || true; }

wait_ready() {
  local cap=${1:-360}
  local started=$(date +%s)
  while [ $(($(date +%s) - started)) -lt "$cap" ]; do
    curl -fsS --max-time 2 http://127.0.0.1:8888/v1/models >/dev/null 2>&1 && {
      echo $(($(date +%s) - started)); return 0
    }
    if ! docker ps --format '{{.Names}}' | grep -qE '^(atlas-bench|vllm-bench)$'; then
      echo "DIED" >&2; return 1
    fi
    sleep 5
  done
  echo "TIMEOUT" >&2; return 1
}

# ─── Run 1: Atlas without MTP ───────────────────────────────────────────────
echo "================================================================="
echo "Run 1: Atlas (Qwen3.6-27B-FP8, no MTP) — baseline decode"
echo "================================================================="
stop_all
TS_START=$(date +%s)

docker run -d --name atlas-bench \
  -p 127.0.0.1:8888:8888 --gpus all --ipc=host \
  -v "$HF_CACHE_HOST":/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface \
  "$ATLAS_IMAGE" \
  serve "$MODEL" \
    --port 8888 --bind 0.0.0.0 \
    --max-seq-len 65536 \
    --kv-cache-dtype fp8 \
    --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.90 \
    --scheduling-policy slai \
    --tool-call-parser qwen3_coder \
    --enable-prefix-caching \
    --disable-thinking \
  > "$DIR/logs/atlas-no-mtp.cid" 2>&1

cold=$(wait_ready 360)
echo "$cold" > "$DIR/results/atlas-no-mtp/cold-start.seconds"
echo "[atlas-no-mtp] cold start: ${cold}s"

python3 "$BENCH_PY" \
  --url http://127.0.0.1:8888 \
  --model "$MODEL" \
  --runs 3 --concurrency 1 4 \
  --out "$DIR/results/atlas-no-mtp/bench.json" \
  2>&1 | tee "$DIR/logs/atlas-no-mtp-bench.log" | grep -E "^=== |tok/s|invariant"

stop_all
sleep 5

# ─── Run 2: Atlas WITH MTP (--speculative) ─────────────────────────────────
echo
echo "================================================================="
echo "Run 2: Atlas (Qwen3.6-27B-FP8, --speculative) — MTP showcase"
echo "================================================================="

docker run -d --name atlas-bench \
  -p 127.0.0.1:8888:8888 --gpus all --ipc=host \
  -v "$HF_CACHE_HOST":/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface \
  "$ATLAS_IMAGE" \
  serve "$MODEL" \
    --port 8888 --bind 0.0.0.0 \
    --max-seq-len 65536 \
    --kv-cache-dtype fp8 \
    --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.90 \
    --scheduling-policy slai \
    --tool-call-parser qwen3_coder \
    --enable-prefix-caching \
    --speculative \
    --disable-thinking \
  > "$DIR/logs/atlas-mtp.cid" 2>&1

cold=$(wait_ready 360)
echo "$cold" > "$DIR/results/atlas-mtp/cold-start.seconds"
echo "[atlas-mtp] cold start: ${cold}s"

python3 "$BENCH_PY" \
  --url http://127.0.0.1:8888 \
  --model "$MODEL" \
  --runs 3 --concurrency 1 4 \
  --out "$DIR/results/atlas-mtp/bench.json" \
  2>&1 | tee "$DIR/logs/atlas-mtp-bench.log" | grep -E "^=== |tok/s|invariant"

stop_all
sleep 5

# ─── Run 3: vLLM control ────────────────────────────────────────────────────
echo
echo "================================================================="
echo "Run 3: vLLM (Qwen3.6-27B-FP8) — control"
echo "================================================================="

docker run -d --name vllm-bench \
  -p 127.0.0.1:8888:8000 --gpus all --ipc=host \
  -v "$HF_CACHE_HOST":/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface \
  "$VLLM_IMAGE" \
  --model "$MODEL" \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --port 8000 --trust-remote-code \
  --served-model-name "$MODEL" \
  > "$DIR/logs/vllm.cid" 2>&1

cold=$(wait_ready 540)
echo "$cold" > "$DIR/results/vllm/cold-start.seconds"
echo "[vllm] cold start: ${cold}s"

python3 "$BENCH_PY" \
  --url http://127.0.0.1:8888 \
  --model "$MODEL" \
  --runs 3 --concurrency 1 4 \
  --out "$DIR/results/vllm/bench.json" \
  2>&1 | tee "$DIR/logs/vllm-bench.log" | grep -E "^=== |tok/s|invariant"

stop_all

# ─── Summary ────────────────────────────────────────────────────────────────
echo
echo "================================================================="
echo "ALL DONE — total wall: $(($(date +%s) - TS_START))s"
echo "================================================================="

bash "$DIR/summarize.sh"
