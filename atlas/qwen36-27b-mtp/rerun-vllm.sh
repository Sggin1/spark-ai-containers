#!/usr/bin/env bash
# Standalone re-run of just the vLLM control leg, with a 1200s cold-start cap
# (the 540s cap in run-mtp.sh was insufficient for 27B-dense-FP8 vLLM startup).
# Same image, same flags, same bench harness as run-mtp.sh.

set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_PY=/home/gx10/projects/atlas/runs/full-bench-2026-05-07/phase2/bench.py
MODEL=Qwen/Qwen3.6-27B-FP8
HF_CACHE_HOST=/mnt/ai/cache/hub

mkdir -p "$DIR"/{results/vllm,logs}

stop_all() { docker rm -f vllm-bench 2>/dev/null || true; }

wait_ready() {
  local cap=${1:-1200}
  local started=$(date +%s)
  while [ $(($(date +%s) - started)) -lt "$cap" ]; do
    curl -fsS --max-time 2 http://127.0.0.1:8888/v1/models >/dev/null 2>&1 && {
      echo $(($(date +%s) - started)); return 0
    }
    if ! docker ps --format '{{.Names}}' | grep -qE '^vllm-bench$'; then
      echo "DIED" >&2; return 1
    fi
    sleep 5
  done
  echo "TIMEOUT" >&2; return 1
}

echo "================================================================="
echo "vLLM rerun (Qwen3.6-27B-FP8) — 1200s cold-start cap"
echo "================================================================="
TS_START=$(date +%s)
stop_all

docker run -d --name vllm-bench \
  -p 127.0.0.1:8888:8000 --gpus all --ipc=host \
  -v "$HF_CACHE_HOST":/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e HF_HOME=/root/.cache/huggingface \
  vllm/vllm-openai:cu130-nightly \
  --model "$MODEL" \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --port 8000 --trust-remote-code \
  --served-model-name "$MODEL" \
  > "$DIR/logs/vllm-rerun.cid" 2>&1

cold=$(wait_ready 1200)
echo "$cold" > "$DIR/results/vllm/cold-start.seconds"
echo "[vllm] cold start: ${cold}s"

if [[ "$cold" == "TIMEOUT" || "$cold" == "DIED" || -z "$cold" ]]; then
  echo "vLLM failed to start within 1200s — capturing container logs and bailing"
  docker logs vllm-bench > "$DIR/logs/vllm-rerun-container.log" 2>&1 || true
  stop_all
  exit 1
fi

# Capture container logs even on success (size of state, kernel-jit notes etc.)
docker logs vllm-bench > "$DIR/logs/vllm-rerun-container.log" 2>&1 || true

python3 "$BENCH_PY" \
  --url http://127.0.0.1:8888 \
  --model "$MODEL" \
  --runs 3 --concurrency 1 4 \
  --out "$DIR/results/vllm/bench.json" \
  2>&1 | tee "$DIR/logs/vllm-rerun-bench.log" | grep -E "^=== |tok/s"

stop_all

echo
echo "================================================================="
echo "vLLM rerun done — total wall: $(($(date +%s) - TS_START))s"
echo "================================================================="

# Regenerate the aggregated summary.md now that vllm/bench.json exists
bash "$DIR/summarize.sh"
