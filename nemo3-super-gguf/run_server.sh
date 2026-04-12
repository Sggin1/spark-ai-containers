#!/usr/bin/env bash
# run_server.sh — Serve Nemotron-3-Super 120B via llama.cpp on DGX Spark
#
# Prerequisites:
#   - llama.cpp built with sm_121 (see BUILD_RECIPE.md)
#   - Model downloaded to MODEL_PATH below
#   - Ollama stopped (sudo systemctl stop ollama) to free memory
#
# Port 8090 is used to avoid conflict with Open WebUI on 8080.

set -euo pipefail

# Configuration — adjust these paths for your setup
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/projects/llama.cpp/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-/path/to/models/Nemotron-3-Super-120B-Q4_K.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8090}"
CTX="${CTX:-32768}"
GPU_LAYERS="${GPU_LAYERS:-99}"

# Validate paths
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    echo "Build llama.cpp first (see BUILD_RECIPE.md) or set LLAMA_SERVER env var."
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Download from https://huggingface.co/ggml-org/nemotron-3-super-120b-GGUF"
    echo "Or set MODEL_PATH env var."
    exit 1
fi

# Free page cache to avoid OOM during model load
# The 66GB GGUF needs ~73GB at runtime. Linux page cache can eat available memory.
echo "Dropping page cache to ensure enough free memory for model load..."
if sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    echo "Page cache dropped."
else
    echo "WARNING: Could not drop page cache (needs sudo). If you hit OOM, run:"
    echo "  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'"
fi

echo ""
echo "Starting Nemotron-3-Super 120B (Q4_K, 66GB)"
echo "  Server:  $LLAMA_SERVER"
echo "  Model:   $MODEL_PATH"
echo "  Address: http://${HOST}:${PORT}"
echo "  Context: ${CTX} tokens"
echo "  GPU layers: ${GPU_LAYERS}"
echo ""
echo "API is OpenAI-compatible. Test with:"
echo "  curl http://localhost:${PORT}/v1/models"
echo ""
echo "Example chat completion:"
echo '  curl http://localhost:'"${PORT}"'/v1/chat/completions \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '"'"'{"model":"nemotron","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256}'"'"
echo ""

exec "$LLAMA_SERVER" \
    -m "$MODEL_PATH" \
    -ngl "$GPU_LAYERS" \
    -c "$CTX" \
    -fa on \
    --host "$HOST" \
    --port "$PORT"
