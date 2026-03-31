# TurboQuant-patched vLLM for DGX Spark (SM121)
# Extends vllm-node:latest (eugr community build with sm_121 FlashInfer)
# Adds TurboQuant KV cache compression from vLLM PR #38479
#
# Build: docker build -t vllm-turboquant:latest -f build/Dockerfile build/
# Run:   See scripts/safe_run.sh for memory-safe launcher

FROM vllm-node:latest

# Copy TurboQuant new files
COPY new_files/vllm/turboquant/ /usr/local/lib/python3.12/dist-packages/vllm/turboquant/
COPY new_files/vllm/v1/attention/backends/turboquant_attn.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/
COPY new_files/vllm/v1/attention/ops/triton_tq_decode.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/
COPY new_files/vllm/v1/attention/ops/triton_tq_store.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/

# CUDA kernels — copy source, compile at build time for sm_121
COPY new_files/vllm/v1/attention/ops/csrc/ /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/csrc/

# Apply patches to existing files
COPY patch_vllm.py /tmp/patch_vllm.py
RUN python3 /tmp/patch_vllm.py && rm /tmp/patch_vllm.py

# Clear stale bytecode cache so Python picks up patched files
RUN find /usr/local/lib/python3.12/dist-packages/vllm -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Install turboquant-torch as fallback (pure PyTorch, used if Triton/CUDA kernels fail)
RUN pip install --no-cache-dir turboquant-torch==0.2.2

# Verify the patch applied
RUN python3 -c "from vllm.turboquant import TurboQuantConfig; print('TurboQuant module OK')" 2>/dev/null || \
    python3 -c "import importlib; m = importlib.import_module('vllm.turboquant'); print('TurboQuant import OK')"

# Label
LABEL maintainer="DGX Spark community"
LABEL description="vLLM + TurboQuant KV cache compression (PR #38479) for SM121"
