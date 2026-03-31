# mamba-ssm + causal-conv1d for DGX Spark (aarch64, SM121)
# Builds from GitHub source to fix PyPI x86_64 wheel + ABI mismatch issues.
#
# Build:  docker build -t spark-mamba-dev:latest .
# Run:    docker run --runtime=nvidia --gpus all -it spark-mamba-dev:latest

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV MAMBA_FORCE_BUILD=TRUE
ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
ENV TORCH_CUDA_ARCH_LIST="12.1"

# Build causal-conv1d from source (PyPI wheel is x86_64 only)
RUN pip install --no-cache-dir --no-build-isolation \
    "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git"

# Build mamba-ssm from source (PyPI has ABI mismatch with PyTorch 2.10+)
RUN pip install --no-cache-dir --no-build-isolation \
    "mamba-ssm @ git+https://github.com/state-spaces/mamba.git"

# Dependencies for NemotronH VL models
RUN pip install --no-cache-dir \
    "transformers>=4.48,<5.0" \
    accelerate \
    timm \
    open_clip_torch

# Verify CUDA extensions load
RUN python3 -c "\
import causal_conv1d_cuda; print('causal_conv1d_cuda OK'); \
import selective_scan_cuda; print('selective_scan_cuda OK'); \
import mamba_ssm; print(f'mamba-ssm {mamba_ssm.__version__} OK')"

LABEL maintainer="DGX Spark community"
LABEL description="mamba-ssm built from source for aarch64 + SM121"
