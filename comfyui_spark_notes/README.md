# ComfyUI on DGX Spark — wheel-shadowing notes

Observations from running ComfyUI on DGX Spark (GB10, sm_121, aarch64, CUDA 13) with the [Triplany/comfyui-dgx-spark](https://github.com/Triplany/comfyui-dgx-spark) kit applied. Two real pip-shadowing risks that can silently undo the kit's hardware-specific wheel installs.

Verified end-to-end on a live ComfyUI install on 2026-05-22:
- Sage 2.2.0 rebuilt against current torch 2.10/2.11 + CUDA 13, kernels confirmed at `sm_120`
- onnxruntime-gpu 1.25.0 (Jay0515 community wheel) installed, `CUDAExecutionProvider` confirmed
- Shadowing mechanism reproduced and traced to shared import-path declaration

## TL;DR

- `pip install onnxruntime` (direct or transitive) **silently kills DWPose / ONNX GPU acceleration** by overwriting the Jay0515 sm_121 build with PyPI's CPU-only wheel. No error, no warning — just slow gens.
- `pip install sageattention` overwrites the local rebuild against your specific torch/CUDA combo. Less catastrophic (PyPI wheel also has sm_120 kernels) but can cause ABI mismatches on torch upgrades.
- Both are idempotent to recover from if you know to look — `build/onnxruntime.sh` and `build/sage.sh` in the upstream kit handle re-install.

## `onnxruntime` from PyPI shadows the Jay0515 sm_121 wheel

The upstream kit's `build/onnxruntime.sh` installs [Jay0515's `onnxruntime-gpu` 1.25.0 wheel](https://huggingface.co/Jay0515/onnxruntime-gpu-aarch64-cuda13-sm121) — the only sm_121/aarch64/cu13 build I'm aware of. PyPI's `onnxruntime` package on aarch64 is **CPU-only** (no `CUDAExecutionProvider`) and ships at version 1.26.0 — *higher* than the community wheel.

The shadowing mechanism: both packages declare `onnxruntime` as their `top_level` import name, even though their PyPI distribution names differ (`onnxruntime` vs `onnxruntime-gpu`). They install to the **same `site-packages/onnxruntime/` directory**. Pip doesn't flag this as a conflict because it only checks distribution names, so any `pip install onnxruntime` (direct or transitive) overwrites the GPU files in-place with no warning.

### Reproducing

Starting from a healthy state (`CUDAExecutionProvider` available), the shadowing trigger is any of:

- `pip install onnxruntime` directly
- `pip install -r requirements.txt` where the file lists `onnxruntime`
- Any custom-node install whose own `requirements.txt` pulls `onnxruntime` transitively
- `pip install --upgrade` against a list that touches it

After the trigger:

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# Unhealthy: ['AzureExecutionProvider', 'CPUExecutionProvider']
# Healthy:   ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Symptom

DWPose, controlnet preprocessors, and any other ONNX-backed nodes silently fall back to CPU. Generations still succeed — just much slower. ComfyUI startup logs may include a benign `GPU device discovery failed` line from onnxruntime that's present even when the GPU build is working, so log-grepping for that string isn't a reliable detector. The `get_available_providers()` check is.

### Verifying mechanism (distribution metadata)

```bash
python -c "
import importlib.metadata as m
for pkg in ['onnxruntime', 'onnxruntime-gpu']:
    try:
        d = m.distribution(pkg)
        print(f'{pkg}: version={d.version}, top_level={d.read_text(\"top_level.txt\").strip().splitlines()}')
    except m.PackageNotFoundError:
        print(f'{pkg}: NOT INSTALLED')
"
```

Healthy output (Jay0515 wheel installed, no PyPI shadow):
```
onnxruntime: NOT INSTALLED
onnxruntime-gpu: version=1.25.0, top_level=['onnxruntime']
```

Both packages claim `onnxruntime` as their import path — that's why the shadowing is silent at the pip level.

### Recovery

Re-run the upstream kit's `build/onnxruntime.sh`. Idempotent — uninstalls anything shadowing, reinstalls Jay0515.

## `pip install sageattention` overwrites the local rebuild

Softer risk. The upstream kit's `build/sage.sh` rebuilds SageAttention 2.2 from source against the local torch + CUDA toolchain, targeting `TORCH_CUDA_ARCH_LIST="12.0"`. Setup.py's `SUPPORTED_ARCHS` doesn't list `"12.1"`, but the SASS-equivalence finding in the sibling [`fp4_SASS/`](../fp4_SASS/) directory confirms `sm_120` kernels execute natively on GB10's `sm_121` silicon — so this is the correct target, not a workaround.

Any subsequent `pip install sageattention` or `pip install --upgrade` will reinstall the PyPI wheel, replacing the locally-built `.so` files. The PyPI wheel **also** ships `sm_120` kernels, so this isn't a CPU-fallback situation — FP8 attention still runs on tensor cores. What you lose:

- Build against your specific torch + CUDA combination (can surface as subtle ABI mismatches on torch upgrades)
- Whatever local patches your build pipeline applied (if any)

### Checking

```bash
/usr/local/cuda/bin/cuobjdump --list-text \
  $HOME/ComfyUI/.venv/lib/python3.12/site-packages/sageattention/_qattn_sm80*.so \
  | grep -oE "sm_[0-9]+" | sort -u
```

Healthy output: `sm_120`. Not `sm_121` — that's deliberate; see [`fp4_SASS/findings.md`](../fp4_SASS/findings.md) for the SASS-level explanation.

### Recovery

Re-run upstream's `build/sage.sh`. Idempotent — skips if kernels already at sm_120, rebuilds otherwise.

## Defensive option: pin in `requirements.txt`

If you maintain a `requirements.txt` for a workflow that uses the upstream kit, pin both:

```
onnxruntime-gpu==1.25.0   # Jay0515 sm_121/aarch64/cu13 build — do not let PyPI 1.26 shadow it
sageattention==2.2.0      # then run upstream's build/sage.sh to rebuild against current torch
```

This stops your own `pip install -r` from clobbering them. Won't stop a custom node from doing it on its own install — re-run the upstream kit's `verify.sh` after any custom-node update.

## Why this happens (root cause)

PyPI is `pip`'s source of truth for version resolution, and PyPI's wheels target the broadest CPU/GPU baselines. Hardware-specific builds (sm_121 aarch64+cu13 ORT) and box-specific rebuilds (SageAttention against your local torch) only exist as community wheels or local artifacts, and pip has no way to know they're "better" than what's on PyPI. Any normal pip operation can undo what the upstream kit set up.

Until upstream package maintainers ship aarch64 GPU wheels for `onnxruntime` (no current ETA from Microsoft on this), the workaround is to keep checking and re-running the relevant `build/*.sh` script. Both are idempotent.

## Environment

- **Hardware:** NVIDIA DGX Spark (GB10, sm_121)
- **OS:** Ubuntu 24.04 LTS, aarch64
- **CUDA:** 13.x
- **PyTorch:** 2.10.x / 2.11.x (cu130 aarch64)
- **ComfyUI:** master @ `b6332446` or newer
- **Upstream kit:** [Triplany/comfyui-dgx-spark](https://github.com/Triplany/comfyui-dgx-spark) (kit used; observations are mine)

## See also

- **Sibling: [`../fp4_SASS/`](../fp4_SASS/)** — SASS-level investigation of FP4 MMA paths on Spark. Establishes the sm_120 / sm_121 SASS equivalence cited above.
- **Upstream: [Triplany/comfyui-dgx-spark](https://github.com/Triplany/comfyui-dgx-spark)** — the practical ComfyUI integration kit these notes are observed against.
- **Sibling: [`../nvfp4/`](../nvfp4/)** — NVFP4 on Spark (friction → working); March landscape snapshot under [`../nvfp4/archive/`](../nvfp4/archive/).
