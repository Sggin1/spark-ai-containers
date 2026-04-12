# Build Recipe: llama.cpp with sm_121 on DGX Spark (GB10)

## Environment

| Component | Value |
|-----------|-------|
| Hardware | NVIDIA DGX Spark — GB10 Grace Blackwell |
| GPU | NVIDIA GB10, 128 GB unified VRAM, compute capability 12.1 |
| CPU | 20-core ARM (Grace) |
| OS | Ubuntu 24.04 |
| Kernel | 6.17.0-1008-nvidia |
| Driver | 580.126.09 |
| CUDA | 13.0 (Build cuda_13.0.r13.0/compiler.36424714_0) |
| Arch | aarch64 |
| GCC | 13.3.0 |

## Build Steps

### 1. Clone llama.cpp

```bash
cd ~/projects
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

**Commit used:** `463b6a963c2de376e102d878a50d26802f15833c`

### 2. Configure with CMake

```bash
cmake -B build \
  -DCMAKE_CUDA_ARCHITECTURES="121" \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
```

**Note:** CMake automatically replaces `121` with `121a` (arch-aware):
```
-- Replacing 121 in CMAKE_CUDA_ARCHITECTURES with 121a
-- Replacing 121-real in CMAKE_CUDA_ARCHITECTURES_NATIVE with 121a-real
-- Using CMAKE_CUDA_ARCHITECTURES=121a CMAKE_CUDA_ARCHITECTURES_NATIVE=121a-real
```

**Warning (non-fatal):** OpenSSL not found — HTTPS support disabled in the built-in server. Not needed for local LAN serving.

### 3. Build

```bash
cmake --build build --config Release -j$(nproc)
```

- Build time: ~3 minutes on 20-core Grace ARM
- All binaries produced in `build/bin/`

### 4. Verify

```bash
ls build/bin/llama-bench build/bin/llama-cli build/bin/llama-server
```

## Model: Getting the GGUF

### Option A: Download from HuggingFace

The upstream-compatible GGUF is [ggml-org/nemotron-3-super-120b-GGUF](https://huggingface.co/ggml-org/nemotron-3-super-120b-GGUF):

```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='ggml-org/nemotron-3-super-120b-GGUF',
    filename='Nemotron-3-Super-120B-Q4_K.gguf',
    local_dir='/path/to/models/'
)
"
```

This is a 65 GiB Q4_K quantization.

### Option B: Reuse Ollama's GGUF — not compatible

Ollama's GGUF for MoE models uses a different tensor layout (split expert dimensions) than upstream llama.cpp. Loading it produces:

```
check_tensor_dims: tensor 'blk.1.ffn_down_exps.weight' has wrong shape;
expected 2688, 4096, 512, got 2688, 1024, 512, 1
```

Use the ggml-org GGUF or convert from safetensors with `llama.cpp/convert_hf_to_gguf.py`.

If you want to find Ollama's blob path (for reference):
```bash
# Find manifest
cat ~/.ollama/models/manifests/registry.ollama.ai/library/nemotron-3-super/120b-a12b-q4_K_M | python3 -m json.tool
# The largest layer digest is the model blob in:
# $(readlink -f ~/.ollama/models)/models/blobs/sha256-<hash>
```

## Running the Server

In recent llama.cpp builds, the `-fa` flag requires an explicit argument:

```bash
./build/bin/llama-server \
  -m /path/to/models/Nemotron-3-Super-120B-Q4_K.gguf \
  -ngl 99 -c 32768 -fa on \
  --host 0.0.0.0 --port 8090
```

Bare `-fa` without `on` causes a parse error. Use `-fa on`.

## OOM on Model Load

The 66 GB GGUF needed ~73 GB memory when loaded (weights + KV cache + CUDA context). On a 128 GB system, Linux page cache can consume most of the "free" memory.

Symptoms: llama-server gets killed by the OOM killer during model load, or hangs with no output.

Workaround: drop the page cache before loading:

```bash
# Free page cache (requires sudo)
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Also stop Ollama if it's running (it holds GPU memory)
sudo systemctl stop ollama

# Then start the server
./build/bin/llama-server \
  -m /path/to/models/Nemotron-3-Super-120B-Q4_K.gguf \
  -ngl 99 -c 32768 -fa on \
  --host 0.0.0.0 --port 8090
```

On the DGX Spark's unified memory architecture, both CPU and GPU share the same 128 GB pool. The model needs to fit alongside the OS and any other processes.

## Errors Encountered

1. **Ollama GGUF incompatibility:** MoE expert tensor layout differs between Ollama and upstream llama.cpp. Ollama packs expert weights in groups of 4 (dim=1024) while upstream expects concatenated (dim=4096). Loading Ollama's blob produces: `check_tensor_dims: tensor 'blk.1.ffn_down_exps.weight' has wrong shape; expected 2688, 4096, 512, got 2688, 1024, 512, 1`. Workaround: use ggml-org GGUF.
2. **VRAM reporting:** `nvidia-smi` reports memory as `[N/A]` for `memory.used`/`memory.free` on GB10 unified memory. Use `nvidia-smi` process list instead.
3. **No OpenSSL:** Build warning only, not needed for local serving.
4. **OOM on model load:** Page cache filled available memory. Workaround: drop caches before loading (see above).
5. **`-fa` parse error:** Newer llama.cpp requires `-fa on` instead of bare `-fa`.

## Quick Test

```bash
./build/bin/llama-bench \
  -m /path/to/models/Nemotron-3-Super-120B-Q4_K.gguf \
  -p 512 -n 128 -r 3
```
