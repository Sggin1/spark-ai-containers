# Findings: DGX Spark FP4 Support, at the SASS Encoder Level

**Hardware tested:** NVIDIA DGX Spark (GB10, sm_121a, Ubuntu 24.04 aarch64, driver 580.142) + NVIDIA RTX 5060 Ti 16 GB (sm_120a, x86_64 Windows).
**Toolchain:** CUDA 13.2 (`nvcc`, `ptxas`, `cuobjdump`), `nvidia-cutlass-dsl[cu13] == 4.5.1`.
**Date:** May 2026.

---

## Verdict

The "missing 1 PFLOP FP4" on DGX Spark is a **silicon-level gap**, not a software gate. Empirically confirmed at three layers — Python admissibility, the `ptxas` SASS encoder, and live runtime arch detection on the actual hardware.

The consumer-Blackwell warp-level FP4 MMA path *does* work on Spark, and emits **byte-identical SASS** to the RTX 5060 Ti. The marketing PFLOP number lives in a different MMA pipe (`tcgen05`) that is datacenter-only and physically absent on GB10.

---

## How CuTeDSL ships kernels (the framing the old forum threads got wrong)

Earlier reports (circa CuTeDSL 4.4.x) framed the issue as "missing precompiled kernel images for sm_121a." That framing no longer applies in 4.5.1. Pulling the wheels (`nvidia-cutlass-dsl-libs-base` and `nvidia-cutlass-dsl-libs-cu13`, ~155 MB combined) and running `cuobjdump --list-elf` on every `.so` returns **`does not contain device code`** for every file. There are zero `.cubin`, `.fatbin`, or `.ptx` files in either wheel.

CuTeDSL 4.5.x switched to runtime MLIR/NVVM JIT compilation. The shipped runtime (`libcute_dsl_runtime.so`, ~40 MB) holds the MLIR pipeline and lowers user-written cute DSL kernels through MLIR → NVVM → PTX → SASS at execution time, using the driver's JIT. There is no per-arch packaging step that could "miss" `sm_121a`. The earlier framing is obsolete.

This eliminates an entire class of bug, so the modern question becomes: does NVIDIA's compiler stack actually know how to emit SASS for the FP4 instructions on `sm_121a`?

---

## Test A — `tcgen05.mma` (the headline "1 PFLOP" path)

`tcgen05.mma` is the datacenter Blackwell MMA — it uses Tensor Memory (TMEM) operands and supports 2-CTA cluster-fabric multicast (the "kind::f4" + "scale_vec" + "block_scale" variants are what advertise the "1 PFLOP" FP4 number for B200).

Compiled a minimal CUDA file with inline `tcgen05.mma.cta_group::1.kind::f16` and `tcgen05.alloc` for three Blackwell targets:

| Target | Result |
|---|---|
| `sm_100a` (B200 datacenter) | ✅ Compiles. Emits SASS opcode `UTCHMMA gdesc[UR8], gdesc[UR10], tmem[UR6], tmem[UR4], idesc[UR5], UPT` with TMEM descriptors — encoding `0xff040a080075ea`. |
| `sm_120a` (RTX 5060 Ti, consumer) | ❌ `ptxas error: Instruction 'tcgen05.mma' not supported on .target 'sm_120a'` |
| **`sm_121a`** (DGX Spark) | ❌ `ptxas error: Instruction 'tcgen05.mma' not supported on .target 'sm_121a'` — identical errors to sm_120a |

**Why this is decisive.** `ptxas` is NVIDIA's PTX-to-SASS encoder. It contains per-arch instruction-encoding tables. If `sm_121a` silicon could execute `tcgen05.mma`, NVIDIA's own compiler would have an encoding for it — there would be no reason to block their own customers from features the silicon supports. The compiler outright refusing to encode is evidence one layer above the silicon datasheet, from the team that defines the instruction set.

The bonus discovery: `tcgen05.mma` on `sm_100a` lowers to the SASS mnemonic **`UTCHMMA`** with TMEM-typed operands. That is the literal "1 PFLOP path" at the lowest level. Spark cannot emit this opcode.

---

## Test B — Warp-level FP4 MMA (the consumer-Blackwell path)

The other FP4 entry point on Blackwell is the warp-level `mma.sync.aligned.kind::mxf4.block_scale...` instruction. This is what CuTeDSL exposes via `MmaMXF4Op` (MXFP4 variant, UE8M0 scales) and `MmaMXF4NVF4Op` (NVFP4 variant, UE4M3 scales).

Canonical PTX syntax (extracted from llama.cpp's `ggml/src/ggml-cuda/mma.cuh:1054`, under their `BLACKWELL_MMA_AVAILABLE` guard):

```ptx
mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
  {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3},
  %10, {0, 0}, %11, {0, 0};
```

Compiled the same source for the same three arches:

| Target | Result |
|---|---|
| `sm_100a` (B200 datacenter) | ❌ `ptxas error: Instruction 'mma with block scale' not supported on .target 'sm_100a'` |
| **`sm_120a`** (RTX 5060 Ti) | ✅ Compiles. SASS: `OMMA.SF.16864.F32.E2M1.E2M1.E8 R8, R8, R12, R16, R14, R15, URZ` — encoding `0x70f00e0c0808747f` |
| **`sm_121a`** (DGX Spark) | ✅ Compiles. SASS: `OMMA.SF.16864.F32.E2M1.E2M1.E8 R8, R8, R12, R16, R14, R15, URZ` — encoding `0x70f00e0c0808747f` |

The `sm_120a` and `sm_121a` SASS dumps are **byte-for-byte identical** apart from the ELF header arch tag (`EF_CUDA_SM120` vs `EF_CUDA_SM121`). Same opcode, same register allocation, same encoded instruction bytes.

The NVFP4 variant (`.kind::mxf4nvf4.scale_vec::4X` ... `.ue4m3`) emits the analogous `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` opcode and is also byte-identical between the two arches.

---

## Test C — CuTeDSL Python API on actual Spark hardware

Installed `nvidia-cutlass-dsl[cu13] == 4.5.1` into a fresh venv on Spark (~298 MB, no torch dependency). The library successfully:

1. Detects the GPU as `Arch.sm_121a` via `cutlass.base_dsl.runtime.cuda` (it has an explicit `(12, 1): ("Blackwell", "sm_121a", ["sm_121a"]),  # DGX Spark` entry).
2. Instantiates `MmaMXF4Op` and `MmaMXF4NVF4Op` without error — `MmaSM120BlockScaledOp.__post_init__` runs `arch = CuTeDSL._get_dsl().get_arch_enum()` and the check `arch in [Arch.sm_120a, Arch.sm_121a, Arch.sm_120f]` passes.

This is end-to-end runtime confirmation that the CuTeDSL Python API treats DGX Spark as a first-class target.

---

## The full Blackwell-family FP4 MMA support matrix

| | `tcgen05.mma` (datacenter, "1 PFLOP" path, TMEM + 2-CTA multicast) | `mma.sync.aligned.kind::mxf4(nvf4).block_scale` (consumer warp-level FP4) |
|---|---|---|
| `sm_100a` (B200) | ✅ `UTCHMMA` SASS | ❌ `not supported on .target 'sm_100a'` |
| `sm_120a` (RTX 5060 Ti) | ❌ `not supported on .target 'sm_120a'` | ✅ `OMMA.SF.16864.F32.E2M1.E2M1.E8` |
| `sm_121a` (DGX Spark) | ❌ `not supported on .target 'sm_121a'` | ✅ `OMMA.SF.16864.F32.E2M1.E2M1.E8` (byte-identical to sm_120a) |

The two MMA pipes do not overlap. NVIDIA designed Blackwell with a hard split between the datacenter and consumer tensor-core pipelines, with FP4 support partitioned cleanly between them. The marketing PFLOP number lives only in the datacenter pipe. Spark is on the consumer pipe.

---

## Practical implications

- **FP4 tensor-core acceleration is available on DGX Spark today.** Use `MmaMXF4Op` / `MmaMXF4NVF4Op` from `cutlass.cute.nvgpu.warp.mma`, or inline `mma.sync.aligned.kind::mxf4.block_scale...` PTX, or CUTLASS C++ block-scaled GEMM templates. All target the same `OMMA.SF.16864` SASS opcode on `sm_121a`.
- **Expected throughput is consumer-Blackwell-class**, scaled by Spark's clocks and power envelope. The 5060 Ti and Spark share the same FP4 silicon path — performance per-clock should match very closely. Spark's overall throughput will be governed by its TDP (~240 W) and clocks, not by missing silicon features within the consumer FP4 pipe.
- **The "1 PFLOP FP4" advertised figure is not reachable on this hardware.** That number describes the `tcgen05` block-scaled MMA with 2-CTA multicast, which requires datacenter Blackwell silicon. No driver update or CUTLASS patch will recover it on GB10.
- **Don't waste time patching CUTLASS admissibility lists for `tcgen05` on `sm_121a`.** The constraint is `ptxas` itself, not the user-facing API.

---

## What changed in CuTeDSL between 4.4 and 4.5

Two simultaneous changes, both of which were necessary to fix the old "Spark can't run FlashAttention 4 / FP4 kernels" reports:

1. **Packaging architecture: precompiled per-arch kernel images → runtime MLIR/NVVM JIT.** The wheel no longer ships `.cubin`/`.fatbin` files for specific archs. Kernels are generated, lowered, and compiled at execution time. This eliminates the packaging-gap class of bug forever.
2. **Admissibility lists: `sm_121a` added.** `MmaSM120BlockScaledOp.admissible_archs = ["sm_120a", "sm_121a", "sm_120f"]`. The `base_dsl/runtime/cuda.py` GPU arch map has explicit `(12, 1): ("Blackwell", "sm_121a", ["sm_121a"])  # DGX Spark`.

Forum threads about FP4 access on Spark that predate CuTeDSL 4.5.x are largely obsolete. The current state in 4.5.1 is what's documented here.

---

## Source citations

NVIDIA source files referenced (all line numbers are CuTeDSL 4.5.1):

- `cutlass/base_dsl/arch.py:80-150` — `Arch` enum including `sm_121`, `sm_121a`, `sm_121f`; `BlackwellArchs()` listing
- `cutlass/base_dsl/runtime/cuda.py:86-99` — `gpu_arch_map` with explicit DGX Spark entry
- `cutlass/cute/nvgpu/warp/mma.py:244` — `MmaSM120BlockScaledOp.admissible_archs = ["sm_120a", "sm_121a", "sm_120f"]`
- `cutlass/cute/nvgpu/warp/mma.py:391-470` — `MmaMXF4Op` and `MmaMXF4NVF4Op` docstrings with PTX qualifier specifications
- `cutlass/cute/nvgpu/tcgen05/mma.py:186-189, 387-390` — datacenter `tcgen05` admissibility (sm_100a/sm_103a only)

llama.cpp source for canonical FP4 MMA inline PTX:

- `ggml/src/ggml-cuda/mma.cuh:1054` — `mma_block_scaled` wrapper with the canonical PTX template
- `ggml/src/ggml-cuda/common.cuh:255` — `BLACKWELL_MMA_AVAILABLE` macro definition

---

## Raw test outputs

Raw outputs of the probing and compile tests (cuobjdump dumps, SASS diffs, ptxas error messages, CuTeDSL install logs) are not included in this repo to keep it lean. They're available on request — open an issue if you'd like to see the unedited trail.
