# Patterns for Low-Bit Custom Kernels on Consumer Blackwell

A practical guide to integrating a custom low-bit weight format with the consumer-Blackwell FP4 tensor-core MMA pipe (`OMMA.SF.16864`) on RTX 50-series and DGX Spark. Structural patterns are extracted from reading llama.cpp's MMQ (Matrix Multiply Quantized) framework, which already does this for MXFP4 and NVFP4. The same shape works for any block-scaled low-bit format you might want to integrate.

If you haven't already, read [`findings.md`](findings.md) first to understand what FP4 paths actually work on which arches.

---

## Overall MMQ kernel architecture

llama.cpp's `mmq.cuh` is a templated GEMM-with-quantization framework. The dispatch shape:

```
mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_*>
  ├── load_tiles      : packs quantized weights from global → shared memory tile
  ├── vec_dot_mma     : MMA-based dot-product (Blackwell, Turing+, AMD MFMA paths)
  └── vec_dot_dp4a    : DP4A fallback for older arches
```

Each quantization format provides three function specializations. The framework handles tiling, scheduling, accumulation, and write-back. **To add a new low-bit format, you provide these three functions and a type-traits specialization.**

References: `mmq.cuh:3300-3360` (type-traits dispatch); `mmq.cuh:135-145` (per-arch iteration sizes).

---

## Pattern: split fast path from fallback

```cpp
// mmq.cuh:3318
template <int mmq_x, int mmq_y, bool need_check>
struct mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_MXFP4> {
    static constexpr int vdr = VDR_MXFP4_Q8_1_MMQ;
#ifdef BLACKWELL_MMA_AVAILABLE
    static constexpr load_tiles_mmq_t load_tiles  = load_tiles_mxfp4_fp4<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_mxfp4_mxfp4_mma<mmq_x, mmq_y>;
#else
    static constexpr load_tiles_mmq_t load_tiles  = load_tiles_mxfp4<mmq_y, need_check>;
    static constexpr vec_dot_mmq_t    vec_dot_mma = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, ...>;
#endif
    static constexpr vec_dot_mmq_t    vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>;
};
```

Two compile-time paths:
- **Blackwell-native (sm_120a / sm_121a):** native FP4 tensor-core MMA via `mma.sync.aligned.kind::mxf4...`
- **Older arches:** dequantize FP4 → INT8 in-tile, then use existing Q8 INT8 MMA via the `vec_dot_q8_*` family

`BLACKWELL_MMA_AVAILABLE` is defined in `common.cuh:255` as:

```cpp
#if __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN
#    define BLACKWELL_MMA_AVAILABLE
#endif
```

i.e. the consumer-Blackwell window (currently sm_120, sm_121).

For your own format: keep a fast path that uses FP4 tensor cores (with quantization-projection from your representation to FP4) and a fallback that uses INT8 MMA with on-the-fly unpack. This gives you reasonable performance on every supported arch without writing N different optimized kernels.

---

## Pattern: tile-load with format conversion

`load_tiles_mxfp4_fp4` (mmq.cuh:793) shows the FP4-specific tile load:

```cpp
const block_mxfp4 * bxi = (const block_mxfp4 *) x + kbx0 + i * stride + kbx;
const int k0 = kbx * 4;
memcpy(x_qs + i * MMQ_MMA_TILE_X_K_FP4 + k0, bxi->qs, 16);  // 16 bytes = 32 FP4 values

// Pack 2 consecutive UE8M0 scales into one uint32
if (kbx % 2 == 0) {
    uint32_t e = bxi->e;
    e |= ((bxi + 1)->e << 8);
    x_sc[i * MMQ_MMA_TILE_X_K_FP4 + kbx / 2] = e;
}
```

Key observations:
- The X-tile is one shared-memory array carrying **both packed FP4 values** (`x_qs`) **and packed scale factors** (`x_sc`) — separate regions, contiguous in shared memory.
- Layout is dimensioned specifically for `ldmatrix` reads in the MMA path.
- Scale packing: 2 consecutive UE8M0 (8-bit) scales packed into one 32-bit word.

`load_tiles_nvfp4` (mmq.cuh:838) shows the LUT-based path for the older fallback:

```cpp
const int2 q0 = get_int_from_table_16(src_qs[2 * sub + 0], kvalues_mxfp4);
```

This converts FP4 codewords to INT8 via a 16-entry lookup table, so older arches can use INT8 MMA.

**For a custom low-bit format with K codewords:** the tile-load pattern is the same — either pack the codewords into the FP4 representation directly (if K ≤ 16 and your values can be projected to E2M1), or expand via a small codebook lookup to INT8 for the Q8 MMA fallback.

---

## Pattern: the MMA inner loop

`vec_dot_mxfp4_mxfp4_mma` (mmq.cuh:1066) is the actual dot-product:

```cpp
typedef tile<16, 8, int>   tile_A;   // FP4 packed as int32 (8 values per int)
typedef tile<8, 8, int>    tile_B;
typedef tile<16, 8, float> tile_C;   // FP32 accumulator

for each k01 step:
    load_ldmatrix(A[n][k], x_qs + offset, stride);  // ldmatrix.sync.aligned
    scaleA[n][k] = *(x_sc + ...);                   // per-quad scale assignment

for each j0 (B tile):
    for each k01:
        load_generic(B, y_qs + offset, stride);
        scaleB = y_sc[...];
        for each n (A minitile):
            tile_C C;
            mma_block_scaled(C, A[n][k], B, scaleA[n][k], scaleB);   // ← OMMA.SF emitted here
            sum[...] += C.x[l];
```

`mma_block_scaled` is the wrapper around the inline PTX:

```cpp
asm volatile(
    "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
    "%10, {0, 0}, %11, {0, 0};"
    : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
    : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]),
      "r"(Bxi[0]), "r"(Bxi[1]),
      "r"(a_scale), "r"(b_scale));
```

The full working file is in [`reference/fp4_mma_reference.cu`](reference/fp4_mma_reference.cu).

### Critical: per-quad scale-thread indexing

The PTX block-scaled MMA expects 2 threads in each lane quad (of 4) to supply the scale data. From mmq.cuh:1102:

```cpp
const int tidx = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
```

This indexing maps the 32-thread warp to the scale-supplying threads. The `MMQ_TILE_NE_K` strides are dimensioned to match. **If you copy this pattern for your own format, keep the scale-thread indexing intact — it's load-bearing for correctness, not a tuning choice.**

The `{0, 0}` immediate tuples in the PTX are `(byte_id_in_register, thread_offset)`. Both being zero means "scale factor data is at byte 0 of the source register, contributed by the canonically-assigned scale-supplying thread." Don't change these unless you're building a different scaling layout entirely.

---

## Integration patterns for custom low-bit formats

Three viable paths, in order of expected throughput:

### Path 1 — Project to FP4 (fastest, with bounded quantization error)

If your custom representation has at most ~16-32 distinct values per block, you can project each value to its nearest representable Float4E2M1FN value at preprocessing time. Store packed FP4 nibbles plus a per-block UE8M0 (MXFP4) or UE4M3 (NVFP4) scale that absorbs the dynamic range.

- **Throughput:** full consumer-Blackwell FP4 (the `OMMA.SF.16864` opcode emits with optimal scheduling).
- **Cost:** small quantization error from your representation → FP4 projection (~bounded by FP4 spacing near each of your codewords). If your representation matches FP4's value distribution (clustered near zero), the projection cost is minimal.

This is the right path for inference-time deployment where throughput matters and the additional quantization error is acceptable.

### Path 2 — Custom-format-exact via Q8 MMA (slower, no projection loss)

Store your representation in its native packed form (e.g. (sign, exp_a, exp_b) triplets for an additive-PoT format, or any custom packed codeword). During tile-load, expand each codeword to its INT8 value via a 256-entry codebook lookup. Pack as Q8 (INT8).

- **Throughput:** INT8 tensor cores (substantially slower than FP4 on consumer Blackwell, but still strong — Q8 MMA uses the `HMMA.16816.S8.S8.S32` family).
- **Cost:** none — your representation is preserved exactly through to the MMA inputs.

This is the right path when you want to validate that your custom representation works as designed (e.g. for research where you're testing the representation itself, not deployment performance).

### Path 3 — Hybrid

Use Path 2 during research/validation (your representation is preserved exactly during forward and backward passes), then Path 1 once you've established that the representation works and you're ready to optimize for deployment throughput.

The path you choose depends on whether your bottleneck is "does the representation work" or "can it run fast enough." For research-stage work, Path 2 first.

---

## Concrete integration starting point (llama.cpp's MMQ framework)

If you decide to integrate via llama.cpp's MMQ pattern (compatible with their wider GEMM scheduler):

1. Add `GGML_TYPE_YOUR_FORMAT` to ggml's type enum (in a fork — this is internal API).
2. Define `block_your_format` struct matching your on-disk packed layout.
3. Implement `load_tiles_your_format` (mirror of `load_tiles_mxfp4_fp4` at mmq.cuh:793).
4. Implement `vec_dot_your_format_mma` (mirror of `vec_dot_mxfp4_mxfp4_mma` at mmq.cuh:1066) — for Path 1, call the existing `mma_block_scaled` after projection; for Path 2, call the existing Q8 MMA wrapper.
5. Wire up the `mmq_type_traits` specialization at mmq.cuh:3320.

Compile with `BLACKWELL_MMA_AVAILABLE` defined (already enabled by `__CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL` for sm_120 and sm_121).

If you're using Triton instead, the structural pattern (load tile → unpack/dequant → MMA → accumulate → store) is the same, but Triton emits its own MMA scheduling and may need version-specific extensions for FP4. Check Triton's release notes for Blackwell consumer-FP4 support before committing — as of mid-2026, support is uneven and treats `sm_121` conservatively (see notes in NVIDIA forums; the situation may have improved by the time you read this).

---

## Things that are NOT load-bearing for correctness

- The exact tile sizes (`MMQ_TILE_NE_K = 32`, `mmq_y`, `mmq_x`). These are tuning parameters.
- The padding pattern (`mmq_y % 2 == 1 for dp4a`, `% 8 == 4 for mma`). Bank-conflict avoidance, optional but recommended.
- The granularity / minitile factoring. Scheduling choices.

## Things that ARE load-bearing

- The per-quad scale-thread indexing (`threadIdx.x / 4 + (threadIdx.x % 2) * 8`). PTX block-scaled MMA expects this exact layout.
- The X-tile layout combining `x_qs` (data) and `x_sc` (scales) in adjacent shared-memory regions, sized per the MMA expectation.
- The exact PTX form for the MMA (see `reference/fp4_mma_reference.cu`).

---

## References

- [`reference/fp4_mma_reference.cu`](reference/fp4_mma_reference.cu) — working CUDA file with canonical inline PTX
- [`reference/cutlass_dsl_starter.py`](reference/cutlass_dsl_starter.py) — minimal CuTeDSL Python entry point
- llama.cpp `ggml/src/ggml-cuda/mma.cuh:1054` — canonical MXFP4 MMA inline PTX
- llama.cpp `ggml/src/ggml-cuda/mmq.cuh:793` — `load_tiles_mxfp4_fp4`
- llama.cpp `ggml/src/ggml-cuda/mmq.cuh:1066` — `vec_dot_mxfp4_mxfp4_mma`
- llama.cpp `ggml/src/ggml-cuda/mmq.cuh:3320` — type-traits dispatch
- llama.cpp `ggml/src/ggml-cuda/common.cuh:255` — `BLACKWELL_MMA_AVAILABLE` macro
- CuTeDSL `cutlass/cute/nvgpu/warp/mma.py` — high-level Python API for the same MMA
