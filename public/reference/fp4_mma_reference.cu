// fp4_mma_reference.cu
//
// Self-contained working reference for consumer-Blackwell FP4 tensor-core MMA
// on DGX Spark (sm_121a) and RTX 50-series (sm_120a). Compiles cleanly with
// nvcc 13.x for either architecture and emits identical SASS:
//
//     OMMA.SF.16864.F32.E2M1.E2M1.E8 ... ;  /* 0x70f00e0c0808747f */     (MXFP4)
//     OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X ... ;                          (NVFP4)
//
// Compile examples:
//   nvcc -arch=sm_121a -cubin -o fp4_mma.cubin fp4_mma_reference.cu
//   nvcc -arch=sm_120a -cubin -o fp4_mma.cubin fp4_mma_reference.cu
//   cuobjdump --dump-sass fp4_mma.cubin
//
// PTX syntax adapted from llama.cpp/ggml/src/ggml-cuda/mma.cuh (BLACKWELL_MMA_AVAILABLE
// guard, ggerganov's hand-rolled block-scaled FP4 MMA).
//
// What this is NOT: a complete FP4 GEMM. It's the minimal MMA atom — a single
// warp computes D += A·B for a (M=16, N=8, K=64) tile of packed FP4 data with
// per-block UE8M0 (MXFP4) or UE4M3 (NVFP4) scale factors. To build a GEMM, tile
// this across the K axis and across warps for M/N — or use CuTeDSL's
// MmaMXF4Op / MmaMXF4NVF4Op which factor that for you.
//
// FP4 packing: each 32-bit register in A and B holds 8 packed FP4 (E2M1) values.
// A holds 4 registers × 8 values = 32 FP4 values per thread, B holds 2 × 8 = 16.
// Across the 32 threads of a warp, this realizes the m16n8k64 tile.
//
// Scale factor layout: a_scale and b_scale are 32-bit registers carrying packed
// scale values. The "{0, 0}" tuples in the asm are (byte_id_in_reg, thread_offset)
// immediates — the canonical form for block-scaled MMA when each thread holds
// its own scale data starting at byte 0.

#include <cstdio>
#include <cstdint>

extern "C" __global__ void fp4_mxfp4_mma(
        float* __restrict__ D,        // [4] accumulator output (per thread; warp-wide is 32×4)
        const unsigned* __restrict__ A,  // [4] packed FP4 values for A operand
        const unsigned* __restrict__ B,  // [2] packed FP4 values for B operand
        unsigned a_scale,             // packed UE8M0 scale factors for A block
        unsigned b_scale)             // packed UE8M0 scale factors for B block
{
    float Dxi[4] = {D[0], D[1], D[2], D[3]};
    unsigned Axi[4] = {A[0], A[1], A[2], A[3]};
    unsigned Bxi[2] = {B[0], B[1]};

    // MXFP4 form: kind::mxf4, scale_vec::2X, scale type UE8M0.
    // D += A·B with block scaling applied to A and B.
    asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, "       // D[4] (output)
        "{%4, %5, %6, %7}, "       // A[4] (FP4 packed)
        "{%8, %9}, "               // B[2] (FP4 packed)
        "{%0, %1, %2, %3}, "       // C[4] (input accumulator = D)
        "%10, {0, 0}, "            // A scale + (byte_id, thread_id) imm tuple
        "%11, {0, 0};"             // B scale + (byte_id, thread_id) imm tuple
        : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]),
          "r"(Bxi[0]), "r"(Bxi[1]),
          "r"(a_scale), "r"(b_scale));

    D[0] = Dxi[0]; D[1] = Dxi[1]; D[2] = Dxi[2]; D[3] = Dxi[3];
}

extern "C" __global__ void fp4_nvfp4_mma(
        float* __restrict__ D,
        const unsigned* __restrict__ A,
        const unsigned* __restrict__ B,
        unsigned a_scale,
        unsigned b_scale)
{
    float Dxi[4] = {D[0], D[1], D[2], D[3]};
    unsigned Axi[4] = {A[0], A[1], A[2], A[3]};
    unsigned Bxi[2] = {B[0], B[1]};

    // NVFP4 form: kind::mxf4nvf4, scale_vec::4X, scale type UE4M3.
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
        : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]),
          "r"(Bxi[0]), "r"(Bxi[1]),
          "r"(a_scale), "r"(b_scale));

    D[0] = Dxi[0]; D[1] = Dxi[1]; D[2] = Dxi[2]; D[3] = Dxi[3];
}
