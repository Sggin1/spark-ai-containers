# DGX Spark FP4 Investigation

A short, evidence-first investigation into what FP4 tensor-core support actually looks like on **NVIDIA DGX Spark (GB10, sm_121a)** — whether the headline "1 PFLOP FP4" performance number is reachable, and what FP4 paths *do* work on the silicon.

## TL;DR

- **The "1 PFLOP" `tcgen05` path is a silicon gap on GB10, not a software switch.** NVIDIA's own `ptxas` SASS encoder refuses to emit `tcgen05.mma` for `sm_121a` — the same way it refuses for `sm_120a` consumer Blackwell. GB10 silicon doesn't have the Tensor Memory + 2-CTA cluster-fabric MMA pipe that produces the marketing number. No patch fixes this.
- **Consumer warp-level FP4 *does* work on Spark, and is identical to the RTX 5060 Ti's FP4 path.** `mma.sync.aligned.kind::mxf4.block_scale...` compiles cleanly for `sm_121a` and emits the same SASS opcode (`OMMA.SF.16864.F32.E2M1.E2M1.E8`, byte-encoding `0x70f00e0c0808747f`) as `sm_120a`. Spark and the 5060 Ti share the same consumer-Blackwell FP4 silicon at the instruction level.
- **CuTeDSL 4.5.1+ supports Spark as a first-class target.** Earlier (4.4.x) reports of "missing kernel images for sm_121a" no longer apply — 4.5.x switched to runtime MLIR/NVVM JIT and added `sm_121a` to the relevant admissibility lists.

## What's in this repo

- **[`findings.md`](findings.md)** — full evidence trail with citations from NVIDIA's source and the verifying compile/SASS tests
- **[`kernel-patterns.md`](kernel-patterns.md)** — concrete patterns for integrating custom low-bit formats with consumer-Blackwell FP4 tensor cores, distilled from reading llama.cpp's MMQ framework
- **[`reference/fp4_mma_reference.cu`](reference/fp4_mma_reference.cu)** — self-contained working CUDA file with the canonical inline-PTX FP4 MMA syntax (MXFP4 and NVFP4 variants), compiles cleanly on `sm_120a` and `sm_121a`
- **[`reference/cutlass_dsl_starter.py`](reference/cutlass_dsl_starter.py)** — minimal CuTeDSL Python starter, verified running on Spark

## Reproducing

You need:
- Either an RTX 50-series (sm_120a) or DGX Spark (sm_121a), or both for the SASS comparison
- CUDA Toolkit 13.0+ (provides `nvcc`, `ptxas`, `cuobjdump`)
- For the Python side: `pip install 'nvidia-cutlass-dsl[cu13]==4.5.1'` in a venv

Quickest discriminating test:

```bash
# Write a single-line tcgen05 probe:
cat > probe.cu <<'EOF'
extern "C" __global__ void probe() {
    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;"
                 :: "r"(0u), "l"(0ull), "l"(0ull), "r"(0u));
}
EOF

# Target sm_100a (B200) — succeeds, emits UTCHMMA SASS:
nvcc -arch=sm_100a -ptx probe.cu

# Target sm_121a (Spark) — fails with:
#   "Instruction 'tcgen05.mma' not supported on .target 'sm_121a'"
nvcc -arch=sm_121a -ptx probe.cu
```

That single test settles the silicon-gap question. See `findings.md` for the full method and the inverse test (consumer-Blackwell `mxf4` MMA works on `sm_121a` byte-identically to `sm_120a`).

## Why this matters

The "1 PFLOP FP4" performance claim associated with DGX Spark / GB10 has produced months of confusion in NVIDIA developer forums and on social media. The number is reachable on **datacenter Blackwell** (B200, sm_100a) via the `tcgen05` MMA path, which uses Tensor Memory and 2-CTA cluster-fabric multicast. That pipe is **physically absent** on GB10 silicon. The FP4 throughput actually available on Spark is the consumer-Blackwell warp-level path — same as the RTX 5060 Ti scaled to Spark's clocks and power envelope. Real FP4 acceleration is available; the marketing peak is not.

This repo collects the empirical evidence in one place so people don't have to rederive it independently.

## See also

- **Sibling: [`../comfyui_spark_notes/`](../comfyui_spark_notes/)** — practical wheel-shadowing gotchas observed while running ComfyUI on Spark. Cites the SASS-equivalence finding here as the basis for the `sm_120` rebuild target on `sm_121` silicon.

## License

MIT — see [LICENSE](LICENSE).
