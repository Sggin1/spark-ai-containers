"""cutlass_dsl_starter.py

Minimal starter showing CuTeDSL's consumer-Blackwell FP4 MMA path on
DGX Spark (sm_121a). Confirmed working on:

  - DGX Spark (GB10, sm_121a), driver 580.142, CUDA 13.2
  - nvidia-cutlass-dsl[cu13] == 4.5.1 in a fresh Python 3.12 venv

Setup:

    python3 -m venv ~/cutlass-fp4-venv
    ~/cutlass-fp4-venv/bin/pip install 'nvidia-cutlass-dsl[cu13]==4.5.1'

Run:

    ~/cutlass-fp4-venv/bin/python cutlass_dsl_starter.py

Expected output: object summaries for both MXFP4 (UE8M0 scale) and NVFP4
(UE4M3 scale) warp-level MMA atoms, with admissible_archs including sm_121a.

The MMA atoms emitted here lower to the SASS opcode:

  OMMA.SF.16864.F32.E2M1.E2M1.E8        (MXFP4 variant)
  OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X  (NVFP4 variant)

Both are byte-identical between sm_120a (RTX 50-series) and sm_121a (Spark)
SASS — verified via cuobjdump --dump-sass.
"""

import cutlass
from cutlass import Float4E2M1FN, Float32, Float8E8M0FNU, Float8E4M3FN
from cutlass.cute.nvgpu.warp.mma import (
    MmaMXF4Op,
    MmaMXF4NVF4Op,
    MmaSM120BlockScaledOp,
)
from cutlass.base_dsl.arch import Arch


def main() -> None:
    print(f"CuTeDSL version: {cutlass.__version__}")
    print(f"MmaSM120BlockScaledOp.admissible_archs: {MmaSM120BlockScaledOp.admissible_archs}")
    print()

    # MXFP4: per-block UE8M0 scale factors, scale_vec::2X.
    mxf4 = MmaMXF4Op(
        ab_dtype=Float4E2M1FN,
        acc_dtype=Float32,
        sf_type=Float8E8M0FNU,
    )
    print("=== MmaMXF4Op (MXFP4, UE8M0 scales) ===")
    print(mxf4)
    print()

    # NVFP4: per-block UE4M3 scale factors, scale_vec::4X.
    nvf4 = MmaMXF4NVF4Op(
        ab_dtype=Float4E2M1FN,
        acc_dtype=Float32,
        sf_type=Float8E4M3FN,
    )
    print("=== MmaMXF4NVF4Op (NVFP4, UE4M3 scales) ===")
    print(nvf4)
    print()

    print("Both MMA atoms instantiated successfully — arch check passed.")
    print()
    print("Underlying SASS opcode on sm_120a and sm_121a:")
    print("  OMMA.SF.16864.F32.E2M1.E2M1.E8         (MXFP4 variant)")
    print("  OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X   (NVFP4 variant)")
    print()
    print("Tile shape (M, N, K): (16, 8, 64) — composes 32 threads of one warp.")


if __name__ == "__main__":
    main()
