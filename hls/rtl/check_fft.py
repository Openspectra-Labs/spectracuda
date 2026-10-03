"""Score cp_fft against spectracuda.

Correlation, not bit-exactness. The reference is numpy's unscaled FFT of
the same int16 samples the RTL sees, so input quantization is already in
both sides and the residual is the core's own truncation rounding.

Reported three ways, because they fail differently:
  correlation   catches a wrong CP offset or a bin ordering error --
                either destroys correlation while leaving magnitudes
                plausible
  scale factor  catches a scaling-convention mismatch, which would show
                as near-perfect correlation with a constant gain != 1
  EVM           the residual once alignment and scale are right
"""
from __future__ import annotations

import os
import sys

import numpy as np

EVM_TOL = 0.02
CORR_TOL = 0.999


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    b = os.path.join(here, "build")
    ref = np.loadtxt(os.path.join(b, "fft_ref.txt"))
    got = np.loadtxt(os.path.join(b, "fft_out.txt"), ndmin=2)

    r = ref[:, 0] + 1j * ref[:, 1]
    if got.size == 0:
        print("FAIL: no output produced")
        sys.exit(1)
    g = got[:, 0] + 1j * got[:, 1]
    if len(g) < len(r):
        print(f"FAIL: got {len(g)} bins, expected {len(r)}")
        sys.exit(1)
    g = g[:len(r)]

    corr = float(np.abs(np.vdot(r, g)) / (np.linalg.norm(r) * np.linalg.norm(g)))
    scale = float(np.abs(np.vdot(r, g)) / np.vdot(r, r).real)
    evm = float(np.linalg.norm(g / scale - r) / np.linalg.norm(r)) if scale else 1.0

    print(f"bins compared : {len(r)}")
    print(f"  correlation : {corr:.6f}  "
          f"{'OK' if corr >= CORR_TOL else 'FAIL (CP offset or bin order?)'}")
    print(f"  scale factor: {scale:.6f}  "
          f"{'OK' if abs(scale - 1) < 0.02 else 'FAIL (scaling convention)'}")
    print(f"  EVM         : {evm * 100:.3f}%  "
          f"{'OK' if evm <= EVM_TOL else 'FAIL'}")

    fails = (corr < CORR_TOL) + (abs(scale - 1) >= 0.02) + (evm > EVM_TOL)
    print("\nPASS" if fails == 0 else f"\nFAIL ({fails} check(s))")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
