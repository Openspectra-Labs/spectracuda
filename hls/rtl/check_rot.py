"""Score cordic_rot against numpy.

Judged as vector error relative to the input magnitude, which is what a
derotator's consumer cares about: a residual that is small compared to
the signal is indistinguishable from a slightly noisier channel. A
16-stage CORDIC with 16-bit data has an error floor of a few LSBs from
the shifted terms plus the Q15 1/K prescale.

The bar is 0.5% of input magnitude. At that level the added EVM is
~ -46 dB, far below the ~ -14 dB QPSK needs and below the quantization
floor of the 16-bit datapath itself.
"""
from __future__ import annotations

import os
import sys

import numpy as np

TOL_FRAC = 0.005


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    exp = np.loadtxt(os.path.join(here, "build", "rot_expected.txt"))
    got = np.loadtxt(os.path.join(here, "build", "rot_out.txt"), ndmin=2)
    if got.shape[0] != exp.shape[0]:
        print(f"FAIL: expected {exp.shape[0]} results, got {got.shape[0]}")
        sys.exit(1)

    mag = np.hypot(exp[:, 0], exp[:, 1])
    err = np.hypot(got[:, 0] - exp[:, 3], got[:, 1] - exp[:, 4])
    rel = err / mag
    worst = float(np.max(rel))
    n_bad = int(np.count_nonzero(rel > TOL_FRAC))
    print(f"vectors      : {exp.shape[0]} (full turn of z x 8 input angles)")
    print(f"  worst error: {worst * 100:.4f}% of |v|  ({np.max(err):.1f} LSB)")
    print(f"  over {TOL_FRAC * 100:.1f}%  : {n_bad}")
    if n_bad:
        for j in np.argsort(-rel)[:5]:
            print(f"    x={exp[j,0]:.0f} y={exp[j,1]:.0f} z={exp[j,2]:.0f} "
                  f"want=({exp[j,3]:.0f},{exp[j,4]:.0f}) "
                  f"got=({got[j,0]:.0f},{got[j,1]:.0f}) rel={rel[j]*100:.2f}%")
    print("\nPASS" if n_bad == 0 else "\nFAIL")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
