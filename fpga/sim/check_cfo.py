"""Score the CFO pair against spectracuda.

Two separate judgements, because the block has two separable jobs and a
failure in one should not be diagnosed through the other:

  angle   estimator only -- compared in LSBs of the phase word
  samples derotator + estimator together -- compared as vector error
          relative to signal magnitude, i.e. as added EVM

EVM is the honest unit for the second one. The derotator's consumer is
the FFT and then a QPSK demapper, and what they care about is how far
each sample moved, relative to how big it is -- not whether it matches
float bit-for-bit, which a 16-bit fixed-point CORDIC cannot do and does
not need to.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ANGLE_TOL_LSB = 4      # see check_cordic.py for the derivation
EVM_TOL = 0.02         # 2% -> -34 dB, well under QPSK's ~-14 dB need


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fpga/
    b = os.path.join(here, "build")
    with open(os.path.join(b, "cfo_expected.json")) as f:
        exp = json.load(f)

    ang = np.loadtxt(os.path.join(b, "cfo_angle.txt"), ndmin=2)
    got_angle = int(ang[0, 0])
    want_angle = int(exp["expected_angle_code"])
    full = 1 << 16
    aerr = (got_angle - want_angle + full // 2) % full - full // 2

    print(f"case: {exp['case']}")
    print(f"  angle  : rtl={got_angle}  python={want_angle}  "
          f"err={aerr} LSB  "
          f"{'OK' if abs(aerr) <= ANGLE_TOL_LSB else 'FAIL'}")
    # Report the cfo in spectracuda's own units too, since that is the
    # number a person recognises.
    got_cfo = 2.0 * got_angle / full
    print(f"           rtl cfo={got_cfo:+.6f}  "
          f"python cfo={exp['expected_cfo']:+.6f}")

    ref = np.loadtxt(os.path.join(b, "cfo_ref_corrected.txt"))
    got = np.loadtxt(os.path.join(b, "cfo_out.txt"), ndmin=2)
    n = min(len(ref), len(got))
    ref, got = ref[:n], got[:n]

    err = np.hypot(got[:, 0] - ref[:, 0], got[:, 1] - ref[:, 1])
    mag = np.hypot(ref[:, 0], ref[:, 1])
    # Frame-wide EVM, not per-sample: the guard samples are near-zero and
    # a per-sample ratio there is dominated by quantization of nothing.
    evm = float(np.sqrt(np.mean(err ** 2) / np.mean(mag ** 2)))
    print(f"  samples: {n} compared")
    print(f"           EVM vs python = {evm * 100:.3f}%  "
          f"({20 * np.log10(max(evm, 1e-12)):.1f} dB)  "
          f"{'OK' if evm <= EVM_TOL else 'FAIL'}")

    fails = (abs(aerr) > ANGLE_TOL_LSB) + (evm > EVM_TOL)
    print("\nPASS" if fails == 0 else f"\nFAIL ({fails} check(s))")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
