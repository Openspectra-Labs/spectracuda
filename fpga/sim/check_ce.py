"""Score ls_chanest against spectracuda's LSChannelEstimator.

Two numbers, because they fail differently:
  correlation  catches a wrong interpolation table or an off-by-one in
               the bin walk -- either scrambles the frequency response
               while leaving magnitudes plausible
  EVM          the residual once alignment is right, i.e. the fixed-point
               cost of the Q15 weights and the 18-bit estimate

The interpolated span is reported separately. Only 32 of 256 bins are
actually interpolated here (the DC bin and the guard band); the other 224
are 1:1 copies of the LS estimate, so a broken interpolation would barely
move a whole-vector metric. Judging the interpolated bins on their own is
what makes this test able to fail.
"""
from __future__ import annotations

import os
import sys

import numpy as np

EVM_TOL = 0.02
CORR_TOL = 0.999
# Bins the generator reported as interpolated, plus the clamped edges.
INTERP_BINS = [0] + list(range(113, 144))


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fpga/
    b = os.path.join(here, "build")
    ref = np.loadtxt(os.path.join(b, "ce_ref.txt"))
    got = np.loadtxt(os.path.join(b, "ce_out.txt"), ndmin=2)
    if got.shape[0] < ref.shape[0]:
        print(f"FAIL: got {got.shape[0]} bins, expected {ref.shape[0]}")
        sys.exit(1)
    got = got[:ref.shape[0]]

    r = ref[:, 0] + 1j * ref[:, 1]
    g = got[:, 0] + 1j * got[:, 1]

    def score(rr, gg, label):
        corr = float(np.abs(np.vdot(rr, gg)) /
                     (np.linalg.norm(rr) * np.linalg.norm(gg) + 1e-30))
        evm = float(np.linalg.norm(gg - rr) / (np.linalg.norm(rr) + 1e-30))
        ok = (corr >= CORR_TOL) and (evm <= EVM_TOL)
        print(f"  {label:<22} corr={corr:.6f}  EVM={evm*100:6.3f}%  "
              f"{'OK' if ok else 'FAIL'}")
        return 0 if ok else 1

    print(f"bins compared : {len(r)}")
    fails = score(r, g, "all bins")
    idx = [i for i in INTERP_BINS if i < len(r)]
    fails += score(r[idx], g[idx], f"interpolated ({len(idx)})")
    print("\nPASS" if fails == 0 else f"\nFAIL ({fails} check(s))")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
