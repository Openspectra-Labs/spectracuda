"""Score cordic_vec against numpy's arctan2.

Judged in LSBs of the phase word, not in radians, because that is the
unit the consumer works in: one LSB is 1/65536 of a turn. A CORDIC with
STAGES stages cannot beat roughly one LSB, so the bar is set by what the
downstream CFO correction needs rather than by exactness.

Degenerate inputs are reported separately: atan2(0, 0) is undefined and
the tiny-magnitude vectors are where quantization dominates, so lumping
them in with the full-circle sweep would hide a real quadrant bug behind
a handful of expected outliers.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Tolerance set by what the CONSUMER needs, not by an arbitrary bar.
# An angle error of e turns in the estimate becomes a per-sample phase
# increment error of e/L (L=128), so over a 4112-sample frame the total
# drift is e*4112/128 turns. At e = 4 LSB = 4/65536 that is 0.0020 turns
# = 0.72 degrees end-to-end -- negligible against QPSK's ~45 degree
# margin. A 16-stage CORDIC's own floor is ~3 LSB (the arctan table
# entries are themselves rounded by up to half an LSB each), so 4 is
# both achievable and comfortably sufficient.
TOL_LSB = 4
ANGLE_W = 16


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    exp = np.loadtxt(os.path.join(here, "build", "cordic_expected.txt"))
    got = np.loadtxt(os.path.join(here, "build", "cordic_out.txt"), ndmin=1)

    if got.size != exp.shape[0]:
        print(f"FAIL: expected {exp.shape[0]} angles, got {got.size}")
        sys.exit(1)

    x, y, want = exp[:, 0], exp[:, 1], exp[:, 2]
    full = 1 << ANGLE_W
    # Circular difference: 0x7FFF and -0x8000 are one LSB apart, not a
    # full turn apart.
    err = (got - want + full // 2) % full - full // 2

    # CORDIC cannot resolve an angle it has no magnitude to work with:
    # once |v| is a few LSBs the shifted terms underflow to zero and the
    # residual angle is meaningless. This is a property of the algorithm,
    # not a bug, and it is why the CFO estimator NORMALIZES P before
    # feeding it in (see cfo_estimate.v). Reported separately so a real
    # quadrant error cannot hide behind these.
    mag = np.hypot(x, y)
    degenerate = mag < 16
    ok = ~degenerate

    worst = int(np.max(np.abs(err[ok])))
    n_bad = int(np.count_nonzero(np.abs(err[ok]) > TOL_LSB))
    print(f"vectors        : {exp.shape[0]} ({int(np.count_nonzero(ok))} "
          f"full-scale, {int(np.count_nonzero(degenerate))} degenerate)")
    print(f"  worst error  : {worst} LSB  ({worst / full * 360:.4f} deg)")
    print(f"  over {TOL_LSB} LSB    : {n_bad}")
    if np.any(degenerate):
        dw = int(np.max(np.abs(err[degenerate])))
        print(f"  degenerate   : worst {dw} LSB (|xy| < 16, informational)")

    if n_bad:
        idx = np.argsort(-np.abs(err))[:5]
        for j in idx:
            print(f"    x={x[j]:.0f} y={y[j]:.0f} want={want[j]:.0f} "
                  f"got={got[j]:.0f} err={err[j]:.0f}")
    print("\nPASS" if n_bad == 0 else "\nFAIL")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
