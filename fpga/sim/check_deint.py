"""Score the de-interleaver bit-exactly against BlockInterleaver."""
from __future__ import annotations
import os, sys
import numpy as np


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fpga/
    b = os.path.join(here, "build")
    ref = np.loadtxt(os.path.join(b, "deint_ref.txt"), dtype=np.int64, ndmin=1)
    got = np.loadtxt(os.path.join(b, "deint_out.txt"), dtype=np.int64, ndmin=1)
    if got.size != ref.size:
        print(f"  FAIL: got {got.size} units, expected {ref.size}")
        sys.exit(1)
    bad = int(np.count_nonzero(got != ref))
    print(f"  units={ref.size}  mismatches={bad}  {'PASS' if bad == 0 else 'FAIL'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
