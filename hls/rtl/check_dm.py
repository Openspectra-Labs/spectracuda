"""Score the demapper bit-exactly against Modem.demodulate().

No tolerance. A demapper makes an integer decision -- the boundary is
either in the same place as Python's or it is not. Any mismatch is
reported with the symbol that caused it, because the interesting failures
cluster near decision boundaries and the symbol value says immediately
whether that is what happened.
"""
from __future__ import annotations
import os, sys
import numpy as np


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    b = os.path.join(here, "build")
    scheme = sys.argv[1] if len(sys.argv) > 1 else "qpsk"
    exp_path = os.path.normpath(os.path.join(here, "..", "golden",
                                             f"dm_{scheme}", "dm_expected.txt"))
    exp = [l.strip() for l in open(exp_path) if l.strip()]
    got = [l.strip() for l in open(os.path.join(b, "dm_out.txt")) if l.strip()]

    if len(got) < len(exp):
        print(f"FAIL: got {len(got)} symbols, expected {len(exp)}")
        sys.exit(1)
    got = got[:len(exp)]

    bad = [i for i, (a, c) in enumerate(zip(exp, got)) if a != c]
    n_bits = len(exp[0]) * len(exp)
    bit_err = sum(sum(1 for x, y in zip(a, c) if x != y)
                  for a, c in zip(exp, got))
    print(f"scheme        : {scheme}")
    print(f"  symbols     : {len(exp)}")
    print(f"  mismatches  : {len(bad)} symbols, {bit_err} of {n_bits} bits")
    if bad:
        sym = np.loadtxt(os.path.normpath(os.path.join(
            here, "..", "golden", f"dm_{scheme}", "dm_sym.txt")))
        for i in bad[:5]:
            print(f"    sym {i}: ({sym[i,0]:+.4f},{sym[i,1]:+.4f})  "
                  f"python={exp[i]}  rtl={got[i]}")
    print("\nPASS" if not bad else "\nFAIL")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
