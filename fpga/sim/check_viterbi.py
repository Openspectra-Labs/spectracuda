"""Score viterbi_dec.v against spectracuda's ConvolutionalCode.decode().

The bar is agreement with PYTHON on the SAME received stream, not with
the transmitted bits. At a nonzero error rate Python's decoder gets some
wrong too, and reproducing the golden model is the contract -- beating
it would mean the two disagree, which is a failure here even if the RTL
happened to be closer to the truth.
"""
from __future__ import annotations
import os, sys


def main() -> None:
    b = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    exp = [l.strip() for l in open(os.path.join(b, "viterbi_expected.txt")) if l.strip()]
    got = [l.strip() for l in open(os.path.join(b, "viterbi_out.txt")) if l.strip()]

    n = len(exp)
    print(f"  python bits : {n}")
    print(f"  rtl bits    : {len(got)}")
    if len(got) < n:
        print(f"\nFAIL: rtl produced {len(got)} bits, need {n}")
        sys.exit(1)
    got = got[:n]

    bad = [i for i, (a, c) in enumerate(zip(exp, got)) if a != c]
    print(f"  mismatches  : {len(bad)} of {n}")
    if bad:
        print(f"  first at    : {bad[:10]}")
    print("\nPASS" if not bad else "\nFAIL")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
