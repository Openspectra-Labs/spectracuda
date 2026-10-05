"""Score the CRC block against spectracuda. Bit-exact, no tolerance."""
from __future__ import annotations
import os, sys


def main() -> None:
    b = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
    exp = int(open(os.path.join(b, "crc_expected.txt")).read().strip())
    got = int(open(os.path.join(b, "crc_out.txt")).read().strip())
    ok = exp == got
    print(f"  expect=0x{exp:04x}  rtl=0x{got:04x}  {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
