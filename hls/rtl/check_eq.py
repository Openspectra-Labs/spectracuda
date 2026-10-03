"""Score mmse_eq against spectracuda's MMSEEqualizer.

Reported overall and for the WEAKEST-CHANNEL subcarriers separately. The
reciprocal's hard case is a deeply faded bin, where the denominator is
small and the ROM's relative error matters most; those bins are a small
minority, so a whole-vector metric would hide a bad reciprocal behind
216 easy ones.
"""
from __future__ import annotations

import os, sys
import numpy as np

EVM_TOL = 0.02
CORR_TOL = 0.999


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    b = os.path.join(here, "build")
    ref = np.loadtxt(os.path.join(b, "eq_ref.txt"))
    got = np.loadtxt(os.path.join(b, "eq_out.txt"), ndmin=2)
    if got.shape[0] < ref.shape[0]:
        print(f"FAIL: got {got.shape[0]} of {ref.shape[0]}"); sys.exit(1)
    got = got[:ref.shape[0]]
    r = ref[:, 0] + 1j * ref[:, 1]
    g = got[:, 0] + 1j * got[:, 1]

    def score(rr, gg, label):
        corr = float(np.abs(np.vdot(rr, gg)) /
                     (np.linalg.norm(rr) * np.linalg.norm(gg) + 1e-30))
        evm = float(np.linalg.norm(gg - rr) / (np.linalg.norm(rr) + 1e-30))
        ok = corr >= CORR_TOL and evm <= EVM_TOL
        print(f"  {label:<26} corr={corr:.6f}  EVM={evm*100:6.3f}%  "
              f"{'OK' if ok else 'FAIL'}")
        return 0 if ok else 1

    print(f"subcarriers : {len(r)}")
    fails = score(r, g, "all")
    # The largest |y| are the deep fades: y = rx/H, so weak H gives big y.
    weak = np.argsort(-np.abs(r))[:max(1, len(r) // 8)]
    fails += score(r[weak], g[weak], f"deepest fades ({len(weak)})")
    print("\nPASS" if fails == 0 else f"\nFAIL ({fails} check(s))")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
