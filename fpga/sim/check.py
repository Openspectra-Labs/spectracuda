"""Score the RTL simulation against spectracuda.

The RTL emits P and R per candidate, deliberately not the metric -- a
divider is expensive in fabric and every consumer of the metric compares
it against something, which is a multiply. So the ratio is formed here,
once, in the same place that reads the golden model's answer. Comparing
through two separate implementations of |P|^2/R^2 would be comparing the
metric implementations, not the detector.

Pass criterion is agreement with PYTHON's start_index, not with the true
frame start. Python's own detector lands a few samples early at low SNR
on some seeds; reproducing the golden model is the contract.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def score(metric_file: str, meta_file: str) -> int:
    rows = np.loadtxt(metric_file)
    if rows.ndim == 1:
        rows = rows[None, :]
    d = rows[:, 0].astype(np.int64)
    p_re, p_im, r_sum = rows[:, 1], rows[:, 2], rows[:, 3]

    # R = (r1 + r2) / 2; the RTL emits r1 + r2 because halving is a shift
    # the consumer can do for free.
    r = r_sum * 0.5
    num = p_re ** 2 + p_im ** 2
    den = r ** 2
    metric = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)

    best = int(d[int(np.argmax(metric))])
    peak = float(np.max(metric))

    with open(meta_file) as f:
        meta = json.load(f)
    want = int(meta["expected_start_index"])
    want_metric = float(meta["expected_metric"])
    true_start = int(meta["expected_true_start"])

    print(f"candidates evaluated : {len(d)}")
    print(f"true frame start     : {true_start}")
    print(f"  start_index : rtl={best}  python={want}  "
          f"{'MATCH' if best == want else 'MISMATCH'}")
    rel = abs(peak - want_metric) / (abs(want_metric) or 1.0)
    print(f"  peak metric : rtl={peak:.6f}  python={want_metric:.6f}  "
          f"rel_err={rel:.3g}")

    fails = 0
    if best != want:
        fails += 1
    # The metric is fed to a threshold of order 0.5, so it is judged at
    # the precision that decision needs, not at float parity.
    if rel > 2e-2:
        fails += 1
        print("  (metric outside the 2% band a threshold decision needs)")
    print("\nPASS" if fails == 0 else f"\nFAIL ({fails} check(s))")
    return fails


def main() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fpga/
    metric = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "build", "rtl_metric.txt")
    meta = sys.argv[2] if len(sys.argv) > 2 else os.path.normpath(
        os.path.join(here, "fixtures", "blocks", "sc_sync_meta.json"))
    sys.exit(score(metric, meta))


if __name__ == "__main__":
    main()
