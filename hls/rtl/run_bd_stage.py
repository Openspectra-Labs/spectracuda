"""Standalone tests of rx_bit_domain against the golden_if/ reference dumps.

Input is each case's i2.txt (what FD handed the bit domain in the
pre-refactor RTL); expected outputs are its o1.txt bytes and its c1.txt
config. Builds tb/rx_bit_domain_tb.v once; every scenario is a run.

    python run_bd_stage.py          # all scenarios
    python run_bd_stage.py --quick  # normal pacing, C=1 and C=10

Scenarios:
  normal     FD-like pacing (a symbol's groups one per C clocks)
  burst      groups back to back: worst case for the coded-bit FIFO
  twoframes  the same frame twice, back to back, fseq 0 then 1: the second
             must not inherit anything from the first (H8: one host
             geometry per run, so both frames are the same case)
  err        header bits inverted: the header must be rejected (cfg_err)
             and no bytes may come out
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "golden_if")
N_DATA = 216
SRCS = ["tb/rx_bit_domain_tb.v", "src/rx_bit_domain.v", "src/sync_fifo_fwft.v",
        "src/header_decode.v", "src/viterbi_dec.v", "src/viterbi_dec_ovl.v", "src/deinterleaver.v"]


def rows(path):
    with open(path) as f:
        return [list(map(int, l.split())) for l in f if l.strip()]


def frame(case, fseq):
    d = os.path.join(GOLD, case)
    i2 = [[fseq] + r[1:] for r in rows(os.path.join(d, "i2.txt"))]
    last_data = max(i for i, r in enumerate(i2) if r[3] == 3)
    items = []
    for i, r in enumerate(i2):
        sym, sc, st = r[1], r[2], r[3]
        items.append(r + [int(sc == 0), int(sc == N_DATA - 1),
                          int(st == 1 and sym == 1 and sc == 0), int(i == last_data)])
    o1 = [int(l) for l in open(os.path.join(d, "o1.txt")) if l.strip()]
    out = [[b, int(k == len(o1) - 1), fseq] for k, b in enumerate(o1)]
    c1 = {k: int(v) for k, v in (l.split() for l in open(os.path.join(d, "c1.txt")) if l.strip())}
    cfg = [fseq, 1, 0, c1["cfg_mod"], c1["cfg_body_syms"], c1["cfg_payload_len_bits"],
           c1["cfg_fec0"], c1["cfg_fec1"], c1["cfg_crc"]]
    meta = json.load(open(os.path.join(d, "meta.json")))
    return items, out, cfg, meta


def write(path, rr):
    with open(path, "w") as f:
        f.write("".join(" ".join(map(str, r)) + "\n" for r in rr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    wdir = os.path.join(HERE, "build", f"bd_stage_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    os.makedirs(wdir)
    r = subprocess.run(["verilator", "--binary", "--timing", "-Wno-WIDTHEXPAND",
                        "-Wno-WIDTHTRUNC", "-Isrc/generated", "-Isrc", "-I.",
                        "--top-module", "rx_bit_domain_tb", "-o", "bd_tb",
                        "--Mdir", os.path.join(wdir, "vsim")] + SRCS,
                       cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit("verilator build failed")
    binary = os.path.join(wdir, "vsim", "bd_tb")

    cases = sorted(d for d in os.listdir(GOLD) if d.startswith("f"))
    tests = []
    for case in cases:
        for c in ([1, 10] if a.quick else [1, 2.5, 10]):
            tests.append((case, c, "normal"))
        if not a.quick:
            tests.append((case, 1, "burst"))
            tests.append((case, 1, "twoframes"))
            tests.append((case, 10, "twoframes"))
    if not a.quick:
        tests += [("f2000_qam64", 1, "err"), ("f16384_qpsk", 10, "err")]

    fails, worst_cb, worst_lat = 0, 0, 0
    for case, c, mode in tests:
        items, out, cfg, meta = frame(case, 0)
        if mode == "twoframes":
            i2b, o2, c2, _ = frame(case, 1)
            items, out, cfgs = items + i2b, out + o2, [cfg, c2]
        elif mode == "err":
            items = [x for x in items if x[3] == 1]          # FD never sends its DATA
            out, cfgs = [], [[0, 0, 1, 0, 0, 0, 0, 0, 0]]
        else:
            cfgs = [cfg]
        tag = f"{case} C={c} {mode}"
        base = os.path.join(wdir, f"t{abs(hash(tag))}")
        write(base + ".i2", items); write(base + ".o1", out); write(base + ".c1", cfgs)
        num, den = (5, 2) if c == 2.5 else (int(c), 1)
        res = subprocess.run([binary, f"+i2={base}.i2", f"+o1={base}.o1", f"+c1={base}.c1",
                              f"+enc={meta['cfg_encoded_bits']}", f"+units={meta['cfg_di_units']}",
                              f"+rows={meta['cfg_di_rows']}", f"+cols={meta['cfg_di_cols']}",
                              f"+cps_num={num}", f"+cps_den={den}",
                              f"+burst={int(mode == 'burst')}", f"+corrupt={int(mode == 'err')}"],
                             capture_output=True, text=True)
        txt = res.stdout + res.stderr
        ok = "TB_RESULT PASS" in txt
        hw = next((l[4:] for l in txt.splitlines() if "CB high-water" in l), "")
        print(f"{'PASS' if ok else 'FAIL'}  {tag:48s} {hw}")
        import re
        m = re.search(r"high-water (\d+) .* latency .* (\d+) clk", hw)
        if m:
            worst_cb = max(worst_cb, int(m.group(1))); worst_lat = max(worst_lat, int(m.group(2)))
        if not ok:
            fails += 1
            for l in txt.splitlines():
                if l.startswith("TB: ") and "high-water" not in l:
                    print("        " + l[4:])
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    print(f"max coded-bit FIFO occupancy: {worst_cb} of {128*216}; "
          f"max config latency (last header item in -> publish): {worst_lat} clk")
    if fails == 0:
        shutil.rmtree(wdir, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
