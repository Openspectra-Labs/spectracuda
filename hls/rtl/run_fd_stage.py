"""Standalone tests of rx_freq_domain against the golden_if/ reference dumps.

Builds tb/rx_freq_domain_tb.v ONCE, then runs every scenario as a separate
simulation with plusargs. Input is each case's i1.txt (what TD fed FD in the
pre-refactor RTL); the expected output is its i2.txt plus the frame/symbol
markers the frozen interface defines. No Python reference is involved: this
checks that the new stage reproduces the old one bit-exactly.

    python run_fd_stage.py            # all scenarios
    python run_fd_stage.py --quick    # normal mode only, C=1 and C=10

Scenarios (docs/rx_modular_architecture.md section 10):
  normal     config 150 clocks after the header is out
  stall      + random out_ready stalls
  delayed    config ~2.5 symbols late, so BODY piles up in B1
  wrongfseq  a config for another frame first (must be ignored), then the right one
  err        cfg_err for the frame: BODY dropped, B1 must drain to empty
  twoframes  two frames back to back, the first one's config late, so the
             second frame queues behind the first in B1 (no inheritance)
  wrap5      five frames, fseq 0,1,2,3,0: a LEGAL wrap -- no collision flag
  collision  NEGATIVE: two consecutive frames both fseq 0, the first still
             resident -- st_fseq_collision must fire
  notrain    NEGATIVE: a frame without its training symbol -- the header can
             never get its H; st_hdr_no_train must fire
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "golden_if")
N_DATA = 216
SRCS = ["tb/rx_freq_domain_tb.v", "src/rx_freq_domain.v", "src/sync_fifo_fwft.v",
        "src/grid_extract.v", "src/ls_chanest.v", "src/mmse_eq.v", "src/pilot_cpe.v",
        "src/demapper.v", "src/cordic_rot.v", "src/cordic_vec.v"]


def read_rows(path):
    with open(path) as f:
        return [list(map(int, l.split())) for l in f if l.strip()]


def read_c1(path):
    with open(path) as f:
        return {k: int(v) for k, v in (l.split() for l in f if l.strip())}


def frame(case, fseq):
    """(stim rows, expected rows, c1) for one golden case, re-tagged with fseq."""
    d = os.path.join(GOLD, case)
    stim = [[fseq] + r[1:] for r in read_rows(os.path.join(d, "i1.txt"))]
    i2 = [[fseq] + r[1:] for r in read_rows(os.path.join(d, "i2.txt"))]
    last_data = max((i for i, r in enumerate(i2) if r[3] == 3), default=-1)
    exp = []
    for i, r in enumerate(i2):
        _, sym, sc, st = r[0], r[1], r[2], r[3]
        ss, se = int(sc == 0), int(sc == N_DATA - 1)
        fs = int(st == 1 and sym == 1 and sc == 0)
        fe = int(i == last_data)
        exp.append(r + [ss, se, fs, fe])
    return stim, exp, read_c1(os.path.join(d, "c1.txt"))


def cfg_line(trig, delay, valid, err, fseq, c1):
    return [trig, delay, valid, err, fseq, c1["cfg_mod"], c1["cfg_body_syms"], 0, 0]


def write(path, rows):
    with open(path, "w") as f:
        f.write("".join(" ".join(map(str, r)) + "\n" for r in rows))


def build(bdir):
    cmd = ["verilator", "--binary", "--timing", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC",
           "-Isrc/generated", "-Isrc", "-I.", "--top-module", "rx_freq_domain_tb",
           "-o", "fd_tb", "--Mdir", bdir] + SRCS
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit("verilator build failed")
    return os.path.join(bdir, "fd_tb")


def run(binary, wdir, name, stim, exp, cfg, cps, stall, gap=64, expect=0):
    sp, ep, cp = (os.path.join(wdir, f"{name}.{x}") for x in ("stim", "exp", "cfg"))
    write(sp, stim); write(ep, exp); write(cp, cfg)
    num, den = (5, 2) if cps == 2.5 else (int(cps), 1)
    r = subprocess.run([binary, f"+stim={sp}", f"+exp={ep}", f"+cfg={cp}",
                        f"+cps_num={num}", f"+cps_den={den}", f"+stall={stall}",
                        f"+gap={gap}", f"+expect={expect}"], capture_output=True, text=True)
    out = r.stdout + r.stderr
    ok = "TB_RESULT PASS" in out
    info = [l[4:] for l in out.splitlines() if l.startswith("TB: ") and
            ("high-water" in l or "MISMATCH" in l or "STABILITY" in l or "latency" in l or
             "EXTRA" in l or "flags" in l or "outputs" in l)]
    return ok, info, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    wdir = os.path.join(HERE, "build", f"fd_stage_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    os.makedirs(wdir)
    binary = build(os.path.join(wdir, "vsim"))
    cases = sorted(d for d in os.listdir(GOLD) if d.startswith("f"))

    tests = []
    for case in cases:
        rates = [1, 10] if a.quick else [1, 2.5, 10]
        for c in rates:
            tests.append((case, c, "normal"))
        if a.quick:
            continue
        for c in (1, 10):
            for mode in ("stall", "delayed", "wrongfseq", "err"):
                tests.append((case, c, mode))
    if not a.quick:
        for pair in (("f64_qpsk", "f2000_qam64"), ("f2000_qam64_cfo0p3", "f512_qam16")):
            for c in (1, 10):
                tests.append((pair, c, "twoframes"))
        for c in (1, 10):
            tests.append(("f64_qpsk", c, "wrap5"))
            tests.append(("f64_qpsk", c, "collision"))
            tests.append(("f64_qpsk", c, "notrain"))

    fails = 0
    worst_b1 = worst_b2 = 0
    hq_seen = 0
    for case, c, mode in tests:
        stall = {1: 20, 2.5: 30, 10: 50}[c] if mode in ("stall", "twoframes", "wrap5") else 0
        expect = {"collision": 1, "notrain": 2}.get(mode, 0)
        if mode == "wrap5":
            stim, exp, cfg = [], [], []
            for i, fq in enumerate((0, 1, 2, 3, 0)):
                s_, e_, c_ = frame(case, fq)
                stim += s_; exp += e_
                cfg.append(cfg_line(i, 150, 1, 0, fq, c_))
            name = f"{case}x5"
        elif mode == "collision":
            s0, e0, c0 = frame(case, 0)
            s1, e1, c1_ = frame("f2000_qam64", 0)
            stim, exp = s0 + s1, e0 + e1
            cfg = [cfg_line(0, int(2.5 * 288 * c), 1, 0, 0, c0)]
            name = f"{case}+f2000_qam64 same-fseq"
        elif mode == "notrain":
            stim, exp, c1 = frame(case, 0)
            stim = [r for r in stim if r[1] != 0]          # drop the training symbol
            cfg = [cfg_line(0, 150, 1, 0, 0, c1)]
            name = f"{case} no-training"
        elif mode == "twoframes":
            s0, e0, c0 = frame(case[0], 0)
            s1, e1, c1 = frame(case[1], 1)
            late = int(2.5 * 288 * c)
            stim, exp = s0 + s1, e0 + e1
            cfg = [cfg_line(0, late, 1, 0, 0, c0), cfg_line(1, 150, 1, 0, 1, c1)]
            name = f"{case[0]}+{case[1]}"
        else:
            stim, exp, c1 = frame(case, 0)
            name = case
            if mode in ("normal", "stall"):
                cfg = [cfg_line(0, 150, 1, 0, 0, c1)]
            elif mode == "delayed":
                cfg = [cfg_line(0, int(2.5 * 288 * c), 1, 0, 0, c1)]
            elif mode == "wrongfseq":
                decoy = dict(c1, cfg_body_syms=1, cfg_mod=1)
                cfg = [cfg_line(0, 50, 1, 0, 1, decoy),
                       cfg_line(0, int(1.5 * 288 * c), 1, 0, 0, c1)]
            elif mode == "err":
                cfg = [cfg_line(0, 150, 0, 1, 0, c1)]
                exp = [r for r in exp if r[3] == 1]           # header only
        tag = f"{name} C={c} {mode}" + (f" stall={stall}%" if stall else "")
        ok, info, out = run(binary, wdir, f"t{len(tag)}_{abs(hash(tag))}", stim, exp, cfg,
                            c, stall, expect=expect)
        hw = next((i for i in info if "high-water" in i), "")
        if expect:
            hw = next((l[4:] for l in out.splitlines() if "negative test" in l), "")
        else:
            import re
            m = re.search(r"B1 high-water (\d+).*B2 high-water (\d+).*hq_held=(\d)", hw)
            if m:
                worst_b1 = max(worst_b1, int(m.group(1)))
                worst_b2 = max(worst_b2, int(m.group(2)))
                hq_seen += int(m.group(3))
        print(f"{'PASS' if ok else 'FAIL'}  {tag:60s} {hw}")
        lat = next((i for i in info if "latency" in i), "")
        if lat and mode == "normal":
            print(f"        {lat}")
        if not ok:
            fails += 1
            for i in info:
                print(f"        {i}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    print(f"max B1 occupancy over all positive tests: {worst_b1} of 1280 "
          f"({100.0 * worst_b1 / 1280:.0f}%); max B2: {worst_b2} of 512; "
          f"header queue actually held a header in {hq_seen} test(s)")
    if not a.keep and fails == 0:
        shutil.rmtree(wdir, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
