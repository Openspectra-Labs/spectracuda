"""Equivalence of every soft-decoder variant against the wide reference.

Each variant of viterbi_dec_soft.v is built against a reference instance
with PM_W = 24, NORM = 0, OPT = 0 (plain add-compare-select, no
normalization, cannot overflow on any frame here) and fed identical 4-bit
LLR frames: realistic max-log LLRs (QPSK / 16-QAM / 64-QAM over AWGN,
clip 2.5) across SNR, and an adversarial random +/-7 stream that maximises
branch metrics every step. Frames up to 20,000 steps, so normalization
fires many times. Every decoded bit must match.

    python run_viterbi_soft.py
"""
import os, sys, subprocess, shutil, uuid
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_pm_spread as sp

HERE = os.path.dirname(os.path.abspath(__file__))
import os as _os
VARIANTS_ALL = [("wide 22, no norm", 22, 0, 1), ("modulo 10, optimized", 10, 0, 1),
            ("modulo 10, plain ACS", 10, 0, 0), ("window offset 12", 12, 1, 1),
            ("threshold 10", 10, 2, 1)]
_only = _os.environ.get("ONLY")
VARIANTS = [v for v in VARIANTS_ALL if not _only or _only in v[0]]


def make_frames(path, kind, rng, truth):
    lens = [41, 126, 2000, 9000, 20000]
    with open(path, "w") as f:
        for n in lens:
            info = rng.integers(0, 2, n)
            coded = sp.encode(info)
            truth.extend(list(info) + [0] * 6)
            if kind == "adversarial":
                l = rng.choice([-7, 7], len(coded))
            else:
                bps, snr = kind
                l = sp.llrs(coded, bps, snr, 2.5, rng)
            pairs = l.reshape(-1, 2)
            f.write(f"{len(pairs)} " + " ".join(f"{a} {b}" for a, b in pairs) + "\n")


def main():
    wdir = os.path.join(HERE, "build", f"vsoft_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    os.makedirs(wdir)
    rng = np.random.default_rng(5)
    stims = []
    for kind in [(2, 2), (2, 6), (4, 8), (4, 14), (6, 14), (6, 22), "adversarial"]:
        p = os.path.join(wdir, f"s{len(stims)}.txt"); truth = []
        make_frames(p, kind, rng, truth)
        stims.append((kind, p, truth))
    fails = 0
    for name, pmw, norm, opt in VARIANTS:
        b = os.path.join(wdir, f"v{pmw}_{norm}_{opt}")
        r = subprocess.run(["verilator", "--binary", "--timing", "-Wno-WIDTHEXPAND",
                            "-Wno-WIDTHTRUNC", f"-DDUT_PM_W={pmw}", f"-DDUT_NORM={norm}",
                            f"-DDUT_OPT={opt}", "--top-module", "viterbi_soft_tb",
                            "-o", "t", "--Mdir", b, "tb/viterbi_soft_tb.v",
                            "src/viterbi_dec_soft.v"], cwd=HERE, capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:]); raise SystemExit("build failed")
        for kind, p, truth in stims:
            for gap in (0, 50):
                out = subprocess.run([f"{b}/t", f"+sym={p}", f"+gap={gap}", f"+dump={p}.{pmw}{norm}{opt}{gap}"],
                                     capture_output=True, text=True).stdout
                ok = "TB_RESULT PASS" in out
                mm = [l for l in out.splitlines() if "mismatches=" in l]
                rate = [l.split("= ")[-1] for l in out.splitlines() if "bits/clock" in l]
                dec = open(f"{p}.{pmw}{norm}{opt}{gap}.new").read()
                errs = sum(int(c) != t for c, t in zip(dec, truth))
                fails += not ok
                k = "adversarial" if kind == "adversarial" else f"{['','','QPSK','','16QAM','','64QAM'][kind[0]]}@{kind[1]}dB"
                print(f"{'PASS' if ok else 'FAIL'}  {name:22s} {k:12s} {'continuous' if gap == 0 else 'sparse':10s} "
                      f"{mm[0].split(';')[1].split(',')[2].strip() if mm else '?':16s} decoded errors {errs:>5} "
                      f"{('rate ' + rate[0]) if rate else ''}")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
    if not fails:
        shutil.rmtree(wdir, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
