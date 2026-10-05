"""Equivalence + throughput of viterbi_dec_ovl against viterbi_dec.

Random frames are convolutionally encoded (K=7, 0o171/0o133, zero tail --
viterbi_dec.v's trellis convention), corrupted at several bit-error rates
(the start-state difference between the two decoders can only matter when
survivors have NOT merged, i.e. under errors), and fed to both decoders.
Every decoded bit must match. Continuous input also measures bits/clock.

    python run_viterbi_ovl.py
"""
import os, random, subprocess, sys, uuid, shutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fpga/


def encode(bits):
    st, out = 0, []
    for b in list(bits) + [0] * 6:
        reg = ((st << 1) | b) & 0x7F
        out.append(bin(reg & 0o171).count("1") & 1 | ((bin(reg & 0o133).count("1") & 1) << 1))
        st = reg & 63
    return out


def frames(seed, ber, lens, truth=None):
    rnd = random.Random(seed)
    fr = []
    for n in lens:
        info = [rnd.getrandbits(1) for _ in range(n)]
        if truth is not None:
            truth.extend(info + [0] * 6)          # the decoder emits the tail too
        s = encode(info)
        s = [x ^ (1 if rnd.random() < ber else 0) ^ (2 if rnd.random() < ber else 0) for x in s]
        fr.append(s)
    return fr


def main():
    wdir = os.path.join(HERE, "build", f"vovl_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    os.makedirs(wdir)
    r = subprocess.run(["verilator", "--binary", "--timing", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC",
                        "--top-module", "viterbi_ovl_tb", "-o", "vtb", "--Mdir", f"{wdir}/vsim",
                        "tb/rx/viterbi_ovl_tb.v", "rtl/common/viterbi_dec.v", "rtl/common/viterbi_dec_ovl.v"],
                       cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stdout + r.stderr); raise SystemExit("build failed")
    lens = [40, 85, 126, 1000, 2000, 5068, 13000, 16390]   # incl. edge cases near 42/84
    fails = 0
    for ber in (0.0, 0.01, 0.03, 0.06, 0.10):
        for gap in (0, 60):
            truth = []
            fr = frames(int(ber * 1000) + 7, ber, lens, truth)
            path = f"{wdir}/s_{ber}_{gap}.txt"
            with open(path, "w") as f:
                for s in fr:
                    f.write(f"{len(s)} " + " ".join(map(str, s)) + "\n")
            out = subprocess.run([f"{wdir}/vsim/vtb", f"+sym={path}", f"+gap={gap}",
                                  f"+dump={path}"], capture_output=True, text=True).stdout
            ok = "TB_RESULT PASS" in out
            # decoded error rate of each decoder against what was sent
            old = open(path + ".old").read(); new = open(path + ".new").read()
            eo = sum(int(c) != t for c, t in zip(old, truth))
            en = sum(int(c) != t for c, t in zip(new, truth))
            out += f"TB: errors vs transmitted bits: old={eo} new={en} (of {len(truth)})\n"
            fails += not ok
            info = [l[4:] for l in out.splitlines() if l.startswith("TB: ")]
            print(f"{'PASS' if ok else 'FAIL'}  BER={ber:<5} input={'continuous' if gap == 0 else 'sparse'}  "
                  + " | ".join(info))
    if not fails:
        shutil.rmtree(wdir, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
