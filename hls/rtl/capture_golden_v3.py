"""Capture v3 golden_if fixtures (golden_if_v3/) for the FD / BD stage TBs.

Each case runs the full rx_top on pinned-Python v3 frames at C=1 and C=10.
A case is kept only if the RTL PASSES against Python at both rates AND
every dump (stim, i1, i2, c1, o1) is identical between the two rates.
"""
import os, sys, shutil, hashlib, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_frame_v3 as fr
from concurrent.futures import ProcessPoolExecutor

CASES = [  # name, bits, mod, dmrs, cfo, snr
    ("f2000_qam64",               2000, "qam64", 0,  0.0,    None),
    ("f16384_qpsk",              16384, "qpsk", 16,  0.0,    None),
    ("f512_qam16",                 512, "qam16", 0,  0.0,    None),
    ("f64_qpsk",                    64, "qpsk",  0,  0.0,    None),
    ("f2000_qam64_cfo0p3",         2000, "qam64", 0,  0.003,  None),
    ("f16384_qam64_d16_snr30",   16384, "qam64", 16, 0.0,    30.0),
    ("f16384_qam16_d32_cfo_snr25", 16384, "qam16", 32, -0.002, 25.0),
    ("f24000_qam16_d64",         24000, "qam16", 64, 0.0,    None),
]
FILES = ["stim.hex", "i1.txt", "i2.txt", "c1.txt", "o1.txt"]


def run(i, c, cps, root):
    name, bits, mod, dm, cfo, snr = c
    cap = os.path.join(root, name, f"c{int(cps)}")
    ok, msg = fr.one(i, bits, mod, dm, cfo=cfo, snr=snr, cps=cps,
                     root=os.path.join(root, name, f"w{int(cps)}"), capture=cap)
    return ok, msg, cap


def main():
    root = os.path.join(HERE, "build", f"golden_v3_tmp_{os.getpid()}")
    os.makedirs(root)
    dst_root = os.path.join(HERE, "golden_if_v3")
    os.makedirs(dst_root, exist_ok=True)
    with ProcessPoolExecutor(6) as ex:
        futs = {(c[0], cps): ex.submit(run, i, c, cps, root) for i, c in enumerate(CASES) for cps in (1.0, 10.0)}
        res = {k: f.result() for k, f in futs.items()}
    bad = 0
    for c in CASES:
        name = c[0]
        (o1, m1, c1d), (o10, m10, c10d) = res[(name, 1.0)], res[(name, 10.0)]
        if not (o1 and o10):
            print(f"{name}: NOT a PASS vs Python ({m1} | {m10})"); bad += 1; continue
        diff = [f for f in FILES if open(os.path.join(c1d, f), "rb").read() != open(os.path.join(c10d, f), "rb").read()]
        if diff:
            print(f"{name}: differs between C=1 and C=10: {diff}"); bad += 1; continue
        dst = os.path.join(dst_root, name)
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(c1d, dst)
        meta = json.load(open(os.path.join(dst, "meta.json")))
        meta.update(bits=c[1], modem=c[2], dmrs_interval=c[3], cfo=c[4], snr=c[5],
                    python_ref=fr.g.REF_COMMIT, rtl_matches_python=True)
        json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=1)
        with open(os.path.join(dst, "SHA256SUMS"), "w") as f:
            for fn in FILES:
                f.write(f"{hashlib.sha256(open(os.path.join(dst, fn), 'rb').read()).hexdigest()}  {fn}\n")
        n1 = sum(1 for _ in open(os.path.join(dst, "i1.txt"))); n2 = sum(1 for _ in open(os.path.join(dst, "i2.txt")))
        print(f"{name}: OK (PASS vs Python at C=1 and C=10, identical dumps; {n1} bins, {n2} LLR groups)")
    print(f"{len(CASES) - bad}/{len(CASES)} v3 fixtures captured into golden_if_v3/")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
