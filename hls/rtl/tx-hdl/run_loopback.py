"""TX RTL -> RX RTL loopback (v3 format: protected header, IL2, DMRS),
no Python in the signal path.

Random payload bytes go into tx_top (full chain, simulation IFFT model),
the radiated Q15 samples go unchanged (plus zero guards) into rx_top, and the
bytes rx_top delivers must equal the bytes that went in; the RX-decoded
header must carry the same length and modulation. Python is used only to
drive files, never as the reference.

    python3 run_loopback.py
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import subprocess, uuid, math, json
import numpy as np

HERE = Path(__file__).resolve().parent
RTL = HERE.parent
sys.path.insert(0, str(RTL))
import run_frame_v3 as rf  # v3 RX sources + build_and_run (pins golden_ref_v3)

CASES = [(64, "qpsk", 0), (2000, "qam16", 16), (16384, "qam64", 16), (512, "qam64", 0),
         (8, "qpsk", 0), (24000, "qpsk", 16), (30000, "qam64", 32), (20000, "qam16", 64)]
DMRS = {0: 0, 16: 1, 32: 2, 64: 3}
CODE = {"qpsk": 1, "qam16": 2, "qam64": 3}

build = HERE / "build" / f"loopback_{uuid.uuid4().hex[:12]}"
build.mkdir(parents=True)

# ---- TX simulator, built once ----
tx_src = [str(p.relative_to(HERE)) for p in sorted((HERE / "src").glob("*.v"))] + \
         ["tb/tx_xfft_256_model.v", "tb/tx_top_tb.v"]
p = subprocess.run(["verilator", "--binary", "--timing", "--top-module", "tx_top_tb",
                    "--Mdir", str(build / "txobj"), "-o", "tx_top_tb"] + tx_src,
                   cwd=HERE, capture_output=True, text=True)
if p.returncode:
    (build / "tx_compile.log").write_text(p.stdout + p.stderr)
    raise SystemExit(f"TX compile failed {build}")

rx_tb_src = (RTL / rf.SRCS[0]).read_text()
inc = '`include "build/rxtop_tb_params.vh"'
assert inc in rx_tb_src


def one(idx, bits, mod, dm):
    d = build / f"case{idx}"
    d.mkdir()
    payload = np.random.default_rng(1000 + idx).integers(0, 256, bits // 8)
    (d / "input.txt").write_text(f"{bits} {CODE[mod]} 0123456789ab {idx % 4} {len(payload)} {DMRS[dm]}\n" +
                                 "".join(f"{int(b)}\n" for b in payload))
    r = subprocess.run([str(build / "txobj/tx_top_tb"), f"+input={d/'input.txt'}", f"+dump={d/'tx_iq.txt'}"],
                       cwd=HERE, capture_output=True, text=True, timeout=120)
    (d / "tx.log").write_text(r.stdout + r.stderr)
    if r.returncode or "PASS frames=1" not in r.stdout:
        return False, "TX sim failed"
    iq = np.loadtxt(d / "tx_iq.txt", dtype=np.int64).reshape(-1, 2)
    stim = np.concatenate([np.zeros((rf.GUARD, 2), np.int64), iq, np.zeros((rf.GUARD, 2), np.int64)])
    with open(d / "stim.hex", "w") as f:
        for i, q in stim:
            f.write(f"{((int(i) & 0xFFFF) << 16) | (int(q) & 0xFFFF):08x}\n")
    out = rf.build_and_run(str(d), (stim[:, 0] + 1j * stim[:, 1]) / rf.FULL_SCALE, bits, 1.0, 125.0)
    if out is None:
        return False, "RX compile failed"
    hdr, vals, status = out
    ok_hdr = len(hdr) >= 2 and hdr[0] == bits and hdr[1] == CODE[mod]
    ok_bytes = vals == [int(b) for b in payload]
    ok_stat = status.get("OVF") == [0] and status.get("FD", [1])[0] == 0 and status.get("BD", [1])[0] == 0
    ok = ok_hdr and ok_bytes and ok_stat
    return ok, (f"{len(iq)} samples, hdr={hdr[:2]} bytes {len(vals)}/{len(payload)} "
                f"{'exact' if ok_bytes else 'MISMATCH'} status={status}")


results = {}
for idx, (bits, mod, dm) in enumerate(CASES):
    ok, msg = one(idx, bits, mod, dm)
    results[f"{bits}_{mod}_dmrs{dm}"] = ok
    print(f"{'PASS' if ok else 'FAIL'} {bits:>6} bits {mod:6s} dmrs={dm:<2} {msg}")
(build / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print(f"{sum(results.values())}/{len(results)} loopback cases passed; {build.relative_to(HERE)}")
raise SystemExit(0 if all(results.values()) else 1)
