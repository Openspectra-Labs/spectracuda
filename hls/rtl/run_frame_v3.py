"""v3 RX regression: pinned-Python v3 frames -> rx_top RTL -> payload bytes.

Frames come from golden_ref_v3 (protected header, interleaver2, DMRS,
hard decision). Each case: Python generates the frame, optional CFO and
AWGN are applied ONCE, the same samples feed Python's own receiver and the
RTL. PASS = RTL header fields equal what was sent, RTL bytes equal the
bytes Python's receiver decoded (and the transmitted bytes on a clean
channel), no FIFO overflow, no FD/BD error flags.

    python3 run_frame_v3.py                 # default matrix
    python3 run_frame_v3.py --quick         # 4 cases
"""
import argparse, math, os, subprocess, sys, uuid, json
from fractions import Fraction
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_ref_v3 as g
REF = g.use()
import numpy as np
from spectracuda.sim import Channel

GUARD = 200
# rx_top sources (v3: header_decode_v3 replaces header_decode)
SRCS = ["tb/rx_top_tb.v", "src/rx_top.v", "src/rx_time_domain.v", "src/rx_freq_domain.v",
        "src/sync_fifo_fwft.v", "src/rx_bit_domain.v", "src/il2_deint.v", "src/cdc_async_fifo.v",
        "src/cdc_bundle.v", "src/cdc_reset_sync.v", "src/sc_sync_rtl.v", "src/frame_sync.v",
        "src/cfo_estimate.v", "src/cfo_correct.v", "src/cordic_rot.v", "src/cordic_vec.v",
        "src/cp_fft.v", "src/grid_extract.v", "src/ls_chanest.v", "src/mmse_eq.v", "src/pilot_cpe.v",
        "src/header_decode_v3.v", "src/demapper.v", "src/demapper_soft.v", "src/llr_scale.v", "src/viterbi_dec.v", "src/viterbi_dec_ovl.v",
        "src/viterbi_dec_soft.v", "src/deinterleaver.v", "tb/stubs/xfft_256.v"]

FULL_SCALE = (1 << 15) - 1


def build_and_run(case_dir, rx, bits, cps, bd_mhz, capture=None):
    stim = os.path.join(case_dir, "stim.hex")
    with open(stim, "w") as f:
        for z in rx:
            i = int(np.clip(round(float(np.real(z)) * FULL_SCALE), -32768, 32767))
            q = int(np.clip(round(float(np.imag(z)) * FULL_SCALE), -32768, 32767))
            f.write(f"{((i & 0xFFFF) << 16) | (q & 0xFFFF):08x}\n")
    units = bits // 8
    rows = 1 + math.isqrt(units); cols = -(-units // rows)
    c = Fraction(cps).limit_denominator(16)
    params = os.path.join(case_dir, "params.vh")
    with open(params, "w") as f:
        f.write(f'`define RXT_STIM_PATH "{stim}"\n`define RXT_HDR_PATH "{case_dir}/hdr.txt"\n'
                f'`define RXT_UNIT_PATH "{case_dir}/units.txt"\n`define RXT_NSAMP {len(rx)}\n'
                f"`define RXT_CPS_NUM {c.numerator}\n`define RXT_CPS_DEN {c.denominator}\n"
                f"`define RXT_BD_HALF_PS {int(round(500000.0 / bd_mhz))}\n`define RXT_DRAIN 400000\n"
                f"`define RXT_ENC_BITS {(bits + 6) * 2}\n`define RXT_DI_UNITS {units}\n"
                f"`define RXT_DI_ROWS {rows}\n`define RXT_DI_COLS {cols}\n")
        if capture:
            os.makedirs(capture, exist_ok=True)
            f.write(f'`define RXT_CAPTURE_DIR "{os.path.abspath(capture)}"\n')
    if capture:
        json.dump(dict(cfg_encoded_bits=(bits + 6) * 2, cfg_di_units=units, cfg_di_rows=rows,
                       cfg_di_cols=cols), open(os.path.join(capture, "meta.json"), "w"))
        import shutil; shutil.copy(stim, os.path.join(capture, "stim.hex"))
    src = open(os.path.join(HERE, SRCS[0])).read()
    inc = '`include "build/rxtop_tb_params.vh"'
    assert inc in src
    tb = os.path.join(case_dir, "rx_top_tb.v")
    open(tb, "w").write(src.replace(inc, f'`include "{params}"'))
    srcs = SRCS[1:]
    r = subprocess.run(["verilator", "--binary", "--timing", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC"]
                       + os.environ.get("RXV3_VFLAGS", "").split() + [
                        "-Isrc/generated", "-Isrc", "-I.", "--top-module", "rx_top_tb", "-o", "rxsim",
                        "--Mdir", os.path.join(case_dir, "obj"), tb] + srcs,
                       cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        open(os.path.join(case_dir, "compile.log"), "w").write(r.stdout + r.stderr)
        return None
    sim = subprocess.run([os.path.join(case_dir, "obj", "rxsim")], cwd=HERE, capture_output=True, text=True,
                         check=True, timeout=3600)
    open(os.path.join(case_dir, "sim.log"), "w").write(sim.stdout + sim.stderr)
    hdr = open(os.path.join(case_dir, "hdr.txt")).read().split()
    vals, status = [], {}
    for line in open(os.path.join(case_dir, "units.txt")):
        t = line.split()
        if not t:
            continue
        if t[0] in ("OVF", "FD", "BD"):
            status[t[0]] = [int(x) for x in t[1:]]
        else:
            vals.append(int(t[0]))
    return [int(x) for x in hdr], vals, status


def one(idx, bits, mod, dmrs, cfo=0.0, snr=None, cps=1.0, bd_mhz=125.0, root=None, capture=None):
    o = g.ofdm(mod, dmrs)
    rng = np.random.default_rng(1000 + idx)
    raw = rng.integers(0, 2, size=(1, bits)).astype("uint8")
    tx = np.asarray(o.generate_frame(raw))
    pad = np.zeros((1, GUARD), dtype=tx.dtype)
    rx = np.concatenate([pad, tx, pad], axis=-1)
    if cfo or snr is not None:
        rx = np.asarray(Channel(snr_db=snr, cfo=cfo if cfo else None,
                                cfo_fft_size=256 if cfo else None, seed=idx).process(rx))
    # Python's 4-bit soft values, grabbed from demodulate_soft (bytes
    # 128 + 127*q/7 -> q in -7..7, > 0 means bit 1), rows = payload symbols
    from spectracuda.modem import Modem
    grabbed = []
    real_soft = Modem.demodulate_soft
    def spy(self, *a, **k):
        out = real_soft(self, *a, **k)
        grabbed.append(np.asarray(out))
        return out
    Modem.demodulate_soft = spy
    try:
        res = o.rx_process(rx)
    finally:
        Modem.demodulate_soft = real_soft
    py_q = None
    if grabbed:
        py_q = np.rint((grabbed[-1].astype(np.int64) - 128) * 7 / 127).astype(np.int64)
    py_bytes = None
    if res.get("frame_found"):
        py_bytes = [int(x) for x in np.packbits(np.asarray(res["bits"]).ravel()[:bits].astype(np.uint8))]
    d = os.path.join(root, f"case{idx}")
    os.makedirs(d)
    cap_dir = capture or os.path.join(d, "cap")
    out = build_and_run(d, rx[0], bits, cps, bd_mhz, cap_dir)
    if out is None:
        return False, "compile failed"
    hdr, vals, status = out
    if capture:
        open(os.path.join(capture, "o1.txt"), "w").write("".join(f"{v}\n" for v in vals))
    sent = [int(x) for x in np.packbits(raw[0])]
    ok_hdr = len(hdr) >= 2 and hdr[0] == bits and hdr[1] == g.MOD_CODE[mod]
    ok_py = py_bytes is not None and vals == py_bytes
    ok_sent = vals == sent
    ok_stat = status.get("OVF") == [0] and status.get("FD", [1])[0] == 0 and status.get("BD", [1])[0] == 0
    # ---- soft values: RTL (FD -> BD capture, > 0 means bit 0) vs Python ----
    llr_msg, ok_llr = "llr: no python soft values", False
    if py_q is not None:
        rtl = []
        for line in open(os.path.join(cap_dir, "i2.txt")):
            t = [int(x) for x in line.split()]
            if t[3] == 3:                                   # DATA items, in (sym, sc) order
                rtl.extend(-v for v in t[5:5 + t[4]])       # -> Python's sign convention
        rtl = np.array(rtl, dtype=np.int64)
        pyv = py_q.reshape(-1)
        n = min(len(rtl), len(pyv))
        diff = np.abs(rtl[:n] - pyv[:n])
        # Acceptance (not identity): Python's front end is floating point and
        # the RTL's is fixed point, so under noise the equalized symbols differ
        # slightly (measured ~2% in post-equalization noise power, ~0.1 dB) and
        # a soft value can land one level away. What must hold: the DECODED
        # bytes equal Python's soft decode (checked above, ok_py), >= 99.5% of
        # soft values within one level, and the same overall scale (+/-5%).
        same = float(np.mean(diff == 0)) if n else 0.0
        near = float(np.mean(diff <= 1)) if n else 0.0
        unsat = (np.abs(pyv[:n]) > 0) & (np.abs(pyv[:n]) < 7)
        ratio = float(np.mean(np.abs(rtl[:n][unsat])) / np.mean(np.abs(pyv[:n][unsat]))) if unsat.any() else 1.0
        ok_llr = n == len(pyv) == len(rtl) and near >= 0.995 and abs(ratio - 1.0) <= 0.05
        llr_msg = (f"llr {n}/{len(pyv)} identical {100*same:.2f}% within1 {100*near:.3f}% "
                   f"maxdiff {int(diff.max()) if n else -1} scale {ratio:.3f}")
    clean = not cfo and snr is None
    ok = ok_hdr and ok_py and ok_stat and ok_llr and (ok_sent or not clean)
    nerr = sum(a != b for a, b in zip(vals, sent)) + abs(len(vals) - len(sent))
    return ok, (f"hdr={hdr[:2]} bytes {len(vals)}/{len(sent)} vs_python={'=' if ok_py else '!='} {llr_msg} "
                f"vs_sent_err={nerr} status={status}")


DEFAULT = [  # bits, mod, dmrs, cfo, snr, cps
    (64, "qpsk", 0, 0.0, None, 1.0),
    (2000, "qam16", 16, 0.0, None, 1.0),
    (16384, "qam64", 16, 0.0, None, 1.0),
    (8000, "qpsk", 32, 0.0, None, 1.0),
    (24000, "qam16", 64, 0.0, None, 1.0),
    (16384, "qam64", 16, 0.003, 30.0, 1.0),
    (8000, "qam16", 16, -0.002, 25.0, 2.5),
    (4000, "qpsk", 16, 0.001, 15.0, 10.0),
    (30000, "qam64", 32, 0.0, 35.0, 1.0),
    (512, "qam64", 0, 0.0, None, 5.0),
    # low SNR: soft values spread over -7..7 instead of saturating
    (8000, "qam16", 16, 0.0, 12.0, 1.0),
    (8000, "qpsk", 16, 0.001, 6.0, 2.5),
    (16384, "qam64", 16, 0.0, 20.0, 1.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=5)
    a = ap.parse_args()
    cases = DEFAULT[:4] if a.quick else DEFAULT
    root = os.path.join(HERE, "build", f"frame_v3_{os.getpid()}_{uuid.uuid4().hex[:6]}")
    os.makedirs(root)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(one, i, *c[:3], cfo=c[3], snr=c[4], cps=c[5], root=root) for i, c in enumerate(cases)]
        res = [f.result() for f in futs]
    npass = 0
    for c, (ok, msg) in zip(cases, res):
        npass += ok
        print(f"{'PASS' if ok else 'FAIL'} {c[0]:>6} {c[1]:6s} dmrs={c[2]:<2} cfo={c[3]:<6} snr={c[4]} C={c[5]}: {msg}")
    json.dump(dict(ref=REF, cases=[list(map(str, c)) for c in cases], passed=[r[0] for r in res]),
              open(os.path.join(root, "results.json"), "w"), indent=1)
    print(f"{npass}/{len(cases)} v3 RX cases passed; {os.path.relpath(root, HERE)}")
    raise SystemExit(0 if npass == len(cases) else 1)


if __name__ == "__main__":
    main()
