"""Push one Python-generated frame through rx_top.v and score the result.

Uses an FPGA-ONLY config: no rs_m8, no crc16. Those run on the host
(rundown.md section 2), so including them would score the RTL against a
chain it does not implement. Dropping them also makes the decode order
    fec1(conv_v27) -> deinterleave -> fec0(none) -> crc(none)
which is exactly what rx_top does, rather than the two-stage order the
full link uses.

Usage:
    python run_frame.py [--bits N] [--cfo F] [--modem qam64]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

# The pinned golden model, NOT the working-tree spectracuda -- see golden_ref.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_ref  # noqa: E402
golden_ref.use()

from spectracuda.fec.fec import FEC                      # noqa: E402
from spectracuda.interleaver.base import _PermutationInterleaverBase  # noqa: E402
from spectracuda.pipeline import Ofdm                    # noqa: E402
from spectracuda.sim import Channel                      # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fpga/
BUILD = os.path.join(HERE, "build")
GUARD = 200
FULL_SCALE = (1 << 15) - 1

SRCS = ["tb/rx/rx_top_tb.v", "rtl/rx/rx_top.v", "rtl/rx/rx_time_domain.v", "rtl/rx/rx_freq_domain.v", "rtl/common/sync_fifo_fwft.v", "rtl/rx/rx_bit_domain.v", "rtl/rx/il2_deint.v", "rtl/common/cdc_async_fifo.v", "rtl/common/cdc_bundle.v", "rtl/common/cdc_reset_sync.v", "rtl/rx/sc_sync_rtl.v", "rtl/rx/frame_sync.v",
        "rtl/rx/cfo_estimate.v", "rtl/rx/cfo_correct.v", "rtl/rx/cordic_rot.v",
        "rtl/rx/cordic_vec.v", "rtl/rx/cp_fft.v", "rtl/rx/grid_extract.v",
        "rtl/rx/ls_chanest.v", "rtl/rx/mmse_eq.v", "rtl/rx/pilot_cpe.v",
        "rtl/rx/header_decode.v", "rtl/rx/demapper.v", "rtl/rx/demapper_soft.v", "rtl/rx/llr_weight.v", "rtl/common/viterbi_dec.v", "rtl/common/viterbi_dec_ovl.v", "rtl/common/viterbi_dec_soft.v",
        "rtl/rx/deinterleaver.v", "tb/rx/stubs/xfft_256.v"]


def capture(ofdm, rx):
    """Run the library's own rx_process, recording the two boundaries the
    RTL is scored at: the Viterbi input and the deinterleaver output."""
    seen = {}
    o_fec, o_int = FEC.decode, _PermutationInterleaverBase.decode

    # Every call is RECORDED, not just the first. With a coded header
    # (spectracuda e374fdf and later) conv_v27 runs for the header too,
    # and "first call wins" silently took the header's 268 bits as the
    # payload geometry. single() below refuses to guess.
    def fec_hook(self, bits, **kw):
        out = o_fec(self, bits, **kw)
        seen.setdefault(f"fec_in_{self.scheme}", []).append(np.array(bits, copy=True))
        seen.setdefault(f"fec_out_{self.scheme}", []).append(np.array(out, copy=True))
        return out

    def int_hook(self, bits):
        out = o_int(self, bits)
        seen.setdefault("deint_in", []).append(np.array(bits, copy=True))
        seen.setdefault("deint_out", []).append(np.array(out, copy=True))
        return out

    FEC.decode, _PermutationInterleaverBase.decode = fec_hook, int_hook
    try:
        res = ofdm.rx_process(rx)
    finally:
        FEC.decode, _PermutationInterleaverBase.decode = o_fec, o_int
    return res, seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=64)
    ap.add_argument("--cfo", type=float, default=0.0)
    ap.add_argument("--capture", default=None,
                    help="write reference dumps (i1/i2/c1/o1 + stimulus + meta) "
                         "to this directory -- see tb/rx/capture_taps.vh")
    ap.add_argument("--dump-i1", default=None,
                    help="write the FFT output (I1) bins to this file")
    ap.add_argument("--bd-mhz", type=float, default=125.0,
                    help="bit-domain clock (TD/FD run at 100 MHz)")
    ap.add_argument("--cps", type=float, default=1.0,
                    help="clocks per input sample (C): 1 = stress, 10 = 10 Msps "
                         "@ 100 MHz, 2.5 = 40 Msps @ 100 MHz")
    ap.add_argument("--modem", default="qam64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evm", type=float, default=0.0,
                    help="target EVM (0.2 = 20%%), split equally in power "
                         "between AWGN and phase noise")
    ap.add_argument("--snr", type=float, default=None,
                    help="AWGN SNR in dB; overrides the --evm split")
    ap.add_argument("--pn-alpha", type=float, default=0.9995,
                    help="AR(1) correlation for the phase noise")
    ap.add_argument("--phase-noise", type=float, default=None,
                    help="phase-noise RMS in radians per sample (Wiener walk)")
    a = ap.parse_args()

    ofdm = Ofdm(
        fft_size=256, cp_len=32,          # cp_fft
        n_data=216, n_pilot=8,            # grid_extract
        sync="schmidl_cox",               # sync_detect
        cfo="schmidl_cox",                # cfo_correct
        channel_estimator="ls",           # chan_est
        equalizer="mmse",                 # equalizer
        modem=a.modem,                    # demapper
        fec1="conv_v27",                  # viterbi
        interleaver="block",              # deinterleaver
        interleaver_kwargs={"unit_bits": 8},
    )

    rng = np.random.default_rng(a.seed)
    bits = rng.integers(0, 2, size=(1, a.bits)).astype("uint8")
    tx = np.asarray(ofdm.generate_frame(bits))
    pad = np.zeros((tx.shape[0], GUARD), dtype=tx.dtype)
    rx = np.concatenate([pad, tx, pad], axis=-1)

    # Impairment budget. EVM from AWGN is 10^(-SNR/20); EVM from a small
    # RMS phase theta is ~theta. Splitting the target equally in power
    # gives each contributor evm/sqrt(2).
    snr_db, pn_rms = a.snr, a.phase_noise
    if a.evm > 0 and snr_db is None and pn_rms is None:
        part = a.evm / np.sqrt(2.0)
        snr_db = -20.0 * np.log10(part)
        pn_rms = part

    if a.cfo or snr_db is not None:
        rx = np.asarray(Channel(snr_db=snr_db,
                                cfo=a.cfo if a.cfo else None,
                                cfo_fft_size=256 if a.cfo else None,
                                seed=a.seed).process(rx))

    # Phase noise as a BOUNDED AR(1) process, steady-state RMS = pn_rms.
    #
    # NOT a Wiener walk: a random walk's variance grows without bound, so
    # a per-sample RMS of 0.035 rad accumulates to ~1 rad across a frame
    # and destroys the header at a nominal "5% EVM". Real oscillator
    # phase noise is bounded and strongly correlated sample to sample,
    # which is also what makes per-symbol CPE correction able to track it.
    #
    # spectracuda has no phase-noise model, so this lives in the harness,
    # not the library. It is applied ONCE and the same impaired samples
    # feed both Python and the RTL, so the comparison stays a like-for-
    # like check of the fixed-point datapath rather than a link budget.
    if pn_rms:
        alpha = a.pn_alpha
        n = rx.shape[-1]
        w = np.random.default_rng(a.seed + 1000).normal(0.0, 1.0, size=n)
        phi = np.empty(n)
        acc = 0.0
        drive = pn_rms * np.sqrt(1.0 - alpha * alpha)
        for i in range(n):
            acc = alpha * acc + drive * w[i]
            phi[i] = acc
        rx = rx * np.exp(1j * phi)[None, :]

    res, seen = capture(ofdm, rx)
    if not res["frame_found"]:
        print(f"PYTHON DID NOT FIND THE FRAME (evm={a.evm}) -- link broken, "
              f"not an RTL failure")
        sys.exit(2)

    def single(key):
        calls = seen.get(key, [])
        if len(calls) != 1:
            raise SystemExit(
                f"run_frame: expected exactly one {key} call (the payload), "
                f"got {len(calls)}. The Python reference is not the frame "
                f"format this RTL implements -- see golden_ref.py.")
        return calls[0]

    enc = single("fec_in_conv_v27").ravel().size
    di_in = single("deint_in").ravel().size
    py_units = np.packbits(single("deint_out").ravel().astype(np.uint8))
    n_units = di_in // 8
    M = 1 + int(np.floor(np.sqrt(n_units)))
    N = -(-n_units // M)

    # PER-PROCESS run directory. Every file this run writes -- stimulus,
    # testbench params, the Verilator build, the RTL's outputs -- lives
    # here, so concurrent runs (a second agent, a parallel sweep) cannot
    # overwrite each other. With shared fixed paths under build/ they did:
    # a 45-run rate matrix lost one run to a half-overwritten Verilator
    # build while another process ran the same script.
    import shutil
    # PID alone is not unique: two sandboxes (separate PID namespaces)
    # both ran as PID 122. Add a random suffix.
    import uuid
    run_dir = os.path.join(BUILD, f"run_frame_{os.getpid()}_{uuid.uuid4().hex[:8]}")
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_dir)
    stim = os.path.join(run_dir, "rxtop_stim.hex")
    with open(stim, "w") as f:
        for z in rx[0]:
            i = int(np.clip(round(float(np.real(z)) * FULL_SCALE), -32768, 32767))
            q = int(np.clip(round(float(np.imag(z)) * FULL_SCALE), -32768, 32767))
            f.write(f"{((i & 0xFFFF) << 16) | (q & 0xFFFF):08x}\n")

    hdrp = os.path.join(run_dir, "rxtop_hdr.txt")
    unitp = os.path.join(run_dir, "rxtop_units.txt")
    params = os.path.join(run_dir, "rxtop_tb_params.vh")
    with open(params, "w") as f:
        f.write("// GENERATED by run_frame.py -- do not edit\n")
        f.write(f'`define RXT_STIM_PATH "{stim}"\n')
        f.write(f'`define RXT_HDR_PATH "{hdrp}"\n')
        f.write(f'`define RXT_UNIT_PATH "{unitp}"\n')
        f.write(f"`define RXT_NSAMP {rx.shape[-1]}\n")
        from fractions import Fraction
        cps = Fraction(a.cps).limit_denominator(16)
        f.write(f"`define RXT_CPS_NUM {cps.numerator}\n`define RXT_CPS_DEN {cps.denominator}\n")
        f.write(f"`define RXT_BD_HALF_PS {int(round(500000.0 / a.bd_mhz))}\n")
        if a.capture:
            os.makedirs(a.capture, exist_ok=True)
            f.write(f'`define RXT_CAPTURE_DIR "{os.path.abspath(a.capture)}"\n')
        if a.dump_i1:
            f.write(f'`define RXT_I1_PATH "{os.path.abspath(a.dump_i1)}"\n')
        f.write("`define RXT_DRAIN 200000\n")
        f.write(f"`define RXT_ENC_BITS {enc}\n")
        f.write(f"`define RXT_DI_UNITS {n_units}\n")
        f.write(f"`define RXT_DI_ROWS {M}\n`define RXT_DI_COLS {N}\n")

    # The testbench includes its params by a fixed relative path; point a
    # private copy at this run's params instead.
    tb_src = open(os.path.join(HERE, SRCS[0])).read()
    tb_inc = '`include "build/rxtop_tb_params.vh"'
    assert tb_inc in tb_src, "rx_top_tb.v params include changed"
    tb_run = os.path.join(run_dir, "rx_top_tb.v")
    with open(tb_run, "w") as f:
        f.write(tb_src.replace(tb_inc, f'`include "{params}"'))

    mdir = os.path.join(run_dir, "vsim")
    cmd = ["verilator", "--binary", "--timing", "-Wno-WIDTHEXPAND",
           "-Wno-WIDTHTRUNC", "-Irtl/generated", "-Irtl", "-Irtl/rx", "-Irtl/common", "-I.",
           "--top-module", "rx_top_tb", "-o", "rxtop_vsim",
           "--Mdir", mdir, tb_run] + SRCS[1:]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit("verilator build failed")
    subprocess.run([os.path.join(mdir, "rxtop_vsim")], cwd=HERE,
                   capture_output=True, text=True, check=True)

    hdr = [int(x) for x in open(hdrp).read().split()] if os.path.getsize(hdrp) else []
    raw = [l.split() for l in open(unitp) if l.strip()]
    ovf = None
    fd_err, fd_b1, fd_b2 = None, None, None
    bd_err, bd_cb = None, None
    vals = []
    for r in raw:
        if r[0] == "OVF":
            ovf = int(r[1])
        elif r[0] == "FD":
            fd_err, fd_b1, fd_b2 = int(r[1]), int(r[2]), int(r[3])
        elif r[0] == "BD":
            bd_err, bd_cb = int(r[1]), int(r[2])
        else:
            vals.append(int(r[0]))
    if ovf is None or fd_err is None or bd_err is None:
        raise SystemExit("run_frame: testbench status lines missing -- cannot judge the run")
    rtl_units = np.array(vals, dtype=int)

    from spectracuda.framing.header import MOD_SCHEME_CODES
    want_hdr = [a.bits, MOD_SCHEME_CODES[a.modem]]

    meas_evm = res.get("evm")
    meas_evm = float(np.asarray(meas_evm).ravel()[0]) if meas_evm is not None else float("nan")
    print(f"config : fft=256 cp=32 n_data=216 n_pilot=8 modem={a.modem} "
          f"fec1=conv_v27 interleaver=block  (no rs, no crc)")
    print(f"channel: target_evm={a.evm}  snr={snr_db}  phase_noise_rms={pn_rms}"
          f"  -> MEASURED evm={meas_evm:.4f}")
    print(f"frame  : {a.bits} payload bits -> {enc} encoded -> "
          f"{rx.shape[-1]} samples, cfo={a.cfo}")
    print(f"python : frame_found={res['frame_found']} "
          f"bits_match={np.array_equal(np.asarray(res['bits']).ravel()[:a.bits], bits.ravel())}")
    if len(hdr) >= 2:
        ok_h = hdr[0] == want_hdr[0] and hdr[1] == want_hdr[1]
        print(f"rtl hdr: len={hdr[0]} mod={hdr[1]}  "
              f"(want len={want_hdr[0]} mod={want_hdr[1]})  "
              f"{'OK' if ok_h else 'MISMATCH'}")
    else:
        ok_h = False
        print("rtl hdr: NONE -- header never decoded")

    n = min(len(rtl_units), len(py_units))
    bad = int(np.sum(rtl_units[:n] != py_units[:n])) if n else -1
    ok_u = (n > 0 and bad == 0 and len(rtl_units) == len(py_units) and ovf == 0
            and fd_err == 0 and bd_err == 0)
    print(f"rtl out: {len(rtl_units)} units vs python {len(py_units)}, "
          f"mismatches={bad}  fifo_overflow={ovf}")
    # fd_err bits: {cfg_unsupported, hdr_no_train, fseq_collision,
    #               seq_err, b2_overflow, b1_overflow}
    print(f"fd     : err=0b{fd_err:06b}  B1 high-water {fd_b1}/1280  "
          f"B2 high-water {fd_b2}/512")
    # bd_err bits: {seq_err, unit_collision}
    print(f"bd     : err=0b{bd_err:02b}  coded-bit FIFO high-water {bd_cb}/27648")
    # Under noise the RTL (fixed point) and Python (float64) can legitimately
    # make DIFFERENT marginal decisions, so report both the divergence and
    # whether each actually recovered the payload.
    py_ok = bool(np.array_equal(np.asarray(res["bits"]).ravel()[:a.bits], bits.ravel()))
    print(f"\nrtl==python : {'YES' if ok_u else 'NO'}   python==tx : {'YES' if py_ok else 'NO'}")
    print(f"VERDICT: {'PASS -- bit-exact' if (ok_h and ok_u) else 'DIVERGED'}")
    if a.capture:
        # Self-contained: everything a stage testbench needs to replay
        # this frame without Python. o1 = the RTL's deinterleaved bytes.
        shutil.copy(stim, os.path.join(a.capture, "stim.hex"))
        with open(os.path.join(a.capture, "o1.txt"), "w") as f:
            f.write("".join(f"{v}\n" for v in rtl_units))
        meta = dict(bits=a.bits, modem=a.modem, cfo=a.cfo, evm=a.evm, snr=a.snr,
                    seed=a.seed, cps=a.cps, nsamp=int(rx.shape[-1]),
                    cfg_encoded_bits=int(enc), cfg_di_units=int(n_units),
                    cfg_di_rows=int(M), cfg_di_cols=int(N),
                    python_ref=golden_ref.REF_COMMIT,
                    rtl_matches_python=bool(ok_h and ok_u),
                    measured_evm=meas_evm)
        with open(os.path.join(a.capture, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)

    # Keep the run directory only when something needs looking at.
    if ok_h and ok_u:
        shutil.rmtree(run_dir, ignore_errors=True)
    else:
        print(f"run files kept in {run_dir}")
    sys.exit(0 if (ok_h and ok_u) else 1)


if __name__ == "__main__":
    main()
