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

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from spectracuda.fec.fec import FEC                      # noqa: E402
from spectracuda.interleaver.base import _PermutationInterleaverBase  # noqa: E402
from spectracuda.pipeline import Ofdm                    # noqa: E402
from spectracuda.sim import Channel                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
GUARD = 200
FULL_SCALE = (1 << 15) - 1

SRCS = ["tb/rx_top_tb.v", "src/rx_top.v", "src/rx_time_domain.v", "src/rx_freq_domain.v", "src/rx_header.v", "src/rx_bit_decoder.v", "src/sc_sync_rtl.v", "src/frame_sync.v",
        "src/cfo_estimate.v", "src/cfo_correct.v", "src/cordic_rot.v",
        "src/cordic_vec.v", "src/cp_fft.v", "src/grid_extract.v",
        "src/ls_chanest.v", "src/mmse_eq.v", "src/pilot_cpe.v",
        "src/header_decode.v", "src/demapper.v", "src/viterbi_dec.v",
        "src/deinterleaver.v", "tb/stubs/xfft_256.v"]


def capture(ofdm, rx):
    """Run the library's own rx_process, recording the two boundaries the
    RTL is scored at: the Viterbi input and the deinterleaver output."""
    seen = {}
    o_fec, o_int = FEC.decode, _PermutationInterleaverBase.decode

    def fec_hook(self, bits, **kw):
        out = o_fec(self, bits, **kw)
        seen.setdefault(f"fec_in_{self.scheme}", np.array(bits, copy=True))
        seen.setdefault(f"fec_out_{self.scheme}", np.array(out, copy=True))
        return out

    def int_hook(self, bits):
        out = o_int(self, bits)
        seen.setdefault("deint_in", np.array(bits, copy=True))
        seen.setdefault("deint_out", np.array(out, copy=True))
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

    enc = seen["fec_in_conv_v27"].ravel().size
    di_in = seen["deint_in"].ravel().size
    py_units = np.packbits(seen["deint_out"].ravel().astype(np.uint8))
    n_units = di_in // 8
    M = 1 + int(np.floor(np.sqrt(n_units)))
    N = -(-n_units // M)

    os.makedirs(BUILD, exist_ok=True)
    stim = os.path.join(BUILD, "rxtop_stim.hex")
    with open(stim, "w") as f:
        for z in rx[0]:
            i = int(np.clip(round(float(np.real(z)) * FULL_SCALE), -32768, 32767))
            q = int(np.clip(round(float(np.imag(z)) * FULL_SCALE), -32768, 32767))
            f.write(f"{((i & 0xFFFF) << 16) | (q & 0xFFFF):08x}\n")

    hdrp = os.path.join(BUILD, "rxtop_hdr.txt")
    unitp = os.path.join(BUILD, "rxtop_units.txt")
    with open(os.path.join(BUILD, "rxtop_tb_params.vh"), "w") as f:
        f.write("// GENERATED by run_frame.py -- do not edit\n")
        f.write(f'`define RXT_STIM_PATH "{stim}"\n')
        f.write(f'`define RXT_HDR_PATH "{hdrp}"\n')
        f.write(f'`define RXT_UNIT_PATH "{unitp}"\n')
        f.write(f"`define RXT_NSAMP {rx.shape[-1]}\n")
        f.write("`define RXT_DRAIN 200000\n")
        f.write(f"`define RXT_ENC_BITS {enc}\n")
        f.write(f"`define RXT_DI_UNITS {n_units}\n")
        f.write(f"`define RXT_DI_ROWS {M}\n`define RXT_DI_COLS {N}\n")

    mdir = os.path.join(BUILD, "vsim_rxtop")
    cmd = ["verilator", "--binary", "--timing", "-Wno-WIDTHEXPAND",
           "-Wno-WIDTHTRUNC", "-Isrc/generated", "-Isrc", "-I.",
           "--top-module", "rx_top_tb", "-o", "rxtop_vsim",
           "--Mdir", mdir] + SRCS
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit("verilator build failed")
    subprocess.run([os.path.join(mdir, "rxtop_vsim")], cwd=HERE,
                   capture_output=True, text=True, check=True)

    hdr = [int(x) for x in open(hdrp).read().split()] if os.path.getsize(hdrp) else []
    raw = [l.split() for l in open(unitp) if l.strip()]
    ovf = 0
    vals = []
    for r in raw:
        if r[0] == "OVF":
            ovf = int(r[1])
        else:
            vals.append(int(r[0]))
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
    ok_u = n > 0 and bad == 0 and len(rtl_units) == len(py_units) and ovf == 0
    print(f"rtl out: {len(rtl_units)} units vs python {len(py_units)}, "
          f"mismatches={bad}  fifo_overflow={ovf}")
    # Under noise the RTL (fixed point) and Python (float64) can legitimately
    # make DIFFERENT marginal decisions, so report both the divergence and
    # whether each actually recovered the payload.
    py_ok = bool(np.array_equal(np.asarray(res["bits"]).ravel()[:a.bits], bits.ravel()))
    print(f"\nrtl==python : {'YES' if ok_u else 'NO'}   python==tx : {'YES' if py_ok else 'NO'}")
    print(f"VERDICT: {'PASS -- bit-exact' if (ok_h and ok_u) else 'DIVERGED'}")
    sys.exit(0 if (ok_h and ok_u) else 1)


if __name__ == "__main__":
    main()
