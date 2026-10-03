"""Separate fixed-point precision from frame-length effects. QAM64 only.

ONE frozen channel realisation is reused across every packet length, so
length is the only variable. That means generating the impairment here
rather than through Channel(): Channel scales its AWGN by the frame's
own power, so calling it per length would give each length a slightly
different realisation and the experiment would prove nothing.

  noise[]  drawn once at max length, sliced per frame
  phi[]    AR(1) phase noise, drawn once at max length, sliced
  sigma    fixed from the reference frame, NOT recomputed per length

Reading the result:
  divergence only as length grows  -> suspect missing SFO/timing-slope
  divergence even on short frames  -> suspect fixed-point precision
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from spectracuda.fec.fec import FEC                                   # noqa: E402
from spectracuda.interleaver.base import _PermutationInterleaverBase  # noqa: E402
from spectracuda.modem.mapper import Modem                            # noqa: E402
from spectracuda.pipeline import Ofdm                                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
GUARD, FULL_SCALE = 200, (1 << 15) - 1
SEED, EVM_TARGET, PN_ALPHA = 0, 0.15, 0.9995
LENGTHS = [256, 1024]

SRCS = ["tb/rx_top_dbg.v", "src/rx_top.v", "src/rx_time_domain.v", "src/rx_freq_domain.v", "src/rx_header.v", "src/rx_bit_decoder.v", "src/sc_sync_rtl.v", "src/frame_sync.v",
        "src/cfo_estimate.v", "src/cfo_correct.v", "src/cordic_rot.v",
        "src/cordic_vec.v", "src/cp_fft.v", "src/grid_extract.v",
        "src/ls_chanest.v", "src/mmse_eq.v", "src/pilot_cpe.v",
        "src/header_decode.v", "src/demapper.v", "src/viterbi_dec.v",
        "src/deinterleaver.v", "tb/stubs/xfft_256.v"]


def build_ofdm():
    return Ofdm(fft_size=256, cp_len=32, n_data=216, n_pilot=8,
                sync="schmidl_cox", cfo="schmidl_cox",
                channel_estimator="ls", equalizer="mmse",
                modem="qam64", fec1="conv_v27",
                interleaver="block", interleaver_kwargs={"unit_bits": 8})


def capture(ofdm, rx):
    """rx_process with the payload constellation and deinterleaver output
    recorded -- the same two boundaries the RTL dumps."""
    seen = {}
    o_int = _PermutationInterleaverBase.decode
    o_dem = Modem.demodulate_stats

    def int_hook(self, bits):
        out = o_int(self, bits)
        seen.setdefault("deint_in", np.array(bits, copy=True))
        seen.setdefault("deint_out", np.array(out, copy=True))
        return out

    def dem_hook(self, syms):
        out = o_dem(self, syms)
        seen.setdefault("constellation", np.array(syms, copy=True))
        return out

    _PermutationInterleaverBase.decode = int_hook
    Modem.demodulate_stats = dem_hook
    try:
        res = ofdm.rx_process(rx)
    finally:
        _PermutationInterleaverBase.decode = o_int
        Modem.demodulate_stats = o_dem
    return res, seen


def main() -> None:
    ofdm = build_ofdm()
    rng = np.random.default_rng(SEED)

    # --- the frozen realisation, sized for the longest frame -----------
    max_bits = max(LENGTHS) * 8
    probe = np.asarray(ofdm.generate_frame(
        rng.integers(0, 2, size=(1, max_bits)).astype("uint8")))
    max_len = probe.shape[-1] + 2 * GUARD

    part = EVM_TARGET / np.sqrt(2.0)
    sig_rms = float(np.sqrt(np.mean(np.abs(probe) ** 2)))
    sigma = part * sig_rms                       # FIXED, not per-length

    nrng = np.random.default_rng(SEED + 500)
    noise = (nrng.normal(0, sigma / np.sqrt(2), max_len)
             + 1j * nrng.normal(0, sigma / np.sqrt(2), max_len))

    wrng = np.random.default_rng(SEED + 1000)
    w = wrng.normal(0.0, 1.0, max_len)
    phi = np.empty(max_len)
    acc, drive = 0.0, part * np.sqrt(1.0 - PN_ALPHA ** 2)
    for i in range(max_len):
        acc = PN_ALPHA * acc + drive * w[i]
        phi[i] = acc

    print(f"frozen realisation: seed={SEED} evm_target={EVM_TARGET} "
          f"sigma={sigma:.5f} pn_alpha={PN_ALPHA}")
    print(f"{'bytes':>6} {'syms':>5} {'meas_EVM':>9} {'py==tx':>7} "
          f"{'rtl==py':>8} {'py_units':>9} {'rtl_units':>10} {'first_diff_bit':>15}")
    print("-" * 82)

    results = []
    for nbytes in LENGTHS:
        nbits = nbytes * 8
        brng = np.random.default_rng(SEED + nbytes)
        bits = brng.integers(0, 2, size=(1, nbits)).astype("uint8")
        tx = np.asarray(ofdm.generate_frame(bits))
        pad = np.zeros((tx.shape[0], GUARD), dtype=tx.dtype)
        rx = np.concatenate([pad, tx, pad], axis=-1)
        n = rx.shape[-1]
        rx = (rx + noise[None, :n]) * np.exp(1j * phi[None, :n])

        res, seen = capture(ofdm, rx)
        if not res["frame_found"]:
            print(f"{nbytes:>6} {'-':>5} {'-':>9} {'NO-SYNC':>7} {'-':>8}")
            continue

        evm = float(np.asarray(res["evm"]).ravel()[0])
        py_ok = bool(np.array_equal(
            np.asarray(res["bits"]).ravel()[:nbits], bits.ravel()))
        py_units = np.packbits(seen["deint_out"].ravel().astype(np.uint8))

        enc = ofdm.packetizer.encoded_length(nbits)
        # The deinterleaver sits AFTER the Viterbi, so it works on the
        # DECODED stream, not the encoded one. Sizing it from
        # encoded_length made it wait for 513 units when 256 were coming
        # and emit nothing -- which looked exactly like an RTL failure
        # under noise. Take the real figure from the hooked call.
        n_units = seen["deint_in"].ravel().size // 8
        M = 1 + int(np.floor(np.sqrt(n_units)))
        N = -(-n_units // M)

        stim = os.path.join(BUILD, "rxtop_stim.hex")
        with open(stim, "w") as f:
            for z in rx[0]:
                i = int(np.clip(round(float(np.real(z)) * FULL_SCALE), -32768, 32767))
                q = int(np.clip(round(float(np.imag(z)) * FULL_SCALE), -32768, 32767))
                f.write(f"{((i & 0xFFFF) << 16) | (q & 0xFFFF):08x}\n")
        unitp = os.path.join(BUILD, "rxtop_units.txt")
        with open(os.path.join(BUILD, "rxtop_tb_params.vh"), "w") as f:
            f.write(f'`define RXT_STIM_PATH "{stim}"\n')
            f.write(f'`define RXT_HDR_PATH "{os.path.join(BUILD,"rxtop_hdr.txt")}"\n')
            f.write(f'`define RXT_UNIT_PATH "{unitp}"\n')
            f.write(f"`define RXT_NSAMP {n}\n`define RXT_DRAIN 400000\n")
            f.write(f"`define RXT_ENC_BITS {enc}\n`define RXT_DI_UNITS {n_units}\n")
            f.write(f"`define RXT_DI_ROWS {M}\n`define RXT_DI_COLS {N}\n")

        mdir = os.path.join(BUILD, "vsim_dbg")
        r = subprocess.run(
            ["verilator", "--binary", "--timing", "-Wno-WIDTHEXPAND",
             "-Wno-WIDTHTRUNC", "--public-flat-rw", "-Isrc/generated", "-Isrc",
             "-I.", "--top-module", "rx_top_dbg", "-o", "dbg_vsim",
             "--Mdir", mdir] + SRCS, cwd=HERE, capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write(r.stdout + r.stderr); raise SystemExit("build failed")
        subprocess.run([os.path.join(mdir, "dbg_vsim")], cwd=HERE,
                       capture_output=True, text=True, check=True)

        rtl_units = (np.loadtxt(unitp, ndmin=1).astype(int)
                     if os.path.getsize(unitp) else np.zeros(0, int))
        m = min(len(rtl_units), len(py_units))
        diff = np.where(rtl_units[:m] != py_units[:m])[0]
        rtl_ok = (m == len(py_units)) and len(diff) == 0
        first_bit = int(diff[0]) * 8 if len(diff) else -1

        a = np.loadtxt(os.path.join(BUILD, "rxtop_cpe.txt"), ndmin=2)
        rtl_con = (a[:, 0] + 1j * a[:, 1]) / 4096.0
        py_con = seen["constellation"].ravel()

        nsym = int(np.asarray(res["n_payload_symbols"]).ravel()[0])
        print(f"{nbytes:>6} {nsym:>5} {evm:>9.4f} {'YES' if py_ok else 'NO':>7} "
              f"{'YES' if rtl_ok else 'NO':>8} {len(py_units):>9} "
              f"{len(rtl_units):>10} {(first_bit if first_bit >= 0 else '-'):>15}")
        nn = min(len(py_con), len(rtl_con))
        if nn:
            d = py_con[:nn] - rtl_con[:nn]
            # Does the py-vs-rtl error GROW through the frame? A constant
            # level is quantisation; a ramp is uncorrected phase drift.
            q = max(1, nn // 4)
            quarts = [float(np.sqrt(np.mean(np.abs(d[i*q:(i+1)*q])**2))) for i in range(4)]
            print(f"        constellation |py-rtl| rms={float(np.sqrt(np.mean(np.abs(d)**2))):.6f}"
                  f"  by quarter: " + " ".join(f"{x:.6f}" for x in quarts))
        results.append((nbytes, nsym, evm, py_ok, rtl_ok, diff,
                        py_con, rtl_con, py_units, rtl_units))

    # --- constellation detail for the first divergence -----------------
    from spectracuda.modem import Modem as M2
    ideal = np.asarray(M2("qam64").modulate(
        np.array([[int(b) for b in f"{v:06b}"] for v in range(64)]).reshape(1, -1)
    )).ravel()

    for (nbytes, nsym, evm, py_ok, rtl_ok, diff,
         py_con, rtl_con, py_units, rtl_units) in results:
        if rtl_ok or not py_ok or len(diff) == 0:
            continue
        u = int(diff[0])
        print(f"\n--- first divergence: {nbytes} B, unit {u} "
              f"(py=0x{py_units[u]:02x} rtl=0x{rtl_units[u]:02x}) ---")
        sym = (u * 8) // 6
        lo, hi = max(0, sym - 2), min(len(py_con), sym + 3)
        print(f"{'sym':>6} {'python':>22} {'rtl':>22} "
              f"{'py_dmin':>9} {'rtl_dmin':>9}")
        for k in range(lo, hi):
            pc = py_con[k] if k < len(py_con) else 0
            rc = rtl_con[k] if k < len(rtl_con) else 0
            pd = float(np.min(np.abs(ideal - pc)))
            rd = float(np.min(np.abs(ideal - rc)))
            mark = " <--" if k == sym else ""
            print(f"{k:>6} {pc.real:>10.4f}{pc.imag:>+10.4f}j "
                  f"{rc.real:>10.4f}{rc.imag:>+10.4f}j {pd:>9.4f} {rd:>9.4f}{mark}")
        n = min(len(py_con), len(rtl_con))
        print(f"  constellation RMS |py - rtl| = "
              f"{float(np.sqrt(np.mean(np.abs(py_con[:n]-rtl_con[:n])**2))):.6f}")
        break


if __name__ == "__main__":
    main()
