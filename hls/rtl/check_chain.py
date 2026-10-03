"""Run the existing RTL blocks in CHAIN order against one clean frame.

Every block here was already verified in isolation. That says nothing
about whether they agree with each other: scaling/Q-format, bit order,
valid timing and symbol boundaries are all still untested assumptions,
and they are exactly where a port like this breaks. This script exists to
find the FIRST boundary where the RTL stops matching Python.

Python fills every stage that has no RTL yet (grid extraction, training
averaging, pilot CPE, header decode) and also the FFT -- cp_fft
instantiates xfft_256, which needs Xilinx simulation primitives that
Verilator does not have. Those stages are reported as PY, not as passes.

Prerequisite:  python -m hls.rtl.trace     (writes build/trace/)
Usage:         python check_chain.py       (run from hls/rtl/)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, Optional

import numpy as np

# The pinned golden model, NOT the working-tree spectracuda -- see golden_ref.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_ref  # noqa: E402
golden_ref.use()

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
TRACE = os.path.join(BUILD, "trace")   # overridden by --trace-dir

FULL_SCALE = (1 << 15) - 1
ACC_W = 48
ANGLE_W = 16
ANGLE_TOL_LSB = 4      # same budget check_cfo.py uses
EVM_TOL = 0.02
CORR_TOL = 0.999      # check_ce.py / check_eq.py budgets
Q12 = 1 << 12

VERILATOR_WARN_OFF = ["-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC"]


def trace_array(name: str) -> np.ndarray:
    return np.load(os.path.join(TRACE, name))


def build_sim(top: str, sources: list, out: str) -> str:
    """Verilate one testbench. Returns the simulator binary path."""
    mdir = os.path.join(BUILD, f"vsim_{out}")
    exe = os.path.join(mdir, out)
    newest = max(os.path.getmtime(os.path.join(HERE, s)) for s in sources)
    if os.path.exists(exe) and os.path.getmtime(exe) >= newest:
        return exe
    cmd = ["verilator", "--binary", "--timing", *VERILATOR_WARN_OFF,
           "-Isrc", "-Isrc/generated", "-I.",
           "--top-module", top, "-o", out, "--Mdir", mdir, *sources]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"verilator failed for {top}")
    return exe


def run_sim(exe: str) -> str:
    r = subprocess.run([exe], cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"sim failed: {exe}")
    return r.stdout


class Result:
    def __init__(self, name: str, kind: str, ok: Optional[bool],
                 detail: str) -> None:
        self.name, self.kind, self.ok, self.detail = name, kind, ok, detail


# --------------------------------------------------------------------
# Stage: sc_sync
# --------------------------------------------------------------------
def stage_sc_sync() -> Result:
    src = os.path.join(TRACE, "rx_iq.txt")
    subprocess.run([sys.executable, "gen_stimulus.py", src],
                   cwd=HERE, check=True, capture_output=True)
    exe = build_sim("sc_sync_tb", ["tb/sc_sync_tb.v", "src/sc_sync_rtl.v"],
                    "sc_sync_vsim")
    run_sim(exe)

    rows = np.loadtxt(os.path.join(BUILD, "rtl_metric.txt"))
    d = rows[:, 0].astype(np.int64)
    p_re, p_im, r_sum = rows[:, 1], rows[:, 2], rows[:, 3]
    # R = (r1 + r2)/2 -- the RTL emits r1+r2 because halving is free for
    # the consumer. Same convention as check.py; not re-derived here.
    den = (r_sum * 0.5) ** 2
    metric = np.where(den > 0, (p_re ** 2 + p_im ** 2) / np.where(den > 0, den, 1.0), 0.0)
    rtl_start = int(d[int(np.argmax(metric))])
    rtl_peak = float(np.max(metric))

    py_start = int(trace_array("sync_0.out.start_index.npy").ravel()[0])
    py_peak = float(trace_array("sync_0.out.metric.npy").ravel()[0])
    rel = abs(rtl_peak - py_peak) / (abs(py_peak) or 1.0)

    ok = (rtl_start == py_start) and rel <= 2e-2
    return Result("sc_sync", "RTL", ok,
                  f"start_index rtl={rtl_start} py={py_start} | "
                  f"peak rtl={rtl_peak:.6f} py={py_peak:.6f} rel={rel:.3g}")


def _pack_iq(iq: np.ndarray, path: str) -> int:
    """int16 I/Q packed one 32-bit word per line -- byte-identical to
    gen_stimulus.py/gen_cfo_stimulus.py's conversion (round half away
    from zero, saturate), so the chain cannot disagree with the
    standalone testbenches about scaling."""
    z = np.asarray(iq).ravel()
    out = []
    for comp in (np.real(z), np.imag(z)):
        scaled = comp * FULL_SCALE
        code = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
        out.append(np.clip(code, -32768, 32767).astype(np.int64))
    with open(path, "w") as f:
        for i, q in zip(*out):
            f.write(f"{((int(i) & 0xFFFF) << 16) | (int(q) & 0xFFFF):08x}\n")
    return len(z)


def stage_cfo(rtl_start: int) -> Result:
    """cfo_estimate + cfo_correct, driven by sc_sync's OWN P output.

    This is the first real inter-block interface in the chain: P crosses
    from sc_sync to cfo_estimate. Feeding Python's P instead would test
    the two blocks but not the seam between them.
    """
    rows = np.loadtxt(os.path.join(BUILD, "rtl_metric.txt"))
    row = rows[rows[:, 0].astype(np.int64) == rtl_start]
    if len(row) != 1:
        return Result("cfo", "RTL", False,
                      f"sc_sync emitted no P at d={rtl_start}")
    p_re, p_im = int(row[0][1]), int(row[0][2])

    mask = (1 << ACC_W) - 1
    with open(os.path.join(BUILD, "cfo_p.hex"), "w") as f:
        f.write(f"{p_re & mask:012x}\n{p_im & mask:012x}\n")

    rx = trace_array("rx_iq.npy")
    n = _pack_iq(rx, os.path.join(BUILD, "cfo_stimulus.hex"))
    with open(os.path.join(BUILD, "cfo_params.vh"), "w") as f:
        f.write("// GENERATED by check_chain.py -- do not edit\n")
        f.write(f"`define CFO_N_SAMPLES {max(n, 8192)}\n")
        f.write(f"`define CFO_N_ACTUAL {n}\n")

    exe = build_sim("cfo_tb", ["tb/cfo_tb.v", "src/cfo_estimate.v",
                               "src/cfo_correct.v", "src/cordic_rot.v",
                               "src/cordic_vec.v"], "cfo_vsim")
    run_sim(exe)

    # --- angle: estimator alone, judged in phase-word LSBs -------------
    ang = np.loadtxt(os.path.join(BUILD, "cfo_angle.txt"), ndmin=2)
    got_angle = int(ang[0, 0])
    py_cfo = float(np.asarray(trace_array("cfo_estimate_0.out.npy")).ravel()[0])
    # angle(P)/2pi = cfo/2  -- see hls/gen/emit_rtl.py's derivation of
    # d_phase from the Schmidl-Cox lag of fft_size/2.
    want_turns = py_cfo / 2.0
    want_angle = int(np.floor(want_turns * (1 << ANGLE_W) + 0.5))
    full = 1 << ANGLE_W
    want_angle = ((want_angle + full // 2) % full) - full // 2
    aerr = (got_angle - want_angle + full // 2) % full - full // 2

    # --- samples: derotator + estimator, judged as added EVM -----------
    got = np.loadtxt(os.path.join(BUILD, "cfo_out.txt"), ndmin=2)
    rtl_iq = (got[:, 0] + 1j * got[:, 1]) / FULL_SCALE
    py_iq = np.asarray(trace_array("cfo_correct_0.out.npy")).ravel()
    m = min(len(rtl_iq), len(py_iq))
    ref = np.sqrt(np.mean(np.abs(py_iq[:m]) ** 2))
    evm = float(np.sqrt(np.mean(np.abs(rtl_iq[:m] - py_iq[:m]) ** 2)) / (ref or 1.0))

    ok = abs(aerr) <= ANGLE_TOL_LSB and evm <= EVM_TOL
    return Result("cfo_est+corr", "RTL", ok,
                  f"angle rtl={got_angle} py={want_angle} err={aerr} LSB | "
                  f"evm={evm*100:.3f}% (py_cfo={py_cfo:.6f})")


def _evm_corr(rtl: np.ndarray, py: np.ndarray):
    """EVM + correlation, the pair check_ce.py/check_eq.py judge on."""
    m = min(len(rtl), len(py))
    a, b = rtl[:m], py[:m]
    ref = np.sqrt(np.mean(np.abs(b) ** 2)) or 1.0
    evm = float(np.sqrt(np.mean(np.abs(a - b) ** 2)) / ref)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    corr = float(np.abs(np.vdot(b, a)) / (na * nb)) if na and nb else 0.0
    return evm, corr


def _write_params(name: str, lines: list) -> None:
    with open(os.path.join(BUILD, name), "w") as f:
        f.write("// GENERATED by check_chain.py -- do not edit\n")
        for l in lines:
            f.write(l + "\n")


def _pack_c(vals, scale, bits, path):
    """Pack complex as two signed fields of `bits`, high=I low=Q."""
    mask = (1 << bits) - 1
    lim = 1 << (bits - 1)
    digits = (2 * bits + 3) // 4
    with open(path, "w") as f:
        for z in np.asarray(vals).ravel():
            re_ = int(np.clip(round(float(np.real(z)) * scale), -lim, lim - 1))
            im_ = int(np.clip(round(float(np.imag(z)) * scale), -lim, lim - 1))
            f.write(f"{((re_ & mask) << bits) | (im_ & mask):0{digits}x}\n")


def stage_ls_chanest() -> Result:
    pil = trace_array("chanest_0.in0.npy").ravel()
    out_p = os.path.join(BUILD, "chain_ce_out.txt")
    _pack_c(pil, Q12, 20, os.path.join(BUILD, "chain_ce_stim.hex"))
    _write_params("ce_params.vh", [
        f'`define CE_STIM_PATH "{os.path.join(BUILD, "chain_ce_stim.hex")}"',
        f'`define CE_OUT_PATH "{out_p}"',
        f"`define CE_N_IN {len(pil)}", "`define CE_N_BINS 256"])
    run_sim(build_sim("ls_chanest_tb", ["tb/ls_chanest_tb.v", "src/ls_chanest.v"],
                      "ce_vsim"))
    got = np.loadtxt(out_p, ndmin=2)
    rtl = (got[:, 0] + 1j * got[:, 1]) / Q12
    py = np.asarray(trace_array("chanest_0.out.npy")).ravel()
    evm, corr = _evm_corr(rtl, py)
    return Result("ls_chanest", "RTL", evm <= EVM_TOL and corr >= CORR_TOL,
                  f"{len(pil)} pilots -> {len(rtl)} bins | evm={evm*100:.3f}% corr={corr:.6f}")


def stage_mmse_eq() -> Result:
    """All three equalize calls, because they are three different cases:
      0 = header      (1 x 216 data subcarriers)
      1 = payload     (4 x 216) -- the one that feeds the demapper
      2 = pilots      (4 x 8)   -- used for CPE, different channel estimate
    Testing only one of these was how an earlier version of this script
    reported a PASS on 32 pilot values and called it the payload path.
    """
    lim, mask = 1 << 17, 0x3FFFF
    worst_evm, worst_corr, parts = 0.0, 1.0, []
    for call, what in ((0, "hdr"), (1, "payload"), (2, "pilots")):
        rx = trace_array(f"equalize_{call}.in0.npy").ravel()
        h = trace_array(f"equalize_{call}.kw.channel_est.npy").ravel()
        py = np.asarray(trace_array(f"equalize_{call}.out.npy")).ravel()
        n = min(len(rx), len(h), len(py))
        rx, h, py = rx[:n], h[:n], py[:n]
        out_p = os.path.join(BUILD, f"chain_eq{call}_out.txt")
        stim = os.path.join(BUILD, f"chain_eq{call}_stim.hex")
        with open(stim, "w") as f:
            for zr, zh in zip(rx, h):
                v = [int(np.clip(round(float(x) * Q12), -lim, lim - 1)) for x in
                     (np.real(zr), np.imag(zr), np.real(zh), np.imag(zh))]
                w = ((v[0] & mask) << 54) | ((v[1] & mask) << 36) \
                    | ((v[2] & mask) << 18) | (v[3] & mask)
                f.write(f"{w:018x}\n")
        _write_params("eq_tb_params.vh", [
            f'`define EQ_STIM_PATH "{stim}"',
            f'`define EQ_OUT_PATH "{out_p}"', f"`define EQ_N {n}"])
        # Rebuild every call: EQ_N is compile-time, so a cached binary would
        # silently run the previous call's length (the same trap
        # gen_cfo_stimulus.py documents for its P value).
        exe = build_sim("mmse_eq_tb", ["tb/mmse_eq_tb.v", "src/mmse_eq.v"],
                        f"eq{call}_vsim")
        run_sim(exe)
        got = np.loadtxt(out_p, ndmin=2)
        rtl = (got[:, 0] + 1j * got[:, 1]) / Q12
        evm, corr = _evm_corr(rtl, py)
        worst_evm, worst_corr = max(worst_evm, evm), min(worst_corr, corr)
        parts.append(f"{what}[{n}] evm={evm*100:.3f}% corr={corr:.6f}")
    return Result("mmse_eq", "RTL",
                  worst_evm <= EVM_TOL and worst_corr >= CORR_TOL,
                  " | ".join(parts))


def stage_demapper() -> Result:
    sym = trace_array("demod_stats:qam64_0.in0.npy").ravel()
    py_bits = np.asarray(trace_array("demod_stats:qam64_0.out.0.npy")).ravel().astype(int)
    out_p = os.path.join(BUILD, "chain_dm_out.txt")
    _pack_c(sym, Q12, 18, os.path.join(BUILD, "chain_dm_stim.hex"))
    _write_params("dm_tb_params.vh", [
        f'`define DM_STIM_PATH "{os.path.join(BUILD, "chain_dm_stim.hex")}"',
        f'`define DM_OUT_PATH "{out_p}"', f"`define DM_N {len(sym)}",
        "`define DM_SCHEME 2", "`define DM_BPS 6"])
    run_sim(build_sim("demapper_tb", ["tb/demapper_tb.v", "src/demapper.v"], "dm2_vsim"))
    rows = [l.strip() for l in open(out_p) if l.strip()]
    rtl_bits = np.array([int(c) for r in rows for c in r], dtype=int)
    m = min(len(rtl_bits), len(py_bits))
    bad = int(np.sum(rtl_bits[:m] != py_bits[:m]))
    return Result("demapper", "RTL", bad == 0,
                  f"{len(sym)} qam64 symbols -> {m} bits | mismatches={bad}")


def stage_viterbi() -> Result:
    enc = np.asarray(trace_array("fec:conv_v27_0.in0.npy")).ravel().astype(int)
    py = np.asarray(trace_array("fec:conv_v27_0.out.npy")).ravel().astype(int)
    n_sym, n_bits = len(enc) // 2, len(py)
    out_p = os.path.join(BUILD, "chain_vit_out.txt")
    with open(os.path.join(BUILD, "chain_vit_stim.hex"), "w") as f:
        for i in range(n_sym):
            f.write(f"{(int(enc[2 * i + 1]) << 1) | int(enc[2 * i]):01x}\n")
    _write_params("vit_tb_params.vh", [
        f'`define VIT_STIM_PATH "{os.path.join(BUILD, "chain_vit_stim.hex")}"',
        f'`define VIT_OUT_PATH "{out_p}"', f"`define VIT_NSYM {n_sym}",
        f"`define VIT_NBITS {n_bits}"])
    run_sim(build_sim("viterbi_tb", ["tb/viterbi_tb.v", "src/viterbi_dec.v"], "vit_vsim"))
    rtl = np.loadtxt(out_p, ndmin=1).astype(int)
    m = min(len(rtl), len(py))
    bad = int(np.sum(rtl[:m] != py[:m]))
    return Result("viterbi_dec", "RTL", bad == 0 and m == n_bits,
                  f"{n_sym} symbols -> {m}/{n_bits} bits | mismatches={bad}")


def stage_deinterleaver() -> Result:
    bits_in = np.asarray(trace_array("deinterleave:BlockInterleaver_0.in0.npy")).ravel().astype(int)
    py_out = np.asarray(trace_array("deinterleave:BlockInterleaver_0.out.npy")).ravel().astype(int)
    units_in = np.packbits(bits_in.astype(np.uint8))
    py_units = np.packbits(py_out.astype(np.uint8))
    n_units = len(units_in)
    # liquid-dsp's geometry, mirrored from interleaver/block.py:67.
    M = 1 + int(np.floor(np.sqrt(n_units)))
    N = int(np.ceil(n_units / M))
    out_p = os.path.join(BUILD, "chain_di_out.txt")
    with open(os.path.join(BUILD, "chain_di_stim.hex"), "w") as f:
        for b in units_in:
            f.write(f"{int(b):02x}\n")
    _write_params("deint_tb_params.vh", [
        f'`define DI_STIM_PATH "{os.path.join(BUILD, "chain_di_stim.hex")}"',
        f'`define DI_OUT_PATH "{out_p}"', f"`define DI_N {n_units}",
        f"`define DI_ROWS {M}", f"`define DI_COLS {N}"])
    run_sim(build_sim("deinterleaver_tb", ["tb/deinterleaver_tb.v",
                                           "src/deinterleaver.v"], "di_vsim"))
    rtl = np.loadtxt(out_p, ndmin=1).astype(int)
    m = min(len(rtl), len(py_units))
    bad = int(np.sum(rtl[:m] != py_units[:m]))
    return Result("deinterleaver", "RTL", bad == 0,
                  f"{n_units} units (M={M} N={N}) | mismatches={bad}")


def stage_grid_extract() -> Result:
    """Pure index routing, so this is judged EXACTLY -- not by EVM.

    grid_extract does no arithmetic: every bin it emits must be the bin
    that went in, bit for bit. An EVM tolerance here would hide exactly
    the failure this block can have, which is routing a bin to the wrong
    stream.
    """
    grid = trace_array("fft_2.out.npy")               # (4, 256) payload
    py_data = np.asarray(trace_array("grid_data_1.out.npy"))
    py_pil = np.asarray(trace_array("grid_pilots_0.out.npy"))
    nsym, nfft = grid.shape

    stim = os.path.join(BUILD, "chain_ge_stim.hex")
    dpath = os.path.join(BUILD, "chain_ge_data.txt")
    ppath = os.path.join(BUILD, "chain_ge_pilot.txt")
    mask = 0xFFFFFFFF
    with open(stim, "w") as f:
        for z in grid.ravel():
            re_ = int(round(float(np.real(z)) * Q12))
            im_ = int(round(float(np.imag(z)) * Q12))
            f.write(f"{((re_ & mask) << 32) | (im_ & mask):016x}\n")
    _write_params("ge_tb_params.vh", [
        f'`define GE_STIM_PATH "{stim}"', f'`define GE_DATA_PATH "{dpath}"',
        f'`define GE_PILOT_PATH "{ppath}"', f"`define GE_NSYM {nsym}",
        f"`define GE_NFFT {nfft}"])
    run_sim(build_sim("grid_extract_tb", ["tb/grid_extract_tb.v",
                                          "src/grid_extract.v"], "ge_vsim"))

    def _read(path):
        a = np.loadtxt(path, ndmin=2)
        return (a[:, 0] + 1j * a[:, 1]) if len(a) else np.zeros(0, complex)

    rtl_d, rtl_p = _read(dpath), _read(ppath)
    # Compare in integer code units: the block is routing, so the exact
    # quantized value must survive.
    def _codes(z):
        return np.stack([np.round(np.real(z) * Q12), np.round(np.imag(z) * Q12)])
    wd = _codes(py_data.ravel())
    wp = _codes(py_pil.ravel())
    gd = np.stack([np.real(rtl_d), np.imag(rtl_d)])
    gp = np.stack([np.real(rtl_p), np.imag(rtl_p)])

    ok_len = gd.shape[1] == wd.shape[1] and gp.shape[1] == wp.shape[1]
    bad_d = int(np.sum(gd != wd)) if ok_len else -1
    bad_p = int(np.sum(gp != wp)) if ok_len else -1
    ok = ok_len and bad_d == 0 and bad_p == 0
    return Result("grid_extract", "RTL", ok,
                  f"{nsym}x{nfft} bins -> data {gd.shape[1]}/{wd.shape[1]} "
                  f"(mism={bad_d}) pilots {gp.shape[1]}/{wp.shape[1]} (mism={bad_p})")


def stage_pilot_cpe() -> Result:
    """CPE correction for the payload symbols.

    in : equalize_1.out (payload data), equalize_2.out (payload pilots)
    out: demod_stats:qam64_0.in0 -- which IS the post-CPE data, because
         ofdm.py applies the rotation to equalized_combined immediately
         before handing it to the demodulator.
    """
    data = np.asarray(trace_array("equalize_1.out.npy"))       # (nsym, n_data)
    pil = np.asarray(trace_array("equalize_2.out.npy"))        # (nsym, n_pilot)
    py = np.asarray(trace_array("demod_stats:qam64_0.in0.npy"))
    nsym, ndata = data.shape
    npil = pil.shape[1]

    lim, mask = 1 << 17, 0x3FFFF

    def _w(path, arr):
        with open(path, "w") as f:
            for z in arr.ravel():
                re_ = int(np.clip(round(float(np.real(z)) * Q12), -lim, lim - 1))
                im_ = int(np.clip(round(float(np.imag(z)) * Q12), -lim, lim - 1))
                f.write(f"{((re_ & mask) << 18) | (im_ & mask):09x}\n")

    dst = os.path.join(BUILD, "chain_cpe_d.hex")
    pst = os.path.join(BUILD, "chain_cpe_p.hex")
    outp = os.path.join(BUILD, "chain_cpe_out.txt")
    angp = os.path.join(BUILD, "chain_cpe_ang.txt")
    _w(dst, data); _w(pst, pil)
    _write_params("cpe_tb_params.vh", [
        f'`define CPE_DSTIM_PATH "{dst}"', f'`define CPE_PSTIM_PATH "{pst}"',
        f'`define CPE_OUT_PATH "{outp}"', f'`define CPE_ANGLE_PATH "{angp}"',
        f"`define CPE_NSYM {nsym}", f"`define CPE_NDATA {ndata}",
        f"`define CPE_NPIL {npil}"])
    run_sim(build_sim("pilot_cpe_tb", ["tb/pilot_cpe_tb.v", "src/pilot_cpe.v",
                                       "src/cordic_vec.v", "src/cordic_rot.v"],
                      "cpe_vsim"))

    got = np.loadtxt(outp, ndmin=2)
    rtl = (got[:, 0] + 1j * got[:, 1]) / Q12
    want = py.ravel()
    n = min(len(rtl), len(want))
    if n == 0:
        return Result("pilot_cpe", "RTL", False, "no output produced")
    evm, corr = _evm_corr(rtl[:n], want[:n])

    # Also report the angle the RTL chose against Python's own, since a
    # correct-looking EVM with a wrong angle is possible when the CPE is
    # near zero.
    ang = np.loadtxt(angp, ndmin=1)
    py_ang = np.angle(np.mean(pil / 1.0, axis=-1))          # pilots are +1
    py_turns = py_ang / (2 * np.pi)
    m = min(len(ang), len(py_turns))
    aerr = 0
    if m:
        want_code = np.round(py_turns[:m] * (1 << 16)).astype(int)
        full = 1 << 16
        aerr = int(np.max(np.abs((ang[:m].astype(int) - want_code + full // 2) % full - full // 2)))

    ok = (n == len(want)) and evm <= EVM_TOL and corr >= CORR_TOL and aerr <= 4
    return Result("pilot_cpe", "RTL", ok,
                  f"{nsym}x{ndata} data, {npil} pilots | out {n}/{len(want)} "
                  f"evm={evm*100:.3f}% corr={corr:.6f} max_angle_err={aerr} LSB")


MOD_CODES = {"bpsk": 0, "qpsk": 1, "qam16": 2, "qam64": 3, "qam256": 8 - 4}
BPS = {"bpsk": 1, "qpsk": 2, "qam16": 4, "qam64": 6, "qam256": 8}
CRC_CODES = {"none": 1, "checksum": 2, "crc8": 3, "crc16": 4, "crc24": 5, "crc32": 6}


def stage_header_decode(summary: dict) -> Result:
    """Whole header symbol in (216 demapped bits), parsed fields out."""
    bits = np.asarray(trace_array("demod_bits:bpsk_0.out.npy")).ravel().astype(int)
    want = summary.get("header") or {}
    if not want:
        return Result("header_decode", "RTL", False, "trace has no header fields")

    stim = os.path.join(BUILD, "chain_hdr_stim.hex")
    outp = os.path.join(BUILD, "chain_hdr_out.txt")
    with open(stim, "w") as f:
        for b in bits:
            f.write(f"{int(b):01x}\n")
    _write_params("hdr_tb_params.vh", [
        f'`define HDRTB_STIM_PATH "{stim}"', f'`define HDRTB_OUT_PATH "{outp}"',
        f"`define HDRTB_NSLOT {len(bits)}"])
    run_sim(build_sim("header_decode_tb", ["tb/header_decode_tb.v",
                                           "src/header_decode.v"], "hdr_vsim"))

    vals = [int(l) for l in open(outp) if l.strip()]
    if len(vals) < 9:
        return Result("header_decode", "RTL", False,
                      f"only {len(vals)} outputs (header never completed)")
    (ver, plen, mods, bps, crcc, f0, f1, udata, fvalid) = vals[:9]

    from spectracuda.framing.header import (MOD_SCHEME_CODES, CRC_SCHEME_CODES,
                                            FEC_SCHEME_CODES)
    checks = [
        ("protocol_version", ver, int(want["protocol_version"])),
        ("payload_len_bits", plen, int(want["payload_len_bits"])),
        ("mod_scheme", mods, MOD_SCHEME_CODES[want["mod_scheme"]]),
        ("bits_per_symbol", bps, BPS[want["mod_scheme"]]),
        ("crc", crcc, CRC_SCHEME_CODES[want["crc"]]),
        ("fec0", f0, FEC_SCHEME_CODES[want["fec0"]]),
        ("fec1", f1, FEC_SCHEME_CODES[want["fec1"]]),
        ("fields_valid", fvalid, 1),
    ]
    bad = [f"{n}: rtl={g} py={w}" for n, g, w in checks if g != w]
    detail = (f"{len(bits)} slots -> 112 bits | "
              f"len={plen} mod={want['mod_scheme']}({mods}) "
              f"crc={want['crc']}({crcc}) fec0={f0} fec1={f1} valid={fvalid}")
    if bad:
        detail = "MISMATCH " + "; ".join(bad)
    return Result("header_decode", "RTL", not bad, detail)


def stage_frame_sync() -> Result:
    """sc_sync + frame_sync together: detect, then replay from the start.

    Tests the seam as well as the block -- frame_sync consumes sc_sync's
    live P/R stream, so a latency mismatch between the two shows up here
    as a shifted start_index rather than hiding until integration.
    """
    rx = trace_array("rx_iq.npy")
    py_start = int(trace_array("sync_0.out.start_index.npy").ravel()[0])
    n = _pack_iq(rx, os.path.join(BUILD, "chain_fs_stim.hex"))
    outp = os.path.join(BUILD, "chain_fs_out.txt")
    metap = os.path.join(BUILD, "chain_fs_meta.txt")
    frame_len = 1984
    _write_params("fs_tb_params.vh", [
        f'`define FS_STIM_PATH "{os.path.join(BUILD, "chain_fs_stim.hex")}"',
        f'`define FS_OUT_PATH "{outp}"', f'`define FS_META_PATH "{metap}"',
        f"`define FS_NSAMP {n}", "`define FS_LAG 128",
        f"`define FS_FRAME_LEN {frame_len}"])
    run_sim(build_sim("frame_sync_tb", ["tb/frame_sync_tb.v", "src/frame_sync.v",
                                        "src/sc_sync_rtl.v"], "fs_vsim"))

    meta = [int(l) for l in open(metap) if l.strip()]
    if not meta:
        return Result("frame_sync", "RTL", False,
                      "never detected a frame (threshold or guard never fired)")
    rtl_start = meta[0]

    got = np.loadtxt(outp, ndmin=2)
    if len(got) == 0:
        return Result("frame_sync", "RTL", False,
                      f"detected at {rtl_start} but replayed nothing")
    rtl_iq = (got[:, 0] + 1j * got[:, 1]) / FULL_SCALE

    # The replay must be the ORIGINAL samples from the detected index --
    # compared against the quantized input, since that is what the buffer
    # holds. Any offset error shows up as a large EVM, not a small one.
    src = np.asarray(rx).ravel()
    want = src[rtl_start:rtl_start + len(rtl_iq)]
    m = min(len(rtl_iq), len(want))
    q = np.round(np.real(want[:m]) * FULL_SCALE) + 1j * np.round(np.imag(want[:m]) * FULL_SCALE)
    g = np.round(np.real(rtl_iq[:m]) * FULL_SCALE) + 1j * np.round(np.imag(rtl_iq[:m]) * FULL_SCALE)
    bad = int(np.sum(g != q))

    off = rtl_start - py_start
    ok = (off == 0) and bad == 0 and m == frame_len
    return Result("frame_sync", "RTL", ok,
                  f"start rtl={rtl_start} py={py_start} (off={off:+d}) | "
                  f"replayed {m}/{frame_len} samples, mismatches={bad}")


# --------------------------------------------------------------------
# Stages still carried by Python (no RTL, or RTL needs xsim)
# --------------------------------------------------------------------
PY_STAGES = [
    ("cp_fft", "needs xsim -- xfft_256 uses Xilinx primitives"),
    ("chanest_avg", "no RTL block exists (training averaging)"),
]


def main() -> None:
    global TRACE
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", default=TRACE)
    args = ap.parse_args()
    TRACE = args.trace_dir
    if not os.path.exists(os.path.join(TRACE, "summary.json")):
        raise SystemExit("no trace -- run:  python -m hls.rtl.trace  first")
    with open(os.path.join(TRACE, "summary.json")) as f:
        summary = json.load(f)

    cfg = summary["config"]
    print(f"clean frame: fft={cfg['fft_size']} cp={cfg['cp_len']} "
          f"modem={cfg['modem']} n_data={cfg['n_data']} n_pilot={cfg['n_pilot']}")
    print(f"  impairments: cfo={summary.get('cfo')} awgn={summary.get('awgn')}")
    print(f"  python reference: frame_found={summary['frame_found']} "
          f"crc_valid={summary['crc_valid']} payload==tx={summary['payload_bits_match_tx']}")
    print()

    sync_res = stage_sc_sync()
    results = [sync_res]
    rtl_start = int(sync_res.detail.split("rtl=")[1].split()[0])
    results.append(stage_cfo(rtl_start))
    try:
        results.append(stage_frame_sync())
    except Exception as exc:
        results.append(Result('frame_sync', 'RTL', False, f'{type(exc).__name__}: {exc}'))
    try:
        results.append(stage_header_decode(summary))
    except Exception as exc:
        results.append(Result('header_decode', 'RTL', False, f'{type(exc).__name__}: {exc}'))
    for fn in (stage_grid_extract, stage_ls_chanest, stage_mmse_eq,
               stage_pilot_cpe, stage_demapper, stage_viterbi,
               stage_deinterleaver):
        try:
            results.append(fn())
        except Exception as exc:   # a stage that cannot even run is a FAIL
            results.append(Result(fn.__name__.replace("stage_", ""), "RTL",
                                  False, f"{type(exc).__name__}: {exc}"))
    for name, why in PY_STAGES:
        results.append(Result(name, "PY", None, why))

    print(f"{'stage':<16} {'by':<5} {'verdict':<8} detail")
    print("-" * 96)
    first_fail = None
    for r in results:
        verdict = "--" if r.ok is None else ("PASS" if r.ok else "FAIL")
        print(f"{r.name:<16} {r.kind:<5} {verdict:<8} {r.detail}")
        if r.ok is False and first_fail is None:
            first_fail = r

    rtl_n = sum(1 for r in results if r.kind == "RTL")
    rtl_ok = sum(1 for r in results if r.kind == "RTL" and r.ok)
    print("-" * 96)
    print(f"RTL stages chained: {rtl_ok}/{rtl_n} passing, "
          f"{len(PY_STAGES)} stages still carried by Python")
    if first_fail is not None:
        raise SystemExit(f"\nFIRST DIVERGENCE: {first_fail.name} -- {first_fail.detail}")
    if rtl_n:
        print("\nNo divergence in the chained RTL stages.")


if __name__ == "__main__":
    main()
