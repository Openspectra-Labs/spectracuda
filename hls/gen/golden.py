"""Dump golden vectors from spectracuda's real pipeline for the HLS
testbenches to check against.

This is the "B" approach (see docs/vitis-hls-ofdm-ip-plan.md): the HLS
blocks implement SPECTRACUDA's frame format, so spectracuda's Python --
which is RF-validated between two Plutos -- is the golden model. Nothing
here reimplements the algorithm; it calls the same `SchmidlCoxSync` the
library uses and writes down what it produced.

File format is deliberately dumb: one `float` per line for IQ (I then Q
interleaved), plain integers/floats for scalars. Text, not binary, so a
mismatch can be eyeballed and diffed rather than hexdumped. These frames
are small enough that parse speed does not matter.

Usage:
    python -m hls.gen.golden            # writes hls/golden/
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np

from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "golden"))

# First HLS target config. Deliberately conv_v27 WITHOUT rs_m8: it maps
# onto FEC blocks that already exist in ~/work/ofdm-hls (same K=7,
# 0171/0133 polynomials). Reed-Solomon in fabric is a later step and is
# still the open build-vs-buy item.
# THE RF-VALIDATED CONFIG, copied from abhi/pluto_rx_standalone_v2.py's
# PHY_KWARGS -- the one actually flown between two Plutos. An earlier
# version of this file used qpsk/conv_v27/crc32/no-interleaver, which was
# a guess and wrong in four places. Every block already built depends only
# on fft_size/n_pilot/n_data/cp_len and the sync/cfo/chanest/equalizer
# choices, all of which are identical between the two, so none of them
# were invalidated -- but the FEC/CRC/interleaver chain would have been
# built to the wrong spec.
#
# The interleaver is NOT decoration: abhi/pluto_rx_standalone_v2.py's own
# comment records it root-causing real 1024-byte over-the-air packet
# losses. Interference arrives as a burst of Viterbi errors in a narrow
# byte range, which blows Reed-Solomon's per-codeword budget; block
# interleaving spreads it across codewords. unit_bits=8 matches rs_m8's
# byte symbol size -- bit-granularity interleaving of a byte-oriented
# code was MEASURED to make bursts worse.
CFG = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qam64",
    fec="rs_m8", fec1="conv_v27", crc="crc16",
    interleaver="block", interleaver_kwargs={"unit_bits": 8},
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
    strict_fec_check=True,
)
PAYLOAD_BITS = 2000
GUARD = 200          # leading/trailing silence, so sync has somewhere to be wrong
SNR_DB = 20.0
CFO_FRAC = 0.05      # fraction of subcarrier spacing
SEED = 7


def _write_iq(path: str, iq: np.ndarray) -> None:
    iq = np.asarray(iq).ravel()
    with open(path, "w") as f:
        for z in iq:
            f.write(f"{float(np.real(z)):.9g} {float(np.imag(z)):.9g}\n")


# A word length claim from ONE frame at ONE comfortable SNR is not a
# claim, it is a coincidence -- quantization noise only competes with
# channel noise near the detection cliff. So the sweep runs over a grid
# that straddles it. spectracuda's own float32 cliff for this config is
# 6-9 dB (measured, examples/fixedpoint_wordlength_study.py).
CASES = [(snr, seed) for snr in (6.0, 8.0, 12.0, 20.0)
                     for seed in (7, 11, 23)]


def emit_case(ofdm: Ofdm, snr_db: float, seed: int, out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=PAYLOAD_BITS).astype(np.uint8)
    tx = np.asarray(ofdm.generate_frame(bits))
    pad = np.zeros((tx.shape[0], GUARD), dtype=tx.dtype)
    tx_padded = np.concatenate([pad, tx, pad], axis=-1)
    ch = Channel(snr_db=snr_db, cfo=CFO_FRAC,
                 cfo_fft_size=CFG["fft_size"], seed=seed)
    rx = np.asarray(ch.process(tx_padded))

    sync_res = ofdm.sync.process(rx)
    start_index = int(np.asarray(sync_res["start_index"]).ravel()[0])
    metric = float(np.asarray(sync_res["metric"]).ravel()[0])

    _write_iq(os.path.join(out_dir, "sc_sync_rx.txt"), rx[0])
    meta = {
        "config": CFG, "snr_db": snr_db, "seed": seed,
        "n_samples": int(rx.shape[-1]), "L": CFG["fft_size"] // 2,
        "expected_start_index": start_index,
        "expected_metric": metric,
        "expected_true_start": GUARD,
    }
    with open(os.path.join(out_dir, "sc_sync_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ofdm = Ofdm(**CFG)

    print("case grid (python SchmidlCoxSync, the golden model):")
    for snr, seed in CASES:
        name = f"snr{int(snr):02d}_s{seed:02d}"
        m = emit_case(ofdm, snr, seed, os.path.join(OUT, name))
        off = m["expected_start_index"] - m["expected_true_start"]
        print(f"  {name}: start_index={m['expected_start_index']:5d} "
              f"(true {m['expected_true_start']}, off by {off:+d})  "
              f"metric={m['expected_metric']:.4f}")

    rng = np.random.default_rng(SEED)

    bits = rng.integers(0, 2, size=PAYLOAD_BITS).astype(np.uint8)
    tx = np.asarray(ofdm.generate_frame(bits))
    pad = np.zeros((tx.shape[0], GUARD), dtype=tx.dtype)
    tx_padded = np.concatenate([pad, tx, pad], axis=-1)

    ch = Channel(snr_db=SNR_DB, cfo=CFO_FRAC,
                 cfo_fft_size=CFG["fft_size"], seed=SEED)
    rx = np.asarray(ch.process(tx_padded))

    # The library's own sync block, not a reimplementation.
    sync_res = ofdm.sync.process(rx)
    start_index = int(np.asarray(sync_res["start_index"]).ravel()[0])
    metric = float(np.asarray(sync_res["metric"]).ravel()[0])

    # Full-pipeline result, so the testbench can also confirm the frame
    # this vector set was cut from actually decodes.
    res = ofdm.rx_process(rx)

    _write_iq(os.path.join(OUT, "sc_sync_rx.txt"), rx[0])

    meta: Dict[str, Any] = {
        "config": CFG,
        "payload_bits": PAYLOAD_BITS,
        "guard": GUARD,
        "snr_db": SNR_DB,
        "cfo_frac": CFO_FRAC,
        "seed": SEED,
        "n_samples": int(rx.shape[-1]),
        "L": CFG["fft_size"] // 2,
        "expected_start_index": start_index,
        "expected_metric": metric,
        "expected_true_start": GUARD,
        "frame_found": bool(res["frame_found"]),
        "crc_valid": bool(np.asarray(res["crc_valid"]).ravel()[0])
                     if res["crc_valid"] is not None else None,
        "cfo_estimate": float(np.asarray(res["cfo_estimate"]).ravel()[0])
                        if res["cfo_estimate"] is not None else None,
    }
    with open(os.path.join(OUT, "sc_sync_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {OUT}/sc_sync_rx.txt  ({meta['n_samples']} samples)")
    print(f"  L                    = {meta['L']}")
    print(f"  expected_start_index = {start_index}  (true frame start {GUARD})")
    print(f"  expected_metric      = {metric:.9g}")
    print(f"  frame decodes        = found={meta['frame_found']} "
          f"crc_valid={meta['crc_valid']}")




# ---------------------------------------------------------------- CFO

def emit_cfo_case(ofdm: Ofdm, snr_db: float, seed: int, out_dir: str) -> Dict[str, Any]:
    """Golden data for the CFO block.

    Three artefacts, because the block has two separable halves and a
    joint behaviour, and a failure in one should not be diagnosed through
    the other:

      p_re/p_im at the detected peak -> what sc_sync_rtl hands over
      expected_cfo                   -> angle(P)/pi, the estimator's job
      corrected IQ                   -> the derotator's job

    Everything comes from spectracuda's own SchmidlCoxCFO, never a
    reimplementation of the formula here.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=PAYLOAD_BITS).astype(np.uint8)
    tx = np.asarray(ofdm.generate_frame(bits))
    pad = np.zeros((tx.shape[0], GUARD), dtype=tx.dtype)
    tx_padded = np.concatenate([pad, tx, pad], axis=-1)
    ch = Channel(snr_db=snr_db, cfo=CFO_FRAC,
                 cfo_fft_size=CFG["fft_size"], seed=seed)
    rx = np.asarray(ch.process(tx_padded))

    start_index = np.asarray(ofdm.sync.process(rx)["start_index"])
    d = int(start_index.ravel()[0])
    L = CFG["fft_size"] // 2

    # P at the detected peak -- the exact quantity sc_sync_rtl emits, so
    # the estimator can be tested standalone on real values rather than
    # on a synthetic angle sweep.
    first = rx[0, d:d + L]
    second = rx[0, d + L:d + 2 * L]
    p = complex(np.sum(np.conj(first) * second))

    cfo = np.asarray(ofdm.cfo.process(rx, start_index))
    corrected = np.asarray(ofdm.cfo.correct(rx, cfo))

    _write_iq(os.path.join(out_dir, "cfo_rx.txt"), rx[0])
    _write_iq(os.path.join(out_dir, "cfo_corrected.txt"), corrected[0])

    meta = {
        "config": CFG, "snr_db": snr_db, "seed": seed,
        "n_samples": int(rx.shape[-1]),
        "fft_size": CFG["fft_size"], "L": L,
        "start_index": d,
        "p_re": float(np.real(p)), "p_im": float(np.imag(p)),
        "expected_angle_rad": float(np.angle(p)),
        # The block holds phase as a fraction of a full turn, so the
        # increment is a shift rather than a multiply -- see
        # hls/rtl/src/cfo_correct.v.
        "expected_angle_turns": float(np.angle(p) / (2 * np.pi)),
        "expected_cfo": float(np.asarray(cfo).ravel()[0]),
        "channel_cfo_applied": CFO_FRAC,
    }
    with open(os.path.join(out_dir, "cfo_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def emit_cfo() -> None:
    ofdm = Ofdm(**CFG)
    print("\nCFO golden data (python SchmidlCoxCFO):")
    for snr, seed in CASES:
        name = f"cfo_snr{int(snr):02d}_s{seed:02d}"
        m = emit_cfo_case(ofdm, snr, seed, os.path.join(OUT, name))
        print(f"  {name}: d={m['start_index']:4d}  "
              f"angle={m['expected_angle_rad']:+.6f} rad  "
              f"cfo={m['expected_cfo']:+.6f}  (channel applied "
              f"{m['channel_cfo_applied']:+.3f})")


# ---------------------------------------------------------------- FFT

def emit_fft_case(ofdm: Ofdm, seed: int, out_dir: str) -> Dict[str, Any]:
    """Golden data for CP-strip + FFT.

    Taken from a REAL frame at the real symbol boundary rather than from
    synthetic tones: the thing most likely to be wrong is the offset the
    CP is stripped at, and a synthetic input with no CP cannot catch that.

    `expected` is numpy's UNSCALED forward FFT, matching
    spectracuda/ofdm/fft.py, which uses numpy's standard convention. The
    Xilinx core applies a scaling schedule, so the comparison has to
    account for the factor -- that mismatch is the classic FFT porting
    bug and is checked explicitly rather than absorbed into a tolerance.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=PAYLOAD_BITS).astype(np.uint8)
    tx = np.asarray(ofdm.generate_frame(bits))

    fft_size = CFG["fft_size"]
    cp_len = CFG["cp_len"]
    slot = fft_size + cp_len

    # Frame layout: [preamble (no CP)][training][header][payload...].
    # The first CP-bearing slot starts right after the preamble.
    sym_start = fft_size + slot          # skip preamble and training
    sym = tx[0, sym_start:sym_start + slot]
    no_cp = sym[cp_len:]
    expected = np.fft.fft(no_cp)

    _write_iq(os.path.join(out_dir, "fft_in.txt"), sym)
    _write_iq(os.path.join(out_dir, "fft_expected.txt"), expected)

    meta = {
        "config": CFG, "seed": seed,
        "fft_size": fft_size, "cp_len": cp_len, "slot_len": slot,
        "sym_start": int(sym_start),
        "n_in": int(sym.size), "n_out": int(expected.size),
        "in_peak": float(np.max(np.abs(sym))),
        "out_peak": float(np.max(np.abs(expected))),
        # numpy's forward FFT is unscaled; the Xilinx core is not.
        "python_convention": "unscaled forward (numpy)",
    }
    with open(os.path.join(out_dir, "fft_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def emit_fft() -> None:
    ofdm = Ofdm(**CFG)
    print("\nFFT golden data (one real OFDM symbol, numpy unscaled FFT):")
    for seed in (7, 11, 23):
        name = f"fft_s{seed:02d}"
        m = emit_fft_case(ofdm, seed, os.path.join(OUT, name))
        print(f"  {name}: {m['n_in']} in -> {m['n_out']} out  "
              f"in_peak={m['in_peak']:.4f}  out_peak={m['out_peak']:.2f}  "
              f"(growth {m['out_peak']/m['in_peak']:.1f}x)")


# Keep this block LAST. It has drifted above newly-appended
# functions twice now, which fails with a NameError only when the
# module is run rather than imported.
def emit_chanest_case(ofdm: Ofdm, snr_db: float, seed: int, out_dir: str) -> Dict[str, Any]:
    """Golden data for the LS channel estimator.

    Inputs are the RECEIVED PILOT BINS taken from a real frame through a
    real channel -- not synthetic values -- so the interpolation is
    exercised on an estimate that actually varies across frequency. A flat
    channel would let a broken interpolation table pass.

    Expected output is spectracuda's own LSChannelEstimator.process(),
    called directly.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=PAYLOAD_BITS).astype(np.uint8)
    tx = np.asarray(ofdm.generate_frame(bits))
    pad = np.zeros((tx.shape[0], GUARD), dtype=tx.dtype)
    tx_padded = np.concatenate([pad, tx, pad], axis=-1)

    # A multipath channel, so H varies across subcarriers and the
    # interpolation has something real to reproduce.
    taps = Channel.random_multipath_taps(4, seed=seed)
    ch = Channel(snr_db=snr_db, multipath_taps=taps, seed=seed)
    rx = np.asarray(ch.process(tx_padded))

    fft_size = CFG["fft_size"]
    cp_len = CFG["cp_len"]
    slot = fft_size + cp_len
    train_start = GUARD + fft_size          # after the (CP-less) preamble
    train = rx[:, train_start:train_start + slot]
    grid = np.asarray(ofdm.demod.process(train))

    # The estimator's own indices -- the TRAINING SYMBOL's known
    # subcarriers (224 of them), not the 8 pilot tones. See
    # hls/gen/emit_rtl.py's emit_chanest for the same correction.
    pilot_idx = np.asarray(ofdm.channel_estimator.pilot_indices)
    rx_pilots = grid[:, pilot_idx]
    h_full = np.asarray(ofdm.channel_estimator.process(rx_pilots))

    _write_iq(os.path.join(out_dir, "ce_pilots.txt"), rx_pilots[0])
    _write_iq(os.path.join(out_dir, "ce_expected.txt"), h_full[0])

    meta = {
        "config": CFG, "snr_db": snr_db, "seed": seed,
        "n_pilot": int(pilot_idx.size), "n_fft": fft_size,
        "pilot_indices": [int(v) for v in pilot_idx],
        "pilot_peak": float(np.max(np.abs(rx_pilots))),
        "h_peak": float(np.max(np.abs(h_full))),
    }
    with open(os.path.join(out_dir, "ce_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def emit_chanest() -> None:
    ofdm = Ofdm(**CFG)
    print("\nChannel-estimate golden data (python LSChannelEstimator, multipath):")
    for snr, seed in [(20.0, 7), (12.0, 11), (8.0, 23)]:
        name = f"ce_snr{int(snr):02d}_s{seed:02d}"
        m = emit_chanest_case(ofdm, snr, seed, os.path.join(OUT, name))
        print(f"  {name}: {m['n_pilot']} pilots -> {m['n_fft']} bins  "
              f"pilot_peak={m['pilot_peak']:.3f}  h_peak={m['h_peak']:.3f}")


# ------------------------------------------------------------- equalizer

def emit_eq_case(ofdm: Ofdm, snr_db: float, seed: int, out_dir: str) -> Dict[str, Any]:
    """Golden data for the MMSE equalizer.

    Inputs are a REAL payload symbol's data subcarriers and the channel
    estimate measured from the same frame's training symbol -- not
    synthetic pairs. The equalizer's hard case is a deeply faded
    subcarrier, where |H| is small and the reciprocal is large, and only a
    real multipath channel produces those.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=PAYLOAD_BITS).astype(np.uint8)
    tx = np.asarray(ofdm.generate_frame(bits))
    pad = np.zeros((tx.shape[0], GUARD), dtype=tx.dtype)
    taps = Channel.random_multipath_taps(4, seed=seed)
    ch = Channel(snr_db=snr_db, multipath_taps=taps, seed=seed)
    rx = np.asarray(ch.process(np.concatenate([pad, tx, pad], axis=-1)))

    fft_size = CFG["fft_size"]; cp = CFG["cp_len"]; slot = fft_size + cp
    base = GUARD + fft_size
    train = rx[:, base:base + slot]
    h = np.asarray(ofdm.channel_estimator.process(
        np.asarray(ofdm.demod.process(train))[
            :, np.asarray(ofdm.channel_estimator.pilot_indices)]))

    sym = rx[:, base + 2 * slot:base + 3 * slot]
    grid = np.asarray(ofdm.demod.process(sym))
    di = np.asarray(ofdm.grid.data_indices)
    rx_d = grid[:, di]
    h_d = h[:, di]
    y = np.asarray(ofdm.equalizer.process(rx_d, channel_est=h_d))

    _write_iq(os.path.join(out_dir, "eq_rx.txt"), rx_d[0])
    _write_iq(os.path.join(out_dir, "eq_h.txt"), h_d[0])
    _write_iq(os.path.join(out_dir, "eq_expected.txt"), y[0])

    meta = {
        "config": CFG, "snr_db": snr_db, "seed": seed,
        "n_data": int(di.size),
        "noise_var": float(ofdm.equalizer.noise_var),
        "h_min": float(np.min(np.abs(h_d))), "h_max": float(np.max(np.abs(h_d))),
        "y_peak": float(np.max(np.abs(y))),
    }
    with open(os.path.join(out_dir, "eq_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def emit_eq() -> None:
    ofdm = Ofdm(**CFG)
    print("\nEqualizer golden data (python MMSEEqualizer, multipath):")
    for snr, seed in [(20.0, 7), (12.0, 11), (8.0, 23)]:
        name = f"eq_snr{int(snr):02d}_s{seed:02d}"
        m = emit_eq_case(ofdm, snr, seed, os.path.join(OUT, name))
        print(f"  {name}: {m['n_data']} subcarriers  "
              f"|H| {m['h_min']:.4f}..{m['h_max']:.4f} "
              f"({m['h_max']/max(m['h_min'],1e-9):.0f}x)  y_peak={m['y_peak']:.2f}")


# -------------------------------------------------------------- demapper

def emit_demap_case(scheme: str, out_dir: str, seed: int = 5) -> Dict[str, Any]:
    """Golden data for the hard-decision demapper, one scheme.

    Symbols are the ideal constellation plus graded noise -- from clean
    right out to symbols pushed across a decision boundary. The demapper's
    only interesting behaviour is WHERE it puts the boundaries, so the
    test has to include symbols near them; clean constellation points
    would pass with the thresholds badly wrong.
    """
    os.makedirs(out_dir, exist_ok=True)
    from spectracuda.modem import Modem
    m = Modem(scheme)
    rng = np.random.default_rng(seed)
    bps = m.bits_per_symbol

    n_sym = 4096
    bits = rng.integers(0, 2, size=n_sym * bps).astype(np.uint8)
    sym = np.asarray(m.modulate(bits[None, :]))[0]
    # Graded noise, including levels that will move symbols across
    # boundaries -- that is the point.
    noise = (rng.normal(size=n_sym) + 1j * rng.normal(size=n_sym))
    lvl = np.repeat(np.array([0.0, 0.02, 0.05, 0.12, 0.25]), n_sym // 5 + 1)[:n_sym]
    noisy = (sym + noise * lvl).astype("complex64")

    exp_bits = np.asarray(m.demodulate(noisy[None, :]))[0]

    _write_iq(os.path.join(out_dir, "dm_sym.txt"), noisy)
    with open(os.path.join(out_dir, "dm_expected.txt"), "w") as f:
        for i in range(n_sym):
            f.write("".join(str(int(b)) for b in exp_bits[i * bps:(i + 1) * bps]) + "\n")

    meta = {"scheme": scheme, "bits_per_symbol": bps, "n_sym": n_sym,
            "sym_peak": float(np.max(np.abs(noisy)))}
    with open(os.path.join(out_dir, "dm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def emit_demap() -> None:
    print("\nDemapper golden data (python Modem.demodulate, graded noise):")
    for scheme in ("qpsk", "qam16", "qam64"):
        m = emit_demap_case(scheme, os.path.join(OUT, f"dm_{scheme}"))
        print(f"  dm_{scheme}: {m['n_sym']} symbols x {m['bits_per_symbol']} bits  "
              f"peak={m['sym_peak']:.3f}")


# ---------------------------------------------------------------- CRC32

def emit_crc() -> None:
    """Golden CRC vectors from spectracuda's own CRC class.

    Uses whatever scheme CFG names -- crc16 for the RF-validated config.
    There is deliberately NO zlib cross-check here: crc.py:17-19 is
    explicit that liquid's crc8/crc16/crc24 are NOT the standard named
    variants (they run the original polynomials through the same 32-bit
    reflected reduction the 32-bit one uses), so zlib would disagree and
    be right to. spectracuda IS the reference.

    Lengths are varied on purpose: a CRC that is right for one length and
    wrong for another usually has a broken init value or a missing final
    complement, and a single-length test cannot tell.
    """
    from spectracuda.fec.crc import CRC
    scheme = CFG["crc"]
    out_dir = os.path.join(OUT, "crc")
    os.makedirs(out_dir, exist_ok=True)
    c = CRC(scheme)
    rng = np.random.default_rng(3)

    # .tobytes() on a uint8 array -- NOT bytes(int64_array), which walks
    # the raw 8-byte-per-element buffer and silently made every "n byte"
    # message 8n bytes of mostly zeros.
    msgs = [bytes(range(256)), b"", b"\x00", b"\xff", b"123456789"]
    for n in (1, 7, 64, 255, 1000):
        msgs.append(rng.integers(0, 256, n, dtype=np.uint8).tobytes())

    lines, meta = [], []
    for m in msgs:
        arr = (np.frombuffer(m, dtype=np.uint8)[None, :] if m
               else np.zeros((1, 0), np.uint8))
        key = int(np.asarray(c.generate_key(arr)).ravel()[0])
        lines.append(" ".join(f"{b:02x}" for b in m))
        meta.append({"len": len(m), "crc": key})

    with open(os.path.join(out_dir, "crc_msgs.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "crc_meta.json"), "w") as f:
        json.dump({"scheme": scheme, "cases": meta}, f, indent=2)
    print(f"\nCRC golden data (spectracuda CRC({scheme!r})):")
    for m in meta:
        print(f"  {m['len']:>5} bytes -> 0x{m['crc']:04x}")


def emit_deint() -> None:
    """Golden vectors for the block de-interleaver.

    Built through spectracuda's OWN BlockInterleaver, and at several
    n_bits values, because the permutation depends on the block size:
    M = 1 + floor(sqrt(n_units)) and N = ceil(n_units/M), so a grid that
    divides evenly and one that leaves virtual padding cells exercise
    different paths. The padding case is the one that breaks naive
    implementations -- those cells must be SKIPPED, not zero-filled.
    """
    from spectracuda.interleaver.block import BlockInterleaver
    out_dir = os.path.join(OUT, "deint")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(11)
    unit_bits = 8

    cases = []
    for n_units in (16, 64, 100, 255, 512, 1000):
        n_bits = n_units * unit_bits
        il = BlockInterleaver(n_bits, unit_bits=unit_bits)
        M = 1 + int(np.floor(np.sqrt(n_units)))
        N = -(-n_units // M)
        data = rng.integers(0, 256, n_units, dtype=np.uint8)
        bits = np.unpackbits(data)[None, :]
        inter = np.asarray(il.encode(bits))[0]
        back = np.asarray(il.decode(inter[None, :]))[0]
        assert np.array_equal(back, bits[0]), "round trip broken"
        cases.append({
            "n_units": n_units, "rows": M, "cols": N,
            "covers_exactly": bool(M * N == n_units),
            "interleaved": np.packbits(inter).tolist(),
            "original": data.tolist(),
        })

    with open(os.path.join(out_dir, "deint_cases.json"), "w") as f:
        json.dump({"unit_bits": unit_bits, "cases": cases}, f)
    print("\nDe-interleaver golden data (python BlockInterleaver):")
    for c in cases:
        pad = c["rows"] * c["cols"] - c["n_units"]
        note = "grid covers exactly" if pad == 0 else f"{pad} virtual cells to skip"
        print(f"  n_units={c['n_units']:>4}  M={c['rows']:>3} N={c['cols']:>3}  {note}")



if __name__ == "__main__":
    main()
    emit_cfo()
    emit_fft()
    emit_chanest()
    emit_eq()
    emit_demap()
    emit_crc()
    emit_deint()
