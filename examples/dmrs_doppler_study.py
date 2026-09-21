#!/usr/bin/env python3
"""Why 16QAM loses frames under Doppler, and what DMRS interval actually
fixes it -- see docs/2026-09-21-dmrs-differential-doppler-characterization.md.

Nothing here touches the production PHY. Two harness-only extensions make
the design space reachable, both reverted before the process exits:

  1. Independent LOS/echo Doppler. `sim/channel.py` offers a CONSTANT
     carrier offset, which is a pure phase rotation that the per-symbol
     CPE correction already removes -- it cannot age H[k]. The channel
     below rotates the two paths SEPARATELY, so the quantity that ages
     H[k] (the DIFFERENTIAL Doppler delta_f = f_echo - f_los) can be set
     independently of the common Doppler that CFO/CPE handle.

  2. DMRS intervals outside the 2-bit wire codes {0,16,32,64}. The wire
     format is NOT widened: the field still carries four codes. The
     harness only re-points what those four codes MEAN for the duration
     of a run, so 4- and 8-data-symbol intervals can be characterised.
     Short intervals also exceed MAX_PAYLOAD_SYMBOLS=128 at this payload
     (128 us needs 150 slots), so that guard is raised on the INSTANCE.

Five phases, runnable independently:

  --part sync     Why a one-sample sync offset costs ~0.09 EVM although
                  CP=64 should absorb it. Forces the FFT-window offset
                  both EARLY and LATE -- the asymmetry is the answer.

  --part common   Common vs differential Doppler. Proves CFO/CPE absorb
                  common Doppler and that only delta_f ages H[k].

  --part sweep    DMRS interval x differential Doppler, PER + EVM, plus
                  the per-FEC-stage error breakdown (raw demapper BER ->
                  post-Viterbi -> RS byte errors per codeword). This is
                  the table the recommendation rests on. Slowest phase.

  --part analytic Checks |dH| = 2a|sin(pi*delta_f*dT)| against the
                  receiver's OWN successive DMRS estimates, and confirms
                  the result is invariant to common Doppler.

  --part genie    Replaces h_hat with a perfect per-symbol channel to
                  confirm channel aging (not estimation, not the
                  interleaver) is what loses the frames.

NOTE on dT: framing/dmrs.py defines interval=n as "a DMRS after every n
DATA symbols", so consecutive DMRS are (n+1) SLOTS apart. The labels
128/256/512/1024 us are data-symbol spans; the true refresh periods are
160/288/544/1056 us, and the analytic fit needs the true value.
"""
from __future__ import annotations

import argparse
import contextlib

import numpy as np

from spectracuda.pipeline import Ofdm
from spectracuda.pipeline import ofdm as _ofdm_mod
from spectracuda.framing import dmrs as _dmrs
from spectracuda.framing import header as _header

FS, N, CP = 10e6, 256, 64
SLOT_US = (N + CP) / FS * 1e6           # 32.0 us
TAIL = 4096                             # trailing capture samples -- see below
NOISE_LEN = 300_000
PAYLOAD_BYTES = 5575
A_DEFAULT, SNR_DEFAULT = 0.2, 25.0


# -- harness-only capability 1: arbitrary DMRS intervals ----------------

@contextlib.contextmanager
def dmrs_codes(intervals):
    """Re-point the four 2-bit wire codes at `intervals` (code 0 stays
    'off'). Header and pipeline bind these dicts at import time, so all
    three module-level copies have to move together."""
    assert len(intervals) == 3
    codes = {0: 0, 1: intervals[0], 2: intervals[1], 3: intervals[2]}
    inv = {v: k for k, v in codes.items()}
    saved = []
    for mod, name, val in ((_dmrs, "DMRS_PERIOD_CODES", codes),
                           (_dmrs, "DMRS_PERIOD_INTERVALS", inv),
                           (_header, "DMRS_PERIOD_CODES", codes),
                           (_header, "DMRS_PERIOD_INTERVALS", inv),
                           (_ofdm_mod, "_DMRS_PERIOD_INTERVALS", inv)):
        saved.append((mod, name, getattr(mod, name)))
        setattr(mod, name, val)
    try:
        yield
    finally:
        for mod, name, old in saved:
            setattr(mod, name, old)


def interval_ctx(iv):
    """dmrs_codes() only when `iv` is outside the production set."""
    return dmrs_codes((4, 8, 16)) if iv in (4, 8) else contextlib.nullcontext()


def make(dmrs_interval, modem="qam16", max_slots=256):
    o = Ofdm(fft_size=N, n_pilot=8, n_data=216, cp_len=CP, modem=modem,
             fec="rs_m8", fec1="conv_v27",
             interleaver="block", interleaver_kwargs={"unit_bits": 8},
             crc="crc32", sync="schmidl_cox", cfo="schmidl_cox",
             n_training_symbols=2, dmrs_interval=dmrs_interval)
    # Instance attribute, not the class one: the production guard stays
    # at 128 for every other caller in this process.
    o.MAX_PAYLOAD_SYMBOLS = max_slots
    o.debug_payload_symbols = True
    return o


# -- harness-only capability 2: independent LOS/echo Doppler ------------

def channel(tx, a=A_DEFAULT, f_los=0.0, f_echo=0.0, delay=1,
            snr_db=SNR_DEFAULT, seed=0, noiseless=False):
    """rx[n] = e^{j2pi f_los n/fs} tx[n] + a e^{j2pi f_echo n/fs} tx[n-delay]

    delta_f = f_echo - f_los is the differential Doppler. Equal f_los and
    f_echo give a frequency-selective channel that is STATIC after common
    CFO/CPE correction, however large the absolute Doppler is.

    Two methodology details that silently corrupted an earlier round of
    measurements with a predecessor of this helper (see
    docs/2026-09-20-dmrs-static-channel-cost.md):

    1. Trailing samples. Multipath can move the detected frame start a
       sample late; a frame ending flush with the array then needs one
       sample past the end and the bounds check raises, which looks like
       a lost packet but is a simulation boundary. Real captures always
       have samples after the frame.
    2. Length-independent noise. Drawing standard_normal(n) twice makes
       the IMAGINARY part depend on n. DMRS changes the frame length, so
       the same seed would otherwise hand each interval different
       preamble noise -- an apparent interval-dependent PER that is pure
       artifact. Draw at a fixed length and slice.
    """
    tx = np.asarray(tx)
    tx = np.concatenate([tx, np.zeros((tx.shape[0], TAIL), tx.dtype)], axis=1)
    n = np.arange(tx.shape[-1])
    d = np.concatenate([np.zeros((tx.shape[0], delay), tx.dtype),
                        tx[:, :-delay]], axis=1)
    rx = (np.exp(1j * 2 * np.pi * f_los * n / FS)[None, :] * tx
          + a * np.exp(1j * 2 * np.pi * f_echo * n / FS)[None, :] * d)
    if noiseless:
        return rx.astype("complex64")
    rng = np.random.default_rng(seed)
    if rx.shape[-1] > NOISE_LEN:
        raise ValueError("raise NOISE_LEN for this frame size")
    s = np.sqrt(float(np.mean(np.abs(rx) ** 2)) / (2 * 10 ** (snr_db / 10)))
    noise = (rng.standard_normal(NOISE_LEN)
             + 1j * rng.standard_normal(NOISE_LEN)).astype("complex64")
    return (rx + s * noise[:rx.shape[-1]][None, :]).astype("complex64")


@contextlib.contextmanager
def forced_sync(o, index):
    """Override the synchroniser's chosen start_index. NEGATIVE values put
    the FFT window EARLY (inside the CP); positive put it LATE."""
    orig = o.sync.process

    def proc(rx, **kw):
        r = dict(orig(rx, **kw))
        r["start_index"] = np.full_like(np.asarray(r["start_index"]), index)
        return r
    o.sync.process = proc
    try:
        yield
    finally:
        o.sync.process = orig


def payload_bits(seed, n_bytes=PAYLOAD_BYTES):
    return np.random.default_rng(seed).integers(
        0, 2, size=(1, n_bytes * 8)).astype("uint8")


def decode(o, rx, forced=None):
    """Returns (crc_valid, per-symbol EVM, diagnostics). An uncorrectable
    RS codeword raises rather than returning crc_valid=False, and the
    pipeline stashes symbol diagnostics for FAILED frames on the instance
    precisely so that case stays measurable."""
    ctx = forced_sync(o, forced) if forced is not None else contextlib.nullcontext()
    crc, extra = False, {}
    with ctx:
        try:
            r = o.rx_process(rx)
            crc = bool(np.asarray(r["crc_valid"])[0])
            extra = {"start_index": int(np.asarray(r["start_index"]).ravel()[0]),
                     "h": np.asarray(r["channel_estimate"])[0]}
        except ValueError:
            pass
    d = o._last_payload_symbol_diagnostics
    evm = np.asarray(d["evm_per_symbol"]) if d else np.array([np.nan])
    return crc, evm, {**(d or {}), **extra}


def segment_position(n_data_total, interval):
    """Within-segment index of every data symbol (0 = first after a
    refresh), for folding EVM against time-since-DMRS."""
    seg = _dmrs.segment_lengths(n_data_total, interval)
    return np.concatenate([np.arange(L) for L in seg])


# -- part: sync --------------------------------------------------------

def part_sync(args):
    """A cyclic prefix makes the FFT-window position free only in the
    EARLY direction: a window starting inside the CP is a true cyclic
    shift (pure phase ramp, which H[k] absorbs), while a window starting
    LATE runs past the symbol and pulls in samples from the NEXT one.
    That leakage depends on the neighbouring symbol's data, so it is not
    a per-subcarrier transfer function and no channel estimate can
    represent it. The noiseless column below is the proof."""
    print("PART sync -- static channel (f_los = f_echo = 0), a=%.1f, %.0f dB\n"
          % (args.a, args.snr))
    print("negative offset = window EARLY (inside CP); positive = LATE\n")
    print(f"{'offset':>7} {'CRC':>5} {'dataEVM':>8} {'EVM noiseless':>14}")
    for idx in (-4, -2, -1, 0, 1, 2, 4):
        o = make(32)
        b = payload_bits(1000)
        rx = channel(o.generate_frame(b), a=args.a, snr_db=args.snr)
        crc, evm, _ = decode(o, rx, forced=idx)
        o2 = make(32)
        rxn = channel(o2.generate_frame(b), a=args.a, noiseless=True)
        _, evmn, _ = decode(o2, rxn, forced=idx)
        print(f"{idx:7d} {('PASS' if crc else 'FAIL'):>5} {evm.mean():8.3f} "
              f"{evmn.mean():14.4f}")

    print("\nH_hat[k] phase ramp -- expected vs measured (should MATCH, i.e.")
    print("the ramp is absorbed and is NOT where the EVM goes):")
    di = np.asarray(make(32).grid.data_indices)
    for idx in (-2, -1, 1, 2):
        o = make(32)
        rx = channel(o.generate_frame(payload_bits(1000)), a=args.a, snr_db=args.snr)
        _, _, d = decode(o, rx, forced=idx)
        if "h" not in d:
            continue
        ph = np.unwrap(np.angle(d["h"]))
        x = di - di.mean()
        slope = float((ph - ph.mean()) @ x / (x ** 2).sum())
        print(f"  offset {idx:+d}: |measured| {abs(slope):.5f} rad/bin, "
              f"|2*pi*dN/N| {abs(2*np.pi*idx/N):.5f}")

    print("\nnatural sync for reference:")
    for a in (0.0, args.a):
        for snr in (25.0, 30.0, 40.0):
            o = make(32)
            rx = channel(o.generate_frame(payload_bits(1000)), a=a, snr_db=snr)
            crc, evm, d = decode(o, rx)
            print(f"  a={a} snr={snr:.0f}: start_index={d.get('start_index')} "
                  f"EVM={evm.mean():.3f} CRC={'PASS' if crc else 'FAIL'}")


# -- part: common ------------------------------------------------------

CASES = ((1, 0.0, 667.0), (2, 667.0, 667.0), (3, 667.0, 767.0),
         (4, 667.0, 867.0), (5, 667.0, 1067.0), (6, 667.0, 1334.0))


def part_common(args):
    """Cases 1 and 6 carry the same delta_f with 0 and 667 Hz of common
    Doppler; case 2 puts 667 Hz on BOTH paths. If CFO/CPE handle common
    Doppler, 1 and 6 agree and 2 looks static."""
    iv = 32
    print(f"PART common -- a={args.a}, 16QAM, {args.snr:.0f} dB, iv={iv} "
          f"({iv*SLOT_US:.0f} us nominal), {args.trials} frames/case\n")
    print(f"{'case':>5} {'f_los':>7} {'f_echo':>7} {'delta_f':>8} {'CRC':>8} "
          f"{'meanEVM':>8} {'max|CPE|':>9}  EVM vs time-since-DMRS")
    for n, fl, fe in CASES:
        ok, evms, cpes, prof = 0, [], [], None
        for t in range(args.trials):
            o = make(iv)
            rx = channel(o.generate_frame(payload_bits(1000 + t)), a=args.a,
                         f_los=fl, f_echo=fe, snr_db=args.snr, seed=t)
            crc, evm, d = decode(o, rx)
            ok += int(crc)
            evms.append(evm.mean())
            cpes.append(np.abs(np.asarray(d["pilot_cpe_per_symbol"])).max())
            if prof is None:
                w = segment_position(evm.shape[0], iv)
                prof = [evm[w == p].mean() for p in (0, 7, 15, 23, 31) if (w == p).any()]
        print(f"{n:5d} {fl:7.0f} {fe:7.0f} {fe-fl:8.0f} {ok:3d}/{args.trials:<4d} "
              f"{np.mean(evms):8.3f} {np.mean(cpes):9.3f}  "
              + " ".join(f"{v:.3f}" for v in prof))


# -- part: sweep (+ FEC stage breakdown) -------------------------------

DFS = (0, 50, 100, 200, 300, 400, 500, 667)
IVS = ((4, 128), (8, 256), (16, 512), (32, 1024))

_CAP = {}
_orig_payload = Ofdm._decode_payload_from_header


def _capture_payload(self, rx, pos, hhd, pm, pk, ebc, nps, hhp=None, di=0, c2=0):
    """Capture the packetizer's input bits -- the demapper output, before
    Viterbi -- so the error distribution can be measured at each FEC
    stage without changing the pipeline."""
    od = pk.decode

    def dec(b):
        _CAP["bits_in"] = np.asarray(b).copy()
        _CAP["pk"] = pk
        return od(b)
    pk.decode = dec
    try:
        return _orig_payload(self, rx, pos, hhd, pm, pk, ebc, nps, hhp, di, c2)
    finally:
        pk.decode = od


def stage_stats(pk, bits_in, bits):
    """Rebuild the TX intermediates with the packetizer's OWN codecs and
    compare at each stage. Chain is payload -> CRC -> RS(fec0) ->
    INTERLEAVER -> conv_v27(fec1) -> 16QAM, so the interleaver sits
    BETWEEN the two codes: it spreads Viterbi's burst output across RS
    codewords, it never shields Viterbi from channel bursts."""
    pb = pk._bits_to_bytes(np.asarray(bits))
    bwc = pk._bytes_to_bits(pk.crc_codec.append_key(pb))
    t0 = np.asarray(pk.fec_codec.encode(bwc))               # after RS
    il = pk._get_interleaver(t0.shape[-1])
    t1 = np.asarray(il.encode(t0))                          # after interleave
    t2 = np.asarray(pk.fec1_codec.encode(t1))               # after conv
    n = min(bits_in.shape[-1], t2.shape[-1])
    raw = float((bits_in[0, :n] != t2[0, :n]).mean())
    v = np.asarray(pk.fec1_codec.decode(bits_in))
    m = min(v.shape[-1], t1.shape[-1])
    post = float((v[0, :m] != t1[0, :m]).mean())
    w = np.asarray(il.decode(v))
    m2 = min(w.shape[-1], t0.shape[-1])
    eC = w[0, :m2] != t0[0, :m2]
    nb = m2 // 8
    bb = eC[:nb * 8].reshape(nb, 8).any(axis=1)
    ncw = nb // 255
    cw = bb[:ncw * 255].reshape(ncw, 255).sum(axis=1)       # RS corrects 16
    return raw, post, float(cw.mean()), int((cw > 16).sum()), int(ncw)


def part_sweep(args):
    Ofdm._decode_payload_from_header = _capture_payload
    try:
        res = {}
        for iv, us in IVS:
            with interval_ctx(iv):
                for df in DFS:
                    ok, evms, worst, stats = 0, [], [], []
                    for t in range(args.trials):
                        o = make(iv)
                        b = payload_bits(5000 + t)
                        rx = channel(o.generate_frame(b), a=args.a,
                                     f_los=args.f_los, f_echo=args.f_los + df,
                                     snr_db=args.snr, seed=t)
                        crc, evm, _ = decode(o, rx)
                        ok += int(crc)
                        evms.append(evm.mean())
                        worst.append(evm.max())
                        if t < args.stat_trials and "pk" in _CAP:
                            try:
                                stats.append(stage_stats(_CAP["pk"], _CAP["bits_in"], b))
                            except Exception:
                                pass
                    s = np.array(stats).mean(axis=0) if stats else np.full(5, np.nan)
                    res[(us, df)] = (ok, float(np.mean(evms)),
                                     float(np.mean(worst)), s)
                    print(f"  iv={iv:2d} ({us:4d}us) df={df:3d}Hz -> "
                          f"{ok:3d}/{args.trials} evm {np.mean(evms):.3f}")
    finally:
        Ofdm._decode_payload_from_header = _orig_payload

    print(f"\nPART sweep -- 16QAM, {args.snr:.0f} dB, a={args.a}, "
          f"f_los={args.f_los:.0f} Hz present in EVERY cell, "
          f"{PAYLOAD_BYTES} B, {args.trials} frames/cell")
    for title, pick, fmt in (
            (f"PACKET SUCCESS (of {args.trials})", lambda v: v[0], "{:>7}"),
            ("MEAN EVM", lambda v: v[1], "{:>7.3f}"),
            ("WORST-SYMBOL EVM", lambda v: v[2], "{:>7.3f}")):
        print(f"\n=== {title} ===")
        print(f"{'DMRS':>8} |" + "".join(f"{d:>7}" for d in DFS))
        for iv, us in IVS:
            print(f"{us:6d}us |" + "".join(fmt.format(pick(res[(us, d)])) for d in DFS))
    print("\n=== FEC stage breakdown "
          f"(first {args.stat_trials} frames/cell) ===")
    print(f"{'DMRS':>8} {'df':>5} {'rawBER':>8} {'postVit':>8} "
          f"{'RSbyte/cw':>10} {'cw>16':>9} {'PER%':>6}")
    for iv, us in IVS:
        for d in DFS:
            ok, _, _, s = res[(us, d)]
            print(f"{us:6d}us {d:5d} {s[0]:8.4f} {s[1]:8.4f} {s[2]:10.1f} "
                  f"{s[3]:4.1f}/{s[4]:<4.0f} {100*(1-ok/args.trials):6.1f}")


# -- part: analytic ----------------------------------------------------

_CAP_H = {}
_orig_est = Ofdm._estimate_channel_from_dmrs


def _capture_est(self, slots):
    hd, hp = _orig_est(self, slots)
    _CAP_H["dmrs_h"] = np.asarray(hd).copy()
    return hd, hp


def part_analytic(args):
    """Measured = the receiver's own successive DMRS estimates, de-rotated
    by the common phase between them (that is the f_los term, which
    CFO/CPE handle) so only the frequency-selective drift is left. Run
    noiseless so the drift is pure channel, not estimator noise."""
    Ofdm._estimate_channel_from_dmrs = _capture_est
    try:
        def measured(iv, f_los, f_echo):
            with interval_ctx(iv):
                o = make(iv)
                rx = channel(o.generate_frame(payload_bits(1000)), a=args.a,
                             f_los=f_los, f_echo=f_echo, noiseless=True)
                decode(o, rx)
            h = _CAP_H.get("dmrs_h")
            if h is None or h.shape[1] < 2:
                return float("nan")
            d = []
            for i in range(h.shape[1] - 1):
                h0, h1 = h[0, i, :], h[0, i + 1, :]
                ph = np.angle(np.vdot(h0, h1))
                d.append(np.abs(h1 * np.exp(-1j * ph) - h0).mean())
            return float(np.mean(d))

        print(f"PART analytic -- a={args.a}, noiseless.")
        print("|dH| = 2a|sin(pi*delta_f*dT)|, dT = TRUE DMRS-to-DMRS spacing")
        print("= (interval+1) slots, not the data-symbol span.\n")
        print(f"{'label':>7} {'delta_f':>8} {'true dT':>9} {'measured':>9} "
              f"{'analytic':>9} {'ratio':>7}")
        for iv, us in IVS:
            dT = (iv + 1) * SLOT_US * 1e-6
            for df in (50, 100, 200, 400, 667):
                an = 2 * args.a * abs(np.sin(np.pi * df * dT))
                m = measured(iv, args.f_los, args.f_los + df)
                print(f"{us:5d}us {df:8d} {dT*1e6:8.0f}us {m:9.4f} {an:9.4f} "
                      f"{m/an if an else float('nan'):7.2f}")
        print("\ncontrol -- same delta_f, different COMMON Doppler "
              "(must not matter):")
        print(f"{'f_los':>7} {'f_echo':>7} {'measured':>9}")
        for fl in (0.0, 667.0, 2000.0):
            print(f"{fl:7.0f} {fl+200:7.0f} {measured(16, fl, fl+200):9.4f}")
    finally:
        Ofdm._estimate_channel_from_dmrs = _orig_est


# -- part: genie -------------------------------------------------------

_GENIE = {"on": False, "a": A_DEFAULT, "f_los": 0.0, "f_echo": 0.0}


def _genie_payload(self, rx, pos, hhd, pm, pk, ebc, nps, hhp=None, di=0, c2=0):
    """Hand the equalizer a perfectly-tracked channel, as a RELATIVE
    correction:

        H_genie[m,k] = h_hat_ref(m)[k] * H_true(t_m)[k] / H_true(t_ref)[k]

    An absolute closed-form H_true is NOT comparable to h_hat, because
    h_hat also absorbs the receiver's sync-offset phase ramp. Expressing
    the genie as a ratio cancels every static calibration term and
    isolates exactly the aging. Self-check: at delta_f = 0 the ratio is
    identically 1, so the genie must reproduce the normal path."""
    args = (rx, pos, hhd, pm, pk, ebc, nps, hhp, di, c2)
    if not _GENIE["on"]:
        return _orig_payload(self, *args)
    nd = int(nps)
    sm = _dmrs.dmrs_slot_map(nd, di) if di else np.zeros(nd, "uint8")
    data_at = np.where(sm == _dmrs.DATA_SLOT)[0]
    dmrs_at = np.where(sm == _dmrs.DMRS_SLOT)[0]
    p0 = int(np.asarray(pos).ravel()[0])

    def centre(slot):
        return p0 + slot * (N + CP) + CP + N / 2.0     # FFT-window centre

    t_m = centre(data_at)
    ref = np.array([centre(dmrs_at[dmrs_at < s][-1]) if (dmrs_at < s).any()
                    else centre(-1) for s in data_at], dtype=float)
    a, fl, fe = _GENIE["a"], _GENIE["f_los"], _GENIE["f_echo"]
    # LOS carries f_los, echo carries f_echo; factor out the common term
    # (CFO/CPE handle it) and keep the differential in the echo.
    gm = a * np.exp(1j * 2 * np.pi * (fe - fl) * t_m / FS)
    gr = a * np.exp(1j * 2 * np.pi * (fe - fl) * ref / FS)

    def ratio(idx):
        e = np.exp(-1j * 2 * np.pi * np.asarray(idx)[None, :] / N)
        return ((1.0 + gm[:, None] * e) / (1.0 + gr[:, None] * e)).astype("complex64")

    Rd, Rp = ratio(self.grid.data_indices), ratio(self.grid.pilot_indices)
    eq = self.equalizer
    op = eq.process

    def proc(x, channel_est=None, **kk):
        if channel_est is not None and channel_est.shape == Rd.shape:
            channel_est = np.asarray(channel_est) * Rd
        elif channel_est is not None and channel_est.shape == Rp.shape:
            channel_est = np.asarray(channel_est) * Rp
        return op(x, channel_est=channel_est, **kk)
    eq.process = proc
    try:
        return _orig_payload(self, *args)
    finally:
        eq.process = op


def part_genie(args):
    Ofdm._decode_payload_from_header = _genie_payload
    try:
        def run(genie, df, iv=32):
            _GENIE.update(on=genie, a=args.a, f_los=args.f_los,
                          f_echo=args.f_los + df)
            with interval_ctx(iv):
                o = make(iv)
                rx = channel(o.generate_frame(payload_bits(1000)), a=args.a,
                             f_los=args.f_los, f_echo=args.f_los + df,
                             snr_db=args.snr, seed=0)
                try:
                    crc, evm, _ = decode(o, rx)
                finally:
                    _GENIE["on"] = False
            w = segment_position(evm.shape[0], iv)
            return crc, evm.mean(), evm[w < 2].mean(), evm[w >= iv - 2].mean()

        print(f"PART genie -- a={args.a}, {args.snr:.0f} dB, iv=32\n")
        print(f"{'equalizer':>22} {'delta_f':>8} {'CRC':>5} {'meanEVM':>8} "
              f"{'EVM@start':>10} {'EVM@end':>8}")
        for df in (0, 200, 400, 667):
            for label, g in ((" normal h_hat", False), (" GENIE H_true", True)):
                crc, mean, st, en = run(g, df)
                tag = label + ("  (self-check)" if df == 0 else "")
                print(f"{tag:>22} {df:8d} {('PASS' if crc else 'FAIL'):>5} "
                      f"{mean:8.3f} {st:10.3f} {en:8.3f}")
    finally:
        Ofdm._decode_payload_from_header = _orig_payload


PARTS = {"sync": part_sync, "common": part_common, "sweep": part_sweep,
         "analytic": part_analytic, "genie": part_genie}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", required=True, choices=sorted(PARTS))
    p.add_argument("--a", type=float, default=A_DEFAULT,
                   help="echo amplitude (Rician K = -20log10(a) dB)")
    p.add_argument("--snr", type=float, default=SNR_DEFAULT)
    p.add_argument("--f-los", dest="f_los", type=float, default=667.0,
                   help="common Doppler on BOTH paths (absorbed by CFO/CPE)")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--stat-trials", dest="stat_trials", type=int, default=10,
                   help="frames per cell that also get the FEC breakdown")
    args = p.parse_args()
    PARTS[args.part](args)


if __name__ == "__main__":
    main()
