#!/usr/bin/env python3
"""Differential-Doppler characterization re-run on the FIXED receiver
(timing_advance=2 + float64 sync accumulators, commit 64c4de6).

Characterization only. No production PHY behaviour is changed: the sole
harness extensions are the ones examples/dmrs_doppler_study.py already
documents (independent LOS/echo Doppler; DMRS intervals outside the 2-bit
wire codes, re-pointed for the duration of a run and restored on exit;
MAX_PAYLOAD_SYMBOLS raised on the INSTANCE so short intervals fit).

Three quantities are kept strictly separate and never converted into one
another:
  * UAV velocity                     -- never appears here at all;
  * absolute/common Doppler f_los    -- rides on BOTH paths, absorbed by
                                        CFO + per-symbol CPE;
  * differential Doppler delta_f     -- f_echo - f_los, the ONLY quantity
                                        that ages H[k].
f_los = 1600 Hz is used throughout Part 3 onward as a realistic common
term; delta_f is swept independently. 300 Hz delta_f is an engineering
test point, NOT a conversion of any speed.

TRUE vs NOMINAL DMRS spacing: framing/dmrs.py defines interval=n as "a
DMRS after every n DATA symbols", so consecutive DMRS sit (n+1) slots
apart. Measured from real frames: nominal 128/256/512/1024 us are
actually 160/288/544/1056 us. Analytic work below uses the TRUE value.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from spectracuda.pipeline import Ofdm
from spectracuda.framing import dmrs as D

import dmrs_doppler_study as H     # channel/dmrs_codes/interval_ctx/stage_stats

FS, N, CP = H.FS, H.N, H.CP
SLOT_US = H.SLOT_US
PAYLOAD_BYTES = 5575
IVS = ((4, 128), (8, 256), (16, 512), (32, 1024))
OUT = Path("debug/dmrs_doppler_postfix")

_CAP = {}
_orig_hdr = Ofdm._decode_header_from_sync
_orig_pay = Ofdm._decode_payload_from_header


def _hdr(self, rx_iq, start_index, n_payload_symbols=None):
    """Capture start_index + CFO even for frames that later raise on an
    uncorrectable RS codeword -- the failing cells are the interesting
    ones, so their diagnostics have to survive."""
    h = _orig_hdr(self, rx_iq, start_index, n_payload_symbols)
    _CAP["start_index"] = int(np.asarray(start_index).ravel()[0])
    _CAP["cfo"] = float(np.asarray(h["cfo_estimate"]).ravel()[0])
    return h


def _pay(self, rx, pos, hhd, pm, pk, ebc, nps, hhp=None, di=0, c2=0):
    od = pk.decode

    def dec(b):
        _CAP["bits_in"] = np.asarray(b).copy()
        _CAP["pk"] = pk
        return od(b)
    pk.decode = dec
    try:
        return _orig_pay(self, rx, pos, hhd, pm, pk, ebc, nps, hhp, di, c2)
    finally:
        pk.decode = od


Ofdm._decode_header_from_sync = _hdr
Ofdm._decode_payload_from_header = _pay


def make(iv, crc="crc16", timing_advance=2):
    o = Ofdm(fft_size=N, n_pilot=8, n_data=216, cp_len=CP, modem="qam16",
             fec="rs_m8", fec1="conv_v27",
             interleaver="block", interleaver_kwargs={"unit_bits": 8},
             crc=crc, sync="schmidl_cox", cfo="schmidl_cox",
             n_training_symbols=2, dmrs_interval=iv,
             timing_advance=timing_advance)
    o.MAX_PAYLOAD_SYMBOLS = 256
    o.debug_payload_symbols = True
    return o


def cell(iv, f_los, f_echo, a=0.2, snr=25.0, trials=100, seed0=5000,
         crc="crc16", timing_advance=2, stat_trials=20):
    """One (interval, delta_f) cell. Returns every metric Part 4 asks for."""
    ok = 0
    evm_mean, evm_worst, cfos, cpe_mean, cpe_max = [], [], [], [], []
    starts, stats = [], []
    prof_sum, prof_n = None, None
    bits = np.random.default_rng(1).integers(0, 2, size=(1, PAYLOAD_BYTES*8)).astype("uint8")
    with H.interval_ctx(iv):
        for t in range(trials):
            o = make(iv, crc, timing_advance)
            rx = H.channel(o.generate_frame(bits), a=a, f_los=f_los,
                           f_echo=f_echo, snr_db=snr, seed=seed0 + t)
            try:
                r = o.rx_process(rx)
                ok += int(bool(np.asarray(r["crc_valid"])[0]))
            except ValueError:
                pass
            d = o._last_payload_symbol_diagnostics
            e = np.asarray(d["evm_per_symbol"])
            evm_mean.append(e.mean())
            evm_worst.append(e.max())
            cpe = np.abs(np.asarray(d["pilot_cpe_per_symbol"]))
            cpe_mean.append(cpe.mean())
            cpe_max.append(cpe.max())
            starts.append(_CAP.get("start_index"))
            cfos.append(_CAP.get("cfo"))
            w = H.segment_position(e.shape[0], iv)
            if prof_sum is None:
                prof_sum = np.zeros(int(w.max()) + 1)
                prof_n = np.zeros(int(w.max()) + 1)
            for p in range(len(prof_sum)):
                m = w == p
                if m.any():
                    prof_sum[p] += e[m].mean(); prof_n[p] += 1
            if t < stat_trials and "pk" in _CAP:
                try:
                    stats.append(H.stage_stats(_CAP["pk"], _CAP["bits_in"], bits))
                except Exception:
                    pass
    s = np.array(stats) if stats else np.full((1, 5), np.nan)
    prof = (prof_sum / np.maximum(prof_n, 1)).tolist()
    return dict(
        iv=iv, f_los=f_los, f_echo=f_echo, delta_f=f_echo - f_los, a=a, snr=snr,
        trials=trials, passed=ok, per=100.0*(1 - ok/trials),
        evm_mean=float(np.mean(evm_mean)), evm_worst=float(np.mean(evm_worst)),
        raw_ber=float(s[:, 0].mean()), post_vit_ber=float(s[:, 1].mean()),
        rs_bytes_mean=float(s[:, 2].mean()), rs_bytes_max=float(s[:, 2].max()),
        rs_cw_over=float(s[:, 3].mean()), rs_cw_total=float(s[:, 4].mean()),
        cfo_mean=float(np.mean(cfos)), cpe_mean=float(np.mean(cpe_mean)),
        cpe_max=float(np.max(cpe_max)),
        start_index=dict(Counter(starts)), evm_vs_t=prof,
        timing_advance=timing_advance, crc=crc,
    )


def geometry():
    rows = []
    bits = np.zeros((1, PAYLOAD_BYTES*8), "uint8")
    for iv, us in IVS:
        with H.interval_ctx(iv):
            o = make(iv)
            f = np.asarray(o.generate_frame(bits))
            r = o.rx_process(np.concatenate([f, np.zeros((1, 2048), "complex64")], axis=1))
            nd = int(np.asarray(r["n_payload_symbols"]).ravel()[0])
            ndm = D.n_dmrs_symbols(nd, iv)
            at = np.where(D.dmrs_slot_map(nd, iv) == D.DMRS_SLOT)[0]
            rows.append(dict(
                nominal_us=us, interval=iv, data_symbols=nd, dmrs_symbols=ndm,
                total_symbols=nd+ndm, overhead_pct=100*ndm/(nd+ndm),
                airtime_ms=f.shape[-1]/FS*1e3,
                true_dT_us=float(np.diff(at).mean()*SLOT_US) if len(at) > 1 else None,
            ))
    return rows


def analytic(a=0.2, f_los=1600.0):
    """|dH| = 2a|sin(pi*delta_f*dT)| against the receiver's own successive
    DMRS estimates, de-rotated by the common phase between them. dT is the
    TRUE DMRS-to-DMRS spacing. Noiseless, so the drift is pure channel."""
    cap = {}
    orig = Ofdm._estimate_channel_from_dmrs

    def w(self, slots):
        hd, hp = orig(self, slots)
        cap["h"] = np.asarray(hd).copy()
        return hd, hp
    Ofdm._estimate_channel_from_dmrs = w
    rows = []
    bits = np.random.default_rng(1).integers(0, 2, size=(1, PAYLOAD_BYTES*8)).astype("uint8")
    try:
        for iv, us in IVS:
            dT = (iv + 1) * SLOT_US * 1e-6
            for df in (100, 200, 300, 400, 500, 667):
                with H.interval_ctx(iv):
                    o = make(iv)
                    rx = H.channel(o.generate_frame(bits), a=a, f_los=f_los,
                                   f_echo=f_los+df, noiseless=True)
                    try:
                        o.rx_process(rx)
                    except ValueError:
                        pass
                h = cap.get("h")
                m = float("nan")
                if h is not None and h.shape[1] >= 2:
                    ds = []
                    for i in range(h.shape[1]-1):
                        h0, h1 = h[0, i, :], h[0, i+1, :]
                        ph = np.angle(np.vdot(h0, h1))
                        ds.append(np.abs(h1*np.exp(-1j*ph) - h0).mean())
                    m = float(np.mean(ds))
                an = 2*a*abs(np.sin(np.pi*df*dT))
                rows.append(dict(nominal_us=us, true_dT_us=dT*1e6, delta_f=df,
                                 measured=m, analytic=an,
                                 ratio=(m/an if an else None)))
    finally:
        Ofdm._estimate_channel_from_dmrs = orig
    return rows


# -- driver ------------------------------------------------------------

DFS = (0, 50, 100, 200, 300, 400, 500, 667, 800, 1000)
TRANSITION = (200, 300, 400, 500)
F_LOS = 1600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="10 frames/cell smoke run")
    args = ap.parse_args()
    q = args.quick
    n = (lambda full: 10 if q else full)
    OUT.mkdir(parents=True, exist_ok=True)
    res, t0 = {}, time.time()

    def log(*a):
        print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

    res["geometry"] = geometry()
    log("part 9 geometry done")

    # PART 1 -- clean baseline, no Doppler at all
    res["part1"] = [cell(16, 0.0, 0.0, a=a, snr=s, trials=n(100))
                    for a in (0.0, 0.2, 0.3, 0.4) for s in (25.0, 30.0, 40.0)]
    log("part 1 baseline done")

    # PART 2 -- common Doppler only (delta_f = 0), nominal 512 us
    res["part2"] = [cell(16, f, f, trials=n(100))
                    for f in (0.0, 667.0, 1000.0, 1600.0, 2000.0)]
    log("part 2 common-Doppler control done")

    # PART 3 -- the matrix, f_los = 1600 Hz throughout
    m = []
    for iv, us in IVS:
        for df in DFS:
            tr = n(200 if (iv in (8, 16) and df in TRANSITION) else 100)
            m.append(cell(iv, F_LOS, F_LOS + df, trials=tr))
            log(f"part 3 iv={iv:2d} ({us:4d}us) df={df:4d} -> "
                f"{m[-1]['passed']}/{m[-1]['trials']}")
    res["part3"] = m

    # PART 5 -- BEFORE vs AFTER the timing fix, at the OLD study's exact
    # config (crc32, f_los=667, seeds 5000+t). timing_advance=0 reproduces
    # pre-fix window placement exactly; the float64 sync fix does not enter
    # because the numba path (unaffected) is what runs on this machine.
    ba = []
    for adv in (0, 2):
        for df in (0, 50, 100, 200, 300, 400, 500, 667):
            ba.append(cell(16, 667.0, 667.0 + df, trials=n(100), crc="crc32",
                           timing_advance=adv))
            log(f"part 5 advance={adv} df={df:3d} -> "
                f"{ba[-1]['passed']}/{ba[-1]['trials']}")
    res["part5"] = ba

    # PART 6 -- 300 Hz focus, 500 frames
    res["part6"] = [cell(iv, F_LOS, F_LOS + 300.0, trials=n(500))
                    for iv, _ in IVS if iv != 4]
    log("part 6 300Hz focus done")

    # PART 7 -- same delta_f, very different common Doppler, same seeds
    res["part7"] = [cell(16, 0.0, 300.0, trials=n(200)),
                    cell(16, 1600.0, 1900.0, trials=n(200))]
    log("part 7 common-Doppler equivalence done")

    res["part8"] = analytic()
    log("part 8 analytic done")

    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"wrote {OUT/'results.json'}")


if __name__ == "__main__":
    main()
