#!/usr/bin/env python3
"""Differential Doppler at 10 MSps AND 20 MSps, 320-sample symbol.

Characterization only -- production PHY unchanged (commit 777d4ea).

Why sample rate matters here even though the PHY is identical: the DMRS
interval counts DATA SYMBOLS, not time. A 320-sample symbol is 32.0 us at
10 MSps and 16.0 us at 20 MSps, so the SAME iv is a different refresh
PERIOD at each rate:

    iv     10 MSps        20 MSps
     8      256 us         128 us
    16      512 us         256 us
    32     1024 us         512 us
    64     2048 us        1024 us

"512 us DMRS" is therefore iv=16 at 10 MSps but iv=32 at 20 MSps -- and
iv=32 costs 1/33 (~3%) overhead where iv=16 costs 1/17 (~5.9%). The same
time-domain tracking is cheaper at the higher rate.

Numerology note: at 20 MSps with fft=256/cp=64 the grid is 78.125 kHz
subcarrier spacing, 12.8 us symbol, 3.2 us CP -- the 802.11ax numerology
(3.2 us is 11ax's long guard interval). At 10 MSps it is half that
spacing.

Hypothesis under test: channel aging is governed by delta_f * dT where dT
is the TRUE DMRS-to-DMRS spacing IN TIME, so matched-time intervals
should give matched tolerance at either sample rate. Matched pairs here:

    10 MSps iv=16 (true 544 us)  vs  20 MSps iv=32 (true 528 us)
    10 MSps iv=32 (true 1056 us) vs  20 MSps iv=64 (true 1040 us)

Doppler is applied in the CHANNEL, which is the only place the sample
rate enters: rx[n] = e^{j2pi f n/fs} tx[n] + a e^{j2pi f_echo n/fs}
tx[n-1]. The frame itself is the same sample sequence at both rates.

The echo delay stays ONE SAMPLE at both rates, so its delay spread halves
in time (100 ns -> 50 ns) as the rate doubles. That is deliberate -- it
keeps the frequency-domain notch shape identical so the comparison
isolates the refresh-period effect -- but it means this run does NOT say
anything about tolerance to a fixed physical delay spread.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import dmrs_doppler_study as H
import dmrs_doppler_postfix_study as P

OUT = Path("debug/dmrs_doppler_rate")
N, CP = 256, 64
PAYLOAD_BYTES = P.PAYLOAD_BYTES
F_LOS = 1600.0                 # ~300 km/h one-way LOS Doppler at 5.8 GHz
DFS = (0, 100, 200, 300, 400, 500, 600)
TAIL, NOISE_LEN = 4096, 300_000

#: (fs, iv) combinations -> each gives a nominal refresh period.
COMBOS = [(10e6, 16), (10e6, 32),
          (20e6, 16), (20e6, 32), (20e6, 64)]


def channel(tx, fs, a=0.2, f_los=0.0, f_echo=0.0, snr_db=15.0, seed=0):
    """Delegates to the shared harness channel (itself `sim.Channel`), so
    the tail/noise-draw discipline lives in one place. fs is the only
    place the sample rate enters; the echo stays at ONE SAMPLE here --
    see the module docstring for why that is deliberate."""
    return H.channel(tx, a=a, f_los=f_los, f_echo=f_echo, delay=1,
                     snr_db=snr_db, seed=seed, fs=fs)


def slot_us(fs):
    return (N + CP) / fs * 1e6


def cell(fs, iv, delta_f, snr=15.0, a=0.2, trials=300, stat_trials=60, seed0=5000):
    ok = 0
    evm_m, evm_w, cfos, cpe_m, cpe_x, starts = [], [], [], [], [], []
    raws, posts, cws = [], [], []
    bits = np.random.default_rng(1).integers(0, 2, size=(1, PAYLOAD_BYTES*8)).astype("uint8")
    with H.interval_ctx(iv):
        for t in range(trials):
            o = P.make(iv, crc="crc16", timing_advance=2)
            rx = channel(o.generate_frame(bits), fs, a=a, f_los=F_LOS,
                         f_echo=F_LOS + delta_f, snr_db=snr, seed=seed0 + t)
            try:
                r = o.rx_process(rx)
                ok += int(bool(np.asarray(r["crc_valid"])[0]))
            except ValueError:
                pass
            d = o._last_payload_symbol_diagnostics
            e = np.asarray(d["evm_per_symbol"])
            evm_m.append(e.mean()); evm_w.append(e.max())
            cpe = np.abs(np.asarray(d["pilot_cpe_per_symbol"]))
            cpe_m.append(cpe.mean()); cpe_x.append(cpe.max())
            starts.append(P._CAP.get("start_index"))
            cfos.append(P._CAP.get("cfo"))
            if t < stat_trials and "pk" in P._CAP:
                try:
                    import dmrs_doppler_15db_study as S15
                    raw, post, cw = S15.rs_detail(P._CAP["pk"], P._CAP["bits_in"], bits)
                    raws.append(raw); posts.append(post); cws.append(cw)
                except Exception:
                    pass
    cw = np.concatenate(cws) if cws else np.array([np.nan])
    su = slot_us(fs)
    return dict(
        fs_msps=fs/1e6, iv=iv, nominal_us=iv*su, true_dT_us=(iv+1)*su,
        delta_f=delta_f, snr=snr, a=a, trials=trials, passed=ok,
        per=100.0*(1-ok/trials),
        evm_mean=float(np.mean(evm_m)), evm_worst=float(np.mean(evm_w)),
        raw_ber=float(np.mean(raws)) if raws else float("nan"),
        post_vit_ber=float(np.mean(posts)) if posts else float("nan"),
        rs_bytes_mean=float(np.nanmean(cw)), rs_bytes_max=float(np.nanmax(cw)),
        rs_frac_over16=float(np.nanmean(cw > 16)),
        overhead_pct=100.0/(iv+1),
        cfo_hz=float(np.mean(cfos))*fs/N,
        cpe_mean=float(np.mean(cpe_m)), cpe_max=float(np.max(cpe_x)),
        start_index=dict(Counter(starts)),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    partial = OUT / "results_partial.json"
    rows = json.loads(partial.read_text()) if partial.exists() else []
    done = {(r["fs_msps"], r["iv"], r["delta_f"]) for r in rows}
    t0 = time.time()
    for fs, iv in COMBOS:
        for df in DFS:
            if (fs/1e6, iv, df) in done:
                continue
            r = cell(fs, iv, df)
            rows.append(r)
            partial.write_text(json.dumps(rows, indent=1, default=float))
            print(f"[{time.time()-t0:6.0f}s] {fs/1e6:4.0f}MSps iv={iv:2d} "
                  f"({r['nominal_us']:4.0f}us nom / {r['true_dT_us']:4.0f}us true) "
                  f"df={df:3d} -> {r['passed']}/{r['trials']} "
                  f"EVM {r['evm_mean']:.3f}", flush=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {OUT/'results.json'}")


if __name__ == "__main__":
    main()
