#!/usr/bin/env python3
"""Fixed PHYSICAL delay spread across sample rates (100 ns echo).

Characterization only; production PHY unchanged (commit 777d4ea).

examples/dmrs_doppler_rate_study.py held the echo at ONE SAMPLE at both
rates, so its physical spread halved (100 ns -> 50 ns) as the rate
doubled. That kept the frequency-domain notch identical, which is what
made the refresh-period comparison clean -- but it means nothing there
speaks to a FIXED physical echo. This run fixes the echo at 100 ns and
lets the sample count follow:

    10 MSps -> 1 sample
    20 MSps -> 2 samples

That changes the channel's frequency selectivity, not just its timing:

    H[k,t] = 1 + a*e^{j2pi*delta_f*t} * e^{-j2pi*k*delay/N}

so `delay` sets how many times the notch sweeps ACROSS the band. At
delay=1 the phase ramps once over the 256 bins; at delay=2 it ramps
twice, i.e. two notches instead of one. A per-subcarrier LS estimate has
no trouble representing either -- but the DMRS refresh is a zero-order
hold on that shape, so a shape with more structure ages differently.

Two things are therefore under test:

  1. Does the matched-time equivalence survive a matched PHYSICAL echo?
     10 MSps iv=16 (544us, delay=1) vs 20 MSps iv=32 (528us, delay=2)
     now match on BOTH refresh period and physical delay spread.

  2. Is timing_advance=2 still enough? Schmidl-Cox tracks the energy
     centroid, and a 2-sample echo pulls it further right than a
     1-sample one. start_index is recorded per cell to show whether the
     detector now lands beyond what a 2-sample advance absorbs. CP=64 is
     3.2 us at 20 MSps against a 100 ns spread, so there is plenty of
     cyclic-prefix room -- the question is purely where sync points.

Control rows keep 20 MSps at delay=1 (50 ns) so the delay change can be
read separately from the rate change.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import dmrs_doppler_study as H
import dmrs_doppler_postfix_study as P
import dmrs_doppler_15db_study as S15

OUT = Path("debug/dmrs_doppler_delay")
N, CP = 256, 64
PAYLOAD_BYTES = P.PAYLOAD_BYTES
F_LOS = 1600.0
DFS = (0, 100, 200, 300, 400, 500, 600)
TAIL, NOISE_LEN = 4096, 300_000

#: (fs, iv, delay_samples, label)
COMBOS = [
    (10e6, 16, 1, "10MS 544us  100ns (reference)"),
    (20e6, 32, 2, "20MS 528us  100ns (matched pair)"),
    (20e6, 32, 1, "20MS 528us   50ns (delay control)"),
    (20e6, 16, 2, "20MS 272us  100ns (margin option)"),
    (20e6, 16, 1, "20MS 272us   50ns (delay control)"),
]


def channel(tx, fs, delay, a=0.2, f_los=0.0, f_echo=0.0, snr_db=15.0, seed=0):
    """Delegates to the shared harness channel (itself `sim.Channel`).
    `delay` is in SAMPLES, so a fixed 100 ns echo is 1 sample at 10 MSps
    and 2 at 20 MSps -- which is the whole point of this study."""
    return H.channel(tx, a=a, f_los=f_los, f_echo=f_echo, delay=delay,
                     snr_db=snr_db, seed=seed, fs=fs)


def cell(fs, iv, delay, delta_f, snr=15.0, a=0.2, trials=300,
         stat_trials=60, seed0=5000, timing_advance=2):
    ok = 0
    evm_m, evm_w, cfos, starts = [], [], [], []
    raws, posts, cws = [], [], []
    bits = np.random.default_rng(1).integers(0, 2, size=(1, PAYLOAD_BYTES*8)).astype("uint8")
    with H.interval_ctx(iv):
        for t in range(trials):
            o = P.make(iv, crc="crc16", timing_advance=timing_advance)
            rx = channel(o.generate_frame(bits), fs, delay, a=a, f_los=F_LOS,
                         f_echo=F_LOS + delta_f, snr_db=snr, seed=seed0 + t)
            try:
                r = o.rx_process(rx)
                ok += int(bool(np.asarray(r["crc_valid"])[0]))
            except ValueError:
                pass
            d = o._last_payload_symbol_diagnostics
            e = np.asarray(d["evm_per_symbol"])
            evm_m.append(e.mean()); evm_w.append(e.max())
            starts.append(P._CAP.get("start_index"))
            cfos.append(P._CAP.get("cfo"))
            if t < stat_trials and "pk" in P._CAP:
                try:
                    raw, post, cw = S15.rs_detail(P._CAP["pk"], P._CAP["bits_in"], bits)
                    raws.append(raw); posts.append(post); cws.append(cw)
                except Exception:
                    pass
    cw = np.concatenate(cws) if cws else np.array([np.nan])
    su = (N + CP) / fs * 1e6
    return dict(
        fs_msps=fs/1e6, iv=iv, delay=delay, delay_ns=delay/fs*1e9,
        nominal_us=iv*su, true_dT_us=(iv+1)*su, delta_f=delta_f, snr=snr,
        trials=trials, passed=ok, per=100.0*(1-ok/trials),
        timing_advance=timing_advance,
        evm_mean=float(np.mean(evm_m)), evm_worst=float(np.mean(evm_w)),
        raw_ber=float(np.mean(raws)) if raws else float("nan"),
        post_vit_ber=float(np.mean(posts)) if posts else float("nan"),
        rs_bytes_mean=float(np.nanmean(cw)), rs_bytes_max=float(np.nanmax(cw)),
        rs_frac_over16=float(np.nanmean(cw > 16)),
        overhead_pct=100.0/(iv+1),
        cfo_hz=float(np.mean(cfos))*fs/N,
        start_index=dict(Counter(starts)),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    partial = OUT / "results_partial.json"
    rows = json.loads(partial.read_text()) if partial.exists() else []
    done = {(r["fs_msps"], r["iv"], r["delay"], r["delta_f"], r["timing_advance"])
            for r in rows}
    t0 = time.time()

    def run(fs, iv, delay, df, adv=2):
        if (fs/1e6, iv, delay, df, adv) in done:
            return
        r = cell(fs, iv, delay, df, timing_advance=adv)
        rows.append(r)
        partial.write_text(json.dumps(rows, indent=1, default=float))
        print(f"[{time.time()-t0:6.0f}s] {fs/1e6:4.0f}MS iv={iv:2d} "
              f"delay={delay} ({r['delay_ns']:.0f}ns) adv={adv} df={df:3d} -> "
              f"{r['passed']}/{r['trials']} EVM {r['evm_mean']:.3f} "
              f"start={r['start_index']}", flush=True)

    for fs, iv, delay, _label in COMBOS:
        for df in DFS:
            run(fs, iv, delay, df)

    # Does a 2-sample echo need more than a 2-sample advance? Swept at the
    # operating point rather than across the whole matrix.
    for adv in (0, 1, 2, 3, 4, 6):
        for df in (0, 400):
            run(20e6, 32, 2, df, adv=adv)

    (OUT / "results.json").write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {OUT/'results.json'}")


if __name__ == "__main__":
    main()
