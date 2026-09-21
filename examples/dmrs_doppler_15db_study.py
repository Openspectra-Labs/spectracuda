#!/usr/bin/env python3
"""Differential-Doppler characterization at SNR = 15 dB, on the fixed
receiver (commit 777d4ea / 64c4de6). Characterization only -- no
production PHY change; the harness extensions are the ones
examples/dmrs_doppler_study.py documents.

Why this run exists: the 25 dB study (debug/dmrs_doppler_postfix/) found
nominal 512 us clean to ~500 Hz of DIFFERENTIAL Doppler. At 15 dB the
AWGN floor alone costs ~0.22 mean EVM, against a chain that turns
unreliable around raw demapper BER 0.018-0.024 -- so the delta_f = 0
baseline has to be measured FIRST and every later failure reported as an
INCREMENT over it, never as a Doppler failure on its own.

Adds per-codeword RS detail the 25 dB harness did not keep: the true
maximum erroneous bytes in any single codeword, and the fraction of
codewords past RS(255,223)'s 16-byte correction capability.

Three quantities stay separate and are never converted into one another:
UAV velocity (absent here), absolute/common Doppler (f_los, on BOTH
paths, absorbed by CFO+CPE) and differential Doppler (delta_f, the only
term that ages H[k]).
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from spectracuda.pipeline import Ofdm

import dmrs_doppler_study as H
import dmrs_doppler_postfix_study as P

OUT = Path("debug/dmrs_doppler_15db")
PAYLOAD_BYTES = P.PAYLOAD_BYTES
F_LOS = 1600.0
DFS = (0, 100, 200, 300, 400, 500)
IVS = ((16, 512), (8, 256))


def rs_detail(pk, bits_in, bits):
    """Per-codeword RS load. H.stage_stats keeps only the mean; the cliff
    is decided by the WORST codeword, because every codeword must decode
    for the frame to survive."""
    pb = pk._bits_to_bytes(np.asarray(bits))
    bwc = pk._bytes_to_bits(pk.crc_codec.append_key(pb))
    t0 = np.asarray(pk.fec_codec.encode(bwc))
    il = pk._get_interleaver(t0.shape[-1])
    t1 = np.asarray(il.encode(t0))
    t2 = np.asarray(pk.fec1_codec.encode(t1))
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
    cw = bb[:ncw * 255].reshape(ncw, 255).sum(axis=1)
    return raw, post, cw


def cell(iv, delta_f, snr=15.0, a=0.2, trials=500, stat_trials=100, seed0=5000):
    ok = 0
    evm_mean, evm_worst, cfos, cpe_m, cpe_x, starts = [], [], [], [], [], []
    raws, posts, cw_all = [], [], []
    bits = np.random.default_rng(1).integers(0, 2, size=(1, PAYLOAD_BYTES * 8)).astype("uint8")
    with H.interval_ctx(iv):
        for t in range(trials):
            o = P.make(iv, crc="crc16", timing_advance=2)
            rx = H.channel(o.generate_frame(bits), a=a, f_los=F_LOS,
                           f_echo=F_LOS + delta_f, snr_db=snr, seed=seed0 + t)
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
            cpe_m.append(cpe.mean())
            cpe_x.append(cpe.max())
            starts.append(P._CAP.get("start_index"))
            cfos.append(P._CAP.get("cfo"))
            if t < stat_trials and "pk" in P._CAP:
                try:
                    raw, post, cw = rs_detail(P._CAP["pk"], P._CAP["bits_in"], bits)
                    raws.append(raw); posts.append(post); cw_all.append(cw)
                except Exception:
                    pass
    cw_all = np.concatenate(cw_all) if cw_all else np.array([np.nan])
    return dict(
        iv=iv, nominal_us=dict(IVS)[iv], delta_f=delta_f, snr=snr, a=a,
        f_los=F_LOS, f_echo=F_LOS + delta_f, trials=trials, passed=ok,
        per=100.0 * (1 - ok / trials),
        evm_mean=float(np.mean(evm_mean)), evm_worst=float(np.mean(evm_worst)),
        raw_ber=float(np.mean(raws)) if raws else float("nan"),
        post_vit_ber=float(np.mean(posts)) if posts else float("nan"),
        rs_bytes_mean=float(np.nanmean(cw_all)),
        rs_bytes_max=float(np.nanmax(cw_all)),
        rs_frac_over16=float(np.nanmean(cw_all > 16)),
        rs_codewords=int(cw_all.size),
        cfo_mean=float(np.mean(cfos)), cfo_hz=float(np.mean(cfos)) * 10e6 / 256,
        cpe_mean=float(np.mean(cpe_m)), cpe_max=float(np.max(cpe_x)),
        start_index=dict(Counter(starts)),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Written after EVERY cell, not once at the end: an earlier attempt
    # was killed at cell 9 of 12 and lost all of it.
    partial = OUT / "results_partial.json"
    rows = json.loads(partial.read_text()) if partial.exists() else []
    done = {(r["iv"], r["delta_f"]) for r in rows}
    for iv, us in IVS:
        for df in DFS:
            if (iv, df) in done:
                print(f"skip {us}us df={df} (already have it)", flush=True)
                continue
            r = cell(iv, df)
            rows.append(r)
            partial.write_text(json.dumps(rows, indent=1, default=float))
            print(f"[{time.time()-t0:6.0f}s] {us:4d}us df={df:3d} -> "
                  f"{r['passed']}/{r['trials']}  EVM {r['evm_mean']:.3f}  "
                  f"rawBER {r['raw_ber']:.4f}", flush=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1, default=float))
    print(f"wrote {OUT/'results.json'}")


if __name__ == "__main__":
    main()
