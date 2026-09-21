#!/usr/bin/env python3
"""PHASE 5 -- where do the errors actually live, and does the interleaver
disperse them? Characterization only; FEC/interleaver untouched.

The money comparison is a=0.6 at 50 ns versus a=0.6 at 1000 ns. Both have
the SAME null depth (12.0 dB -- depth is set by echo amplitude alone), the
same echo power, and no Doppler. Yet Phase 2 gives 0/300 at 50 ns and
284/300 at 1000 ns. The only thing that differs is how the fade is spread
across frequency: a 1-sample echo puts ONE broad null across the 256 bins,
a 20-sample echo puts twenty narrow ones.

Chain (verified in framing/packetizer.py): payload -> CRC -> RS(255,223)
-> INTERLEAVER -> conv_v27 -> 16QAM. So RX is Viterbi -> deinterleave ->
RS: the interleaver sits BETWEEN the codes and spreads Viterbi's OUTPUT
across RS codewords. It never shields Viterbi from a channel burst.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel

import dmrs_doppler_study as H
import dmrs_doppler_postfix_study as P
import multipath_stress_study as M

OUT = Path("debug/multipath_stress")


def clustering(mask, win=864):
    """Max error density in a sliding window, relative to the mean.

    Run length alone cannot separate clustered from uniform damage here: a
    faded subcarrier raises error PROBABILITY, it does not make every bit
    wrong, so even a badly faded region yields short runs. What changes is
    local DENSITY. `win` defaults to one OFDM symbol's coded bits
    (216 data subcarriers x 4 bits for 16QAM).
    """
    m = np.asarray(mask).astype(float).ravel()
    if m.sum() == 0 or m.size < win:
        return 1.0
    c = np.concatenate([[0.0], np.cumsum(m)])
    dens = (c[win:] - c[:-win]) / win
    return float(dens.max() / max(m.mean(), 1e-12))


def runs(mask):
    m = np.asarray(mask).astype(int).ravel()
    if m.sum() == 0:
        return 0, 0, 0.0
    d = np.diff(np.concatenate([[0], m, [0]]))
    L = np.where(d == -1)[0] - np.where(d == 1)[0]
    return len(L), int(L.max()), float(L.mean())


def diagnose(a, delay_ns, iv=32, trials=12, seed0=9000):
    """Error distribution at three points in the chain, plus how the damage
    sits across frequency and across OFDM symbols."""
    bits = np.random.default_rng(1).integers(0, 2, size=(1, M.PAYLOAD_BYTES*8)).astype("uint8")
    acc = dict(pre=[], post=[], cw=[], sym=[], sc=[], badsc=[], ok=0, n=0)
    with H.interval_ctx(iv):
        for t in range(trials):
            rng = np.random.default_rng(seed0 + t)
            paths = M.two_path(a, delay_ns)(rng)
            taps, dop, _ = Channel.paths_to_taps(paths, M.FS)
            o = P.make(iv, crc="crc16", timing_advance=2)
            M._capture_equalized(o)
            rx = Channel(snr_db=M.SNR_DB, multipath_taps=taps, tap_doppler_hz=dop,
                         sample_rate_hz=M.FS, tail_samples=M.TAIL,
                         noise_draw_len=M.NOISE_LEN, seed=seed0 + t,
                         backend="numpy").process(o.generate_frame(bits))
            try:
                r = o.rx_process(rx)
                acc["ok"] += int(bool(np.asarray(r["crc_valid"])[0]))
            except ValueError:
                pass
            acc["n"] += 1
            d = getattr(o, "_last_payload_symbol_diagnostics", None)
            if d is not None:
                acc["sym"].append(np.asarray(d["evm_per_symbol"]))
            if "eq" in M._EQ:
                acc["sc"].append(M._per_subcarrier_evm(M._EQ["eq"], o.modem))
                di = np.asarray(o.grid.data_indices)
                ht = np.zeros(len(di), dtype="complex128")
                for k in np.flatnonzero(taps):
                    ht += taps[k]*np.exp(-1j*2*np.pi*di*k/M.N)
                htm = np.abs(ht)
                thr = 0.5 * htm.mean()
                nreg, widest, _ = runs(htm < thr)
                acc["badsc"].append((float((htm < thr).mean()), nreg, widest))
                M._EQ.clear()
            if "pk" in P._CAP:
                pk, bi = P._CAP["pk"], P._CAP["bits_in"]
                pb = pk._bits_to_bytes(np.asarray(bits))
                bwc = pk._bytes_to_bits(pk.crc_codec.append_key(pb))
                t0 = np.asarray(pk.fec_codec.encode(bwc))
                il = pk._get_interleaver(t0.shape[-1])
                t1 = np.asarray(il.encode(t0))
                t2 = np.asarray(pk.fec1_codec.encode(t1))
                n = min(bi.shape[-1], t2.shape[-1])
                acc["pre"].append(bi[0, :n] != t2[0, :n])
                v = np.asarray(pk.fec1_codec.decode(bi))
                m = min(v.shape[-1], t1.shape[-1])
                acc["post"].append(v[0, :m] != t1[0, :m])
                w = np.asarray(il.decode(v))
                m2 = min(w.shape[-1], t0.shape[-1])
                eC = w[0, :m2] != t0[0, :m2]
                nb = m2//8
                bb = eC[:nb*8].reshape(nb, 8).any(axis=1)
                ncw = nb//255
                acc["cw"].append(bb[:ncw*255].reshape(ncw, 255).sum(axis=1))
    sc = np.mean(acc["sc"], axis=0)
    # per-frame, then averaged: the echo phase moves every frame, so
    # averaging the SPECTRA first erases the nulls entirely
    fb = np.array(acc["badsc"]) if acc["badsc"] else np.zeros((1, 3))
    sym = np.mean(acc["sym"], axis=0) if acc["sym"] else np.array([np.nan])
    pre = np.concatenate([p[None, :] for p in acc["pre"]]) if acc["pre"] else None
    post = np.concatenate([p[None, :] for p in acc["post"]]) if acc["post"] else None
    cw = np.concatenate(acc["cw"]) if acc["cw"] else np.array([np.nan])
    out = dict(a=a, delay_ns=delay_ns, iv=iv, trials=acc["n"], passed=acc["ok"])
    if pre is not None:
        c, mx, mn = runs(pre[0]); out["pre_runs"] = dict(n=c, max=mx, mean=mn)
        c, mx, mn = runs(post[0]); out["post_runs"] = dict(n=c, max=mx, mean=mn)
        out["pre_ber"] = float(pre.mean()); out["post_ber"] = float(post.mean())
        out["pre_cluster"] = float(np.mean([clustering(p) for p in pre]))
        out["post_cluster"] = float(np.mean([clustering(p) for p in post]))
    out.update(
        cw_mean=float(np.nanmean(cw)), cw_max=float(np.nanmax(cw)),
        cw_min=float(np.nanmin(cw)), cw_over=float(np.nanmean(cw > 16)),
        cw_std=float(np.nanstd(cw)),
        sym_evm_spread=float(sym.max()/sym.min()) if sym.size > 1 else float("nan"),
        sc_evm_spread=float(sc.max()/sc.min()),
        # how much of the band sits below half the mean |H|, per frame
        frac_band_faded=float(fb[:, 0].mean()),
        n_faded_regions=float(fb[:, 1].mean()),
        widest_faded_run=float(fb[:, 2].mean()),
    )
    return out


if __name__ == "__main__":
    rows = []
    for a, d in ((0.6, 50), (0.6, 200), (0.6, 1000), (0.4, 50), (0.8, 1000)):
        r = diagnose(a, d)
        rows.append(r)
        print(f"a={a} d={d:4d}ns: {r['passed']}/{r['trials']}  "
              f"preBER {r.get('pre_ber',float('nan')):.4f} "
              f"(max run {r.get('pre_runs',{}).get('max','-')})  "
              f"postBER {r.get('post_ber',float('nan')):.4f} "
              f"(max run {r.get('post_runs',{}).get('max','-')})  "
              f"RS cw {r['cw_min']:.0f}/{r['cw_mean']:.1f}/{r['cw_max']:.0f} "
              f"over16 {100*r['cw_over']:.0f}%  "
              f"| faded {100*r['frac_band_faded']:.0f}% of band in "
              f"{r['n_faded_regions']:.1f} region(s), widest {r['widest_faded_run']:.0f} bins "
              f"| cluster pre {r.get('pre_cluster',float('nan')):.2f} "
              f"post {r.get('post_cluster',float('nan')):.2f}",
              flush=True)
    (OUT / "phase5.json").write_text(json.dumps(rows, indent=1, default=float))
    print("wrote", OUT / "phase5.json")
