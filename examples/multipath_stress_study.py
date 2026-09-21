#!/usr/bin/env python3
"""Multipath severity stress test -- characterization only.

Production PHY unchanged. Every channel here is built from the shared
`spectracuda.sim.Channel` (paths_to_taps + tap_doppler_hz); no channel
maths is re-implemented in this file.

Separates three failure mechanisms that a single "it failed" does not:

  1. STATIC frequency-selective fading -- deep spectral nulls. Isolated by
     running delta_f = 0 first, so nothing is ageing.
  2. TIME-VARYING multipath -- differential Doppler staling H[k].
  3. SYNC/TIMING failure -- a strong delayed path dragging the
     Schmidl-Cox energy centroid, which start_index records per cell.

Reference PHY: 20 MSps, fft=256, cp=64 (320-sample symbol = 16.0 us,
78.125 kHz spacing = 802.11ax numerology), 16QAM, 15 dB, 8 pilots, 2
training symbols, timing_advance default (2), 5575 B, rs_m8 + block
interleaver + conv_v27. CP is 3.2 us, so every delay used here is well
inside it.

DELAY QUANTIZATION: at 20 MSps one sample is exactly 50 ns, so the
50/100/200/500/1000 ns delays are EXACT -- no rounding. Phase 4 draws
delays on that same 50 ns grid, and paths_to_taps reports what it
realized rather than implying finer resolution exists.

RANDOM PHASE: the echo's initial phase is drawn per frame from a seeded
RNG, not pinned at zero. At a -> 1.0 the phase decides whether a
subcarrier sits on a deep null or a constructive peak, so a fixed phase
characterizes one lucky channel rather than the condition.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel
from spectracuda.framing import dmrs as D

import dmrs_doppler_15db_study as S15   # rs_detail
import dmrs_doppler_postfix_study as P  # make() + _CAP capture hooks

OUT = Path("debug/multipath_stress")
FS, N, CP = 20e6, 256, 64
SLOT_US = (N + CP) / FS * 1e6           # 16.0 us
PAYLOAD_BYTES = 5575
SNR_DB = 15.0
TAIL, NOISE_LEN = 4096, 300_000
DELAYS_NS = (50, 100, 200, 500, 1000)
AMPS = (0.2, 0.4, 0.6, 0.8, 1.0)

_EQ = {}


def _capture_equalized(o):
    """Grab the equalized payload grid so per-SUBCARRIER EVM can be formed.
    evm_per_symbol collapses frequency away, and a strong reflection can
    leave the average healthy while destroying a localized group of bins."""
    orig = o.equalizer.process

    def proc(x, channel_est=None, **kw):
        y = orig(x, channel_est=channel_est, **kw)
        if channel_est is not None and channel_est.shape[-1] == 216 and channel_est.shape[0] > 1:
            _EQ["eq"] = np.asarray(y).copy()
            _EQ["h"] = np.asarray(channel_est).copy()
        return y
    o.equalizer.process = proc


def _per_subcarrier_evm(eq, modem):
    """Decision-directed per-subcarrier EVM over all payload symbols."""
    flat = eq.reshape(-1)
    bits = modem.demodulate(flat[None, :])
    ideal = np.asarray(modem.modulate(bits)).reshape(eq.shape)
    err = np.abs(eq - ideal) ** 2
    return np.sqrt(err.mean(axis=0) / (np.abs(ideal) ** 2).mean())


def cell(paths_fn, iv=32, delta_f_note=None, trials=300, stat_trials=60,
         seed0=9000, snr=SNR_DB, subcarrier=False, label=""):
    """paths_fn(rng) -> list of path dicts. Called once per frame so the
    echo phase (and, in phase 4, the whole channel) can be randomized."""
    ok = 0
    # Counted apart from CRC failure on purpose: "the frame was never
    # detected" is mechanism 3 (sync/timing), not a decode problem, and a
    # strong equal-amplitude echo can defeat acquisition outright.
    no_frame = 0
    evm_m, evm_w, cfos, cpe_m, cpe_x, starts = [], [], [], [], [], []
    raws, posts, cws = [], [], []
    sc_evm, sc_h, sc_true = [], [], []
    los_p, refl_p, strongest = [], [], []
    bits = np.random.default_rng(1).integers(0, 2, size=(1, PAYLOAD_BYTES * 8)).astype("uint8")
    with P.H.interval_ctx(iv) if hasattr(P, "H") else _null():
        pass
    import dmrs_doppler_study as H
    with H.interval_ctx(iv):
        for t in range(trials):
            rng = np.random.default_rng(seed0 + t)
            paths = paths_fn(rng)
            taps, dop, realized = Channel.paths_to_taps(paths, FS)
            a0 = abs(taps[0])
            rest = np.abs(taps[1:]) if taps.shape[-1] > 1 else np.array([0.0])
            los_p.append(a0 ** 2)
            refl_p.append(float((rest ** 2).sum()))
            strongest.append(float(rest.max()) / a0 if a0 else float("nan"))
            o = P.make(iv, crc="crc16", timing_advance=2)
            if subcarrier and t < 20:
                _capture_equalized(o)
            rx = Channel(snr_db=snr, multipath_taps=taps, tap_doppler_hz=dop,
                         sample_rate_hz=FS, tail_samples=TAIL,
                         noise_draw_len=NOISE_LEN, seed=seed0 + t,
                         backend="numpy").process(o.generate_frame(bits))
            try:
                r = o.rx_process(rx)
                if not r.get("frame_found", True):
                    no_frame += 1
                ok += int(bool(np.asarray(r["crc_valid"])[0]))
            except ValueError:
                pass
            d = getattr(o, "_last_payload_symbol_diagnostics", None)
            if d is None:          # sync never reached payload decode
                no_frame += 1
                continue
            e = np.asarray(d["evm_per_symbol"])
            evm_m.append(e.mean()); evm_w.append(e.max())
            cpe = np.abs(np.asarray(d["pilot_cpe_per_symbol"]))
            cpe_m.append(cpe.mean()); cpe_x.append(cpe.max())
            starts.append(P._CAP.get("start_index"))
            cfos.append(P._CAP.get("cfo"))
            if subcarrier and t < 40 and "eq" in _EQ:
                # PER-FRAME statistics. The echo phase is redrawn every
                # frame, so each frame's nulls sit on different
                # subcarriers; averaging the SPECTRA first would flatten
                # them into nothing (an a=1.0 channel would report a
                # 3 dB ripple instead of a true null). Reduce each frame
                # to scalars against ITS OWN nulls, then average those.
                ev = _per_subcarrier_evm(_EQ["eq"], o.modem)
                hh = np.abs(_EQ["h"][0])
                di = np.asarray(o.grid.data_indices)
                ht = np.zeros(len(di), dtype="complex128")
                nz = np.flatnonzero(taps)
                for k_s in nz:
                    ht += taps[k_s] * np.exp(-1j * 2 * np.pi * di * k_s / N)
                ht = np.abs(ht)
                order = np.argsort(ht)
                dec = max(1, len(ht) // 10)
                sc_evm.append(dict(
                    corr=float(np.corrcoef(ev, ht)[0, 1]),
                    null_db=float(20 * np.log10(ht.max() / max(ht.min(), 1e-9))),
                    h_min=float(ht.min()), h_max=float(ht.max()),
                    evm_deep=float(ev[order[:dec]].mean()),
                    evm_peak=float(ev[order[-dec:]].mean()),
                    h_shape_err=float(np.abs(hh / hh.mean() - ht / ht.mean()).mean()),
                    h_shape_err_deep=float(
                        np.abs(hh / hh.mean() - ht / ht.mean())[order[:dec]].mean()),
                ))
                _EQ.clear()
            if t < stat_trials and "pk" in P._CAP:
                try:
                    raw, post, cw = S15.rs_detail(P._CAP["pk"], P._CAP["bits_in"], bits)
                    raws.append(raw); posts.append(post); cws.append(cw)
                except Exception:
                    pass
    cw = np.concatenate(cws) if cws else np.array([np.nan])
    out = dict(
        label=label, iv=iv, nominal_us=iv * SLOT_US, true_dT_us=(iv + 1) * SLOT_US,
        delta_f=delta_f_note, snr=snr, trials=trials, passed=ok,
        per=100.0 * (1 - ok / trials),
        no_frame=no_frame, no_frame_pct=100.0 * no_frame / trials,
        evm_mean=float(np.mean(evm_m)) if evm_m else float("nan"),
        evm_worst=float(np.mean(evm_w)) if evm_w else float("nan"),
        raw_ber=float(np.mean(raws)) if raws else float("nan"),
        post_vit_ber=float(np.mean(posts)) if posts else float("nan"),
        rs_bytes_mean=float(np.nanmean(cw)), rs_bytes_max=float(np.nanmax(cw)),
        rs_frac_over16=float(np.nanmean(cw > 16)),
        cfo_hz=float(np.mean(cfos)) * FS / N if cfos else float("nan"),
        cpe_mean=float(np.mean(cpe_m)) if cpe_m else float("nan"),
        cpe_max=float(np.max(cpe_x)) if cpe_x else float("nan"),
        start_index=dict(Counter(starts)),
        los_power=float(np.mean(los_p)),
        reflected_power=float(np.mean(refl_p)),
        strongest_echo_ratio=float(np.mean(strongest)),
    )
    if sc_evm:
        out["sc"] = {k: float(np.mean([d[k] for d in sc_evm])) for k in sc_evm[0]}
        out["sc"]["n_frames"] = len(sc_evm)
    return out


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def two_path(a, delay_ns, delta_f=0.0):
    def fn(rng):
        return [{"amplitude": 1.0, "delay_ns": 0, "doppler_hz": 0.0},
                {"amplitude": a, "delay_ns": delay_ns,
                 "phase_rad": float(rng.uniform(0, 2 * np.pi)),
                 "doppler_hz": delta_f}]
    return fn


def random_taps(n_lo=3, n_hi=6, a_lo=0.1, a_hi=0.8, d_lo=50, d_hi=1000, df=300.0):
    """Ensemble channel: LOS plus 2-5 reflections on the 50 ns grid, each
    with independent amplitude, phase and differential Doppler. Delays are
    drawn WITHOUT replacement so no two paths collide on one tap."""
    def fn(rng):
        n_refl = int(rng.integers(n_lo - 1, n_hi))
        grid = np.arange(d_lo, d_hi + 1, 50)
        chosen = rng.choice(grid, size=min(n_refl, len(grid)), replace=False)
        paths = [{"amplitude": 1.0, "delay_ns": 0, "doppler_hz": 0.0}]
        for dns in chosen:
            paths.append({
                "amplitude": float(rng.uniform(a_lo, a_hi)),
                "delay_ns": int(dns),
                "phase_rad": float(rng.uniform(0, 2 * np.pi)),
                "doppler_hz": float(rng.uniform(-df, df)),
            })
        return paths
    return fn


def per_channel_ensemble(iv, n_channels=100, frames=20, seed0=4000, **kw):
    """Phase 4: a PER per CHANNEL, so the report can give a distribution
    across channel realizations rather than one pooled number. Each
    channel is fixed for its `frames` frames; only noise varies."""
    gen = random_taps(**kw)
    rows = []
    import dmrs_doppler_study as H
    bits = np.random.default_rng(1).integers(0, 2, size=(1, PAYLOAD_BYTES * 8)).astype("uint8")
    with H.interval_ctx(iv):
        for c in range(n_channels):
            paths = gen(np.random.default_rng(seed0 + c))
            taps, dop, _ = Channel.paths_to_taps(paths, FS)
            rest = np.abs(taps[1:]) if taps.shape[-1] > 1 else np.array([0.0])
            ok, evms = 0, []
            for f in range(frames):
                o = P.make(iv, crc="crc16", timing_advance=2)
                rx = Channel(snr_db=SNR_DB, multipath_taps=taps, tap_doppler_hz=dop,
                             sample_rate_hz=FS, tail_samples=TAIL,
                             noise_draw_len=NOISE_LEN, seed=seed0 * 10 + c * 100 + f,
                             backend="numpy").process(o.generate_frame(bits))
                try:
                    r = o.rx_process(rx)
                    ok += int(bool(np.asarray(r["crc_valid"])[0]))
                except ValueError:
                    pass
                dd = getattr(o, "_last_payload_symbol_diagnostics", None)
                if dd is not None:
                    evms.append(np.asarray(dd["evm_per_symbol"]).mean())
            rows.append(dict(
                channel=c, iv=iv, frames=frames, passed=ok,
                per=100.0 * (1 - ok / frames),
                evm_mean=float(np.mean(evms)) if evms else float("nan"),
                n_paths=int(np.count_nonzero(taps)),
                reflected_power=float((rest ** 2).sum()),
                strongest_echo_ratio=float(rest.max()),
                max_delay_ns=float(np.flatnonzero(taps).max() * 1e9 / FS),
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["2", "3", "4"])
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--channels", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"phase{args.phase}.json"
    rows = json.loads(f.read_text()) if f.exists() else []
    done = {r["label"] for r in rows if "label" in r}
    t0 = time.time()

    def save():
        f.write_text(json.dumps(rows, indent=1, default=float))

    if args.phase == "2":
        # Static frequency-selective only: delta_f = 0 everywhere.
        for a in AMPS:
            for dns in DELAYS_NS:
                lab = f"a={a}_d={dns}ns"
                if lab in done:
                    continue
                r = cell(two_path(a, dns), iv=32, delta_f_note=0.0,
                         trials=args.trials, subcarrier=True, label=lab)
                rows.append(r); save()
                print(f"[{time.time()-t0:6.0f}s] {lab:>18} -> {r['passed']}/{r['trials']} "
                      f"EVM {r['evm_mean']:.3f} rawBER {r['raw_ber']:.4f} "
                      f"start={r['start_index']}", flush=True)

    elif args.phase == "3":
        for a in (0.4, 0.6, 0.8):
            for dns in (200, 500):
                for df in (100.0, 300.0):
                    for iv in (32, 16):
                        lab = f"a={a}_d={dns}ns_df={df:.0f}_iv={iv}"
                        if lab in done:
                            continue
                        r = cell(two_path(a, dns, df), iv=iv, delta_f_note=df,
                                 trials=args.trials, label=lab)
                        rows.append(r); save()
                        print(f"[{time.time()-t0:6.0f}s] {lab:>28} -> "
                              f"{r['passed']}/{r['trials']} EVM {r['evm_mean']:.3f}",
                              flush=True)

    else:
        for iv in (32, 16):
            lab = f"ensemble_iv={iv}"
            if any(r.get("label") == lab for r in rows):
                continue
            got = per_channel_ensemble(iv, n_channels=args.channels)
            rows.append(dict(label=lab, iv=iv, channels=got))
            save()
            pers = np.array([g["per"] for g in got])
            print(f"[{time.time()-t0:6.0f}s] iv={iv}: median PER {np.median(pers):.1f}% "
                  f"p90 {np.percentile(pers,90):.1f}% worst {pers.max():.1f}% "
                  f"clean {(pers==0).mean()*100:.0f}%", flush=True)

    save()
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
