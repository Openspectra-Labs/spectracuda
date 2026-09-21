#!/usr/bin/env python3
"""The combination nothing has been tested against: strong reflections AND
differential Doppler at the same time.

Characterization only; both features stay off by default in the library.

Everything before this isolated one mechanism or the other. Phase 3 of the
multipath study had Doppler plus reflections but neither soft decision nor
interleaver2. The interleaver2 matrix had reflections and both features but
delta_f = 0, so DMRS was inert and nothing was ageing.

They compound rather than add: a STATIC null damages fixed subcarriers,
which interleaver2 disperses; a MOVING null sweeps that damage across the
band as the frame runs, so the stale-H[k] error and the fade land on the
same bits. This runs all four feature combinations against both.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel

import dmrs_doppler_study as H

OUT = Path("debug/worst_case")
FS, SNR, PAYLOAD_BYTES = 20e6, 15.0, 5575
F_LOS = 1600.0          # common Doppler, absorbed by CFO/CPE -- present throughout


def make(soft, il2, iv=32):
    o = Ofdm(fft_size=256, n_pilot=8, n_data=216, cp_len=64, modem="qam16",
             fec="rs_m8", fec1="conv_v27", interleaver="block",
             interleaver_kwargs={"unit_bits": 8}, crc="crc16",
             sync="schmidl_cox", cfo="schmidl_cox", n_training_symbols=2,
             dmrs_interval=iv, soft_decision=soft,
             interleaver2="block" if il2 else "none")
    o.MAX_PAYLOAD_SYMBOLS = 256
    return o


def run(a, delay_ns, delta_f, soft, il2, iv=32, trials=100, seed0=9000):
    bits = np.random.default_rng(1).integers(
        0, 2, size=(1, PAYLOAD_BYTES * 8)).astype("uint8")
    ok = 0
    with H.interval_ctx(iv):
        for t in range(trials):
            rng = np.random.default_rng(seed0 + t)
            taps, dop, _ = Channel.paths_to_taps([
                {"amplitude": 1.0, "delay_ns": 0, "doppler_hz": F_LOS},
                {"amplitude": a, "delay_ns": delay_ns,
                 "phase_rad": float(rng.uniform(0, 2 * np.pi)),
                 "doppler_hz": F_LOS + delta_f}], FS)
            o = make(soft, il2, iv)
            rx = Channel(snr_db=SNR, multipath_taps=taps, tap_doppler_hz=dop,
                         sample_rate_hz=FS, tail_samples=4096,
                         noise_draw_len=300_000, seed=seed0 + t,
                         backend="numpy").process(o.generate_frame(bits))
            try:
                r = o.rx_process(rx)
                ok += int(bool(np.asarray(r["crc_valid"])[0]))
            except ValueError:
                pass
    return ok


CONFIGS = [("baseline", False, False), ("il2", False, True),
           ("soft", True, False), ("both", True, True)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    print(f"15 dB, 16QAM, iv=32, f_los=1600 Hz, {args.trials} frames/cell")
    print(f"{'echo a':>7} {'delay':>7} {'delta_f':>8} |"
          + "".join(f"{n:>10}" for n, _, _ in CONFIGS))
    for a in (0.6, 0.8):
        for dns in (50, 200):
            for df in (0.0, 100.0, 300.0):
                got = {}
                for name, soft, il2 in CONFIGS:
                    got[name] = run(a, dns, df, soft, il2, trials=args.trials)
                rows.append(dict(a=a, delay_ns=dns, delta_f=df,
                                 trials=args.trials, **got))
                print(f"{a:7.1f} {dns:5d}ns {df:7.0f}H |"
                      + "".join(f"{got[n]:>7}/{args.trials:<3}" for n, _, _ in CONFIGS),
                      flush=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1, default=float))
    print(f"\nwrote {OUT/'results.json'}  [{time.time()-t0:.0f}s]")
