#!/usr/bin/env python3
"""How many LLR bits does soft-decision Viterbi actually need?

Characterization only; production PHY unchanged and soft_decision stays
off by default.

This is a HARDWARE SIZING question, not a software one. An FPGA Viterbi's
branch-metric and path-metric widths follow directly from the LLR width,
so the interesting number is the smallest quantization that keeps the
coding gain measured in
docs/2026-09-21-multipath-severity-characterization.md.

Two knobs, both mattering:
  * llr_bits  -- signed quantization, 2L+1 levels with L = 2^(b-1)-1.
                 A level AT zero is kept deliberately: that is the "no
                 information" symbol a faded subcarrier must be able to
                 emit.
  * llr_clip  -- what those levels SPAN, in nats. Bit width alone does
                 not determine performance; too tight a clip throws away
                 the confident bits, too loose wastes levels on them.

Run against exactly the channel seeds the multipath study characterized,
so every cell has a known hard-decision and float-soft reference.
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

OUT = Path("debug/soft_llr_quantization")
FS = 20e6
PAYLOAD_BYTES = 5575
TAIL, NOISE_LEN = 4096, 300_000

#: (label, echo amplitude, delay ns, differential Doppler, hard reference)
CELLS = [
    ("a=0.2 clean",            0.2, 100, 0.0),
    ("a=0.6 d=50ns",           0.6, 50, 0.0),
    ("a=0.6 d=200ns",          0.6, 200, 0.0),
    ("a=0.6 d=500ns",          0.6, 500, 0.0),
    ("a=0.8 d=1000ns",         0.8, 1000, 0.0),
    ("a=0.4 d=500ns df=300",   0.4, 500, 300.0),
]


def make(soft, llr_bits=None, llr_clip=6.0, iv=32):
    o = Ofdm(fft_size=256, n_pilot=8, n_data=216, cp_len=64, modem="qam16",
             fec="rs_m8", fec1="conv_v27",
             interleaver="block", interleaver_kwargs={"unit_bits": 8},
             crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
             n_training_symbols=2, dmrs_interval=iv,
             soft_decision=soft, soft_llr_bits=llr_bits, soft_llr_clip=llr_clip)
    o.MAX_PAYLOAD_SYMBOLS = 256
    return o


def run(a, delay_ns, df, soft, llr_bits=None, llr_clip=6.0, trials=40,
        seed0=9000, snr=15.0, iv=32):
    bits = np.random.default_rng(1).integers(
        0, 2, size=(1, PAYLOAD_BYTES * 8)).astype("uint8")
    ok = 0
    with H.interval_ctx(iv):
        for t in range(trials):
            rng = np.random.default_rng(seed0 + t)
            paths = [{"amplitude": 1.0, "delay_ns": 0, "doppler_hz": 0.0},
                     {"amplitude": a, "delay_ns": delay_ns,
                      "phase_rad": float(rng.uniform(0, 2 * np.pi)),
                      "doppler_hz": df}]
            taps, dop, _ = Channel.paths_to_taps(paths, FS)
            o = make(soft, llr_bits, llr_clip, iv)
            rx = Channel(snr_db=snr, multipath_taps=taps, tap_doppler_hz=dop,
                         sample_rate_hz=FS, tail_samples=TAIL,
                         noise_draw_len=NOISE_LEN, seed=seed0 + t,
                         backend="numpy").process(o.generate_frame(bits))
            try:
                r = o.rx_process(rx)
                ok += int(bool(np.asarray(r["crc_valid"])[0]))
            except ValueError:
                pass
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--part", default="bits", choices=["bits", "clip"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{args.part}.json"
    rows = json.loads(f.read_text()) if f.exists() else []
    done = {(r["label"], r.get("setting")) for r in rows}
    t0 = time.time()

    if args.part == "bits":
        settings = [("hard", dict(soft=False)),
                    ("2-bit", dict(soft=True, llr_bits=2)),
                    ("3-bit", dict(soft=True, llr_bits=3)),
                    ("4-bit", dict(soft=True, llr_bits=4)),
                    ("5-bit", dict(soft=True, llr_bits=5)),
                    ("6-bit", dict(soft=True, llr_bits=6)),
                    ("float(8)", dict(soft=True, llr_bits=None))]
    else:
        settings = [(f"clip={c}", dict(soft=True, llr_bits=4, llr_clip=c))
                    for c in (2.0, 3.0, 4.0, 6.0, 10.0, 20.0)]

    for label, a, dns, df in CELLS:
        if args.part == "clip" and label == "a=0.2 clean":
            continue
        for name, kw in settings:
            if (label, name) in done:
                continue
            ok = run(a, dns, df, trials=args.trials, **kw)
            rows.append(dict(label=label, setting=name, a=a, delay_ns=dns,
                             delta_f=df, trials=args.trials, passed=ok, **{
                                 k: v for k, v in kw.items() if k != "soft"}))
            f.write_text(json.dumps(rows, indent=1, default=float))
            print(f"[{time.time()-t0:6.0f}s] {label:>22} {name:>9} -> "
                  f"{ok}/{args.trials}", flush=True)
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
