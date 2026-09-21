#!/usr/bin/env python3
"""Render debug/dmrs_doppler_rate/results.json.

Organised around TRUE refresh period rather than iv, because that is the
quantity the physics depends on -- iv means a different period at each
sample rate. Read-only."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("debug/dmrs_doppler_rate")
R = json.loads((OUT / "results.json").read_text())
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    L.append(s)


def get(fs, iv, df):
    m = [r for r in R if r["fs_msps"] == fs and r["iv"] == iv and r["delta_f"] == df]
    return m[0] if m else None


DFS = sorted({r["delta_f"] for r in R})
COMBOS = sorted({(r["fs_msps"], r["iv"]) for r in R})

P("=" * 92)
P("320-sample symbol (fft=256, cp=64), 16QAM, a=0.2, SNR=15 dB,")
P("f_los=1600 Hz (~300 km/h one-way LOS at 5.8 GHz), 5575 B, 300 frames/cell")
P("=" * 92)
P("")
P("NUMEROLOGY -- iv counts DATA SYMBOLS, so the same iv is a different period")
P(f"{'Fs':>9} {'sc spacing':>12} {'slot':>9} {'iv':>4} {'nominal':>9} "
  f"{'true dT':>9} {'overhead':>9}")
for fs, iv in COMBOS:
    r = get(fs, iv, 0)
    P(f"{fs:7.0f}MS {fs*1e6/256/1e3:9.2f} kHz {(256+64)/(fs*1e6)*1e6:7.2f}us "
      f"{iv:4d} {r['nominal_us']:7.0f}us {r['true_dT_us']:7.0f}us "
      f"{r['overhead_pct']:8.1f}%")

P("")
P("--- PACKET SUCCESS (of 300) ---")
P(f"{'Fs':>8} {'iv':>4} {'nominal':>9} {'true dT':>9} |"
  + "".join(f"{int(d):>8}" for d in DFS))
for fs, iv in COMBOS:
    r0 = get(fs, iv, 0)
    row = (f"{fs:6.0f}MS {iv:4d} {r0['nominal_us']:7.0f}us "
           f"{r0['true_dT_us']:7.0f}us |")
    for d in DFS:
        r = get(fs, iv, d)
        row += f"{r['passed']:>8}" if r else f"{'-':>8}"
    P(row)

for title, key, fmt in (("MEAN EVM", "evm_mean", "{:>8.3f}"),
                        ("RAW DEMAPPER BER", "raw_ber", "{:>8.4f}"),
                        ("RS BYTES/CODEWORD (mean)", "rs_bytes_mean", "{:>8.2f}"),
                        ("RS CODEWORDS OVER 16 (%)", "rs_frac_over16", "{:>8.2%}")):
    P("")
    P(f"--- {title} ---")
    P(f"{'Fs':>8} {'iv':>4} {'true dT':>9} |" + "".join(f"{int(d):>8}" for d in DFS))
    for fs, iv in COMBOS:
        r0 = get(fs, iv, 0)
        row = f"{fs:6.0f}MS {iv:4d} {r0['true_dT_us']:7.0f}us |"
        for d in DFS:
            r = get(fs, iv, d)
            row += fmt.format(r[key]) if r else f"{'-':>8}"
        P(row)

P("")
P("=" * 92)
P("MATCHED-TIME PAIRS -- the hypothesis: tolerance follows delta_f * dT,")
P("so near-equal TRUE periods should behave alike at either sample rate.")
P("=" * 92)
for (fa, ia), (fb, ib) in (((10.0, 16), (20.0, 32)), ((10.0, 32), (20.0, 64))):
    a0, b0 = get(fa, ia, 0), get(fb, ib, 0)
    P("")
    P(f"  {fa:.0f} MSps iv={ia} (true {a0['true_dT_us']:.0f}us, "
      f"{a0['overhead_pct']:.1f}% overhead)  vs  "
      f"{fb:.0f} MSps iv={ib} (true {b0['true_dT_us']:.0f}us, "
      f"{b0['overhead_pct']:.1f}% overhead)")
    P(f"  {'delta_f':>8} | {'pass A':>8} {'EVM A':>8} | {'pass B':>8} {'EVM B':>8} "
      f"| {'dPass':>7} {'dEVM':>7}")
    for d in DFS:
        ra, rb = get(fa, ia, d), get(fb, ib, d)
        if not (ra and rb):
            continue
        P(f"  {int(d):7d} | {ra['passed']:8d} {ra['evm_mean']:8.3f} | "
          f"{rb['passed']:8d} {rb['evm_mean']:8.3f} | "
          f"{rb['passed']-ra['passed']:+7d} {rb['evm_mean']-ra['evm_mean']:+7.3f}")

P("")
P("--- CFO sanity (common Doppler must stay absorbed at both rates) ---")
P(f"{'Fs':>8} {'iv':>4} {'delta_f':>8} {'cfo_Hz':>8} (applied f_los = 1600 Hz)")
for fs, iv in COMBOS:
    for d in (0, 400):
        r = get(fs, iv, d)
        if r:
            P(f"{fs:6.0f}MS {iv:4d} {int(d):8d} {r['cfo_hz']:8.0f}")

(OUT / "report.txt").write_text("\n".join(L) + "\n")
print(f"\n[written to {OUT/'report.txt'}]")
