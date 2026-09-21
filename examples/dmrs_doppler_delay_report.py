#!/usr/bin/env python3
"""Render debug/dmrs_doppler_delay/results.json. Read-only.

Central question: does a fixed PHYSICAL delay spread (100 ns = 2 samples
at 20 MSps, 1 sample at 10 MSps) change Doppler tolerance?

Analytic expectation -- the stale-H error is

    dH[k] = a*(e^{j*theta(t)} - e^{j*theta(t0)}) * e^{-j2pi*k*delay/N}
    |dH[k]| = 2a|sin(pi*delta_f*dT)|

The delay factor has UNIT MAGNITUDE, so it drops out: `delay` rotates the
error across subcarriers but does not change its size. Aging should
therefore be delay-independent, and only sync placement (an energy
centroid, which a longer echo pulls right) has any reason to care.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("debug/dmrs_doppler_delay")
R = json.loads((OUT / "results.json").read_text())
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    L.append(s)


def get(fs, iv, delay, df, adv=2):
    m = [r for r in R if r["fs_msps"] == fs and r["iv"] == iv
         and r["delay"] == delay and r["delta_f"] == df
         and r["timing_advance"] == adv]
    return m[0] if m else None


DFS = sorted({r["delta_f"] for r in R if r["timing_advance"] == 2})
ROWS = [(10.0, 16, 1), (20.0, 32, 2), (20.0, 32, 1), (20.0, 16, 2), (20.0, 16, 1)]

P("=" * 96)
P("FIXED PHYSICAL DELAY SPREAD -- 100 ns echo across sample rates")
P("320-sample symbol, 16QAM, a=0.2, SNR=15 dB, f_los=1600 Hz, 300 frames/cell")
P("timing_advance=2 unless stated")
P("=" * 96)

for title, key, fmt in (("PACKET SUCCESS (of 300)", "passed", "{:>8}"),
                        ("MEAN EVM", "evm_mean", "{:>8.3f}"),
                        ("RAW DEMAPPER BER", "raw_ber", "{:>8.4f}"),
                        ("RS BYTES/CODEWORD (mean)", "rs_bytes_mean", "{:>8.2f}")):
    P("")
    P(f"--- {title} ---")
    P(f"{'Fs':>7} {'iv':>3} {'true dT':>9} {'delay':>10} |"
      + "".join(f"{int(d):>8}" for d in DFS))
    for fs, iv, dl in ROWS:
        r0 = get(fs, iv, dl, 0)
        if not r0:
            continue
        row = (f"{fs:5.0f}MS {iv:3d} {r0['true_dT_us']:7.0f}us "
               f"{dl}smp/{r0['delay_ns']:3.0f}ns |")
        for d in DFS:
            r = get(fs, iv, dl, d)
            row += fmt.format(r[key]) if r else f"{'-':>8}"
        P(row)

P("")
P("=" * 96)
P("DELAY-ONLY CONTRAST -- same rate, same interval, 100 ns vs 50 ns")
P("If |dH| = 2a|sin(pi*delta_f*dT)| is right, these must agree.")
P("=" * 96)
for iv in (32, 16):
    a0 = get(20.0, iv, 2, 0)
    if not a0:
        continue
    P("")
    P(f"  20 MSps iv={iv} (true {a0['true_dT_us']:.0f}us)")
    P(f"  {'delta_f':>8} | {'100ns pass':>11} {'100ns EVM':>10} | "
      f"{'50ns pass':>10} {'50ns EVM':>9} | {'dPass':>7} {'dEVM':>8}")
    for d in DFS:
        ra, rb = get(20.0, iv, 2, d), get(20.0, iv, 1, d)
        if not (ra and rb):
            continue
        P(f"  {int(d):7d} | {ra['passed']:11d} {ra['evm_mean']:10.3f} | "
          f"{rb['passed']:10d} {rb['evm_mean']:9.3f} | "
          f"{ra['passed']-rb['passed']:+7d} {ra['evm_mean']-rb['evm_mean']:+8.3f}")

P("")
P("=" * 96)
P("MATCHED ON BOTH refresh period AND physical delay spread")
P("  10 MSps iv=16 (544us, 1 sample = 100ns, 5.9% overhead)")
P("  20 MSps iv=32 (528us, 2 samples = 100ns, 3.0% overhead)")
P("=" * 96)
P(f"  {'delta_f':>8} | {'10MS pass':>10} {'10MS EVM':>9} | "
  f"{'20MS pass':>10} {'20MS EVM':>9} | {'dPass':>7} {'dEVM':>8}")
for d in DFS:
    ra, rb = get(10.0, 16, 1, d), get(20.0, 32, 2, d)
    if not (ra and rb):
        continue
    P(f"  {int(d):7d} | {ra['passed']:10d} {ra['evm_mean']:9.3f} | "
      f"{rb['passed']:10d} {rb['evm_mean']:9.3f} | "
      f"{rb['passed']-ra['passed']:+7d} {rb['evm_mean']-ra['evm_mean']:+8.3f}")

P("")
P("=" * 96)
P("IS timing_advance=2 STILL ENOUGH for a 2-sample echo?")
P("20 MSps, iv=32, 100 ns (2 samples). Sync tracks an energy centroid, so")
P("a longer echo pulls its peak further right.")
P("=" * 96)
P(f"  {'advance':>8} {'delta_f':>8} {'pass':>10} {'meanEVM':>9} {'rawBER':>9} "
  f"{'start_index':>22}")
for adv in (0, 1, 2, 3, 4, 6):
    for d in (0, 400):
        r = get(20.0, 32, 2, d, adv=adv)
        if r:
            P(f"  {adv:8d} {int(d):8d} {r['passed']:5d}/{r['trials']:<4d} "
              f"{r['evm_mean']:9.3f} {r['raw_ber']:9.4f} "
              f"{str(r['start_index']):>22}")

(OUT / "report.txt").write_text("\n".join(L) + "\n")
print(f"\n[written to {OUT/'report.txt'}]")
