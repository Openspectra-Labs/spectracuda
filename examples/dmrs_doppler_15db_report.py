#!/usr/bin/env python3
"""Render debug/dmrs_doppler_15db/results.json, with every Doppler cell
expressed as an INCREMENT over the delta_f = 0 baseline at the same SNR,
and compared against the 25 dB run in debug/dmrs_doppler_postfix/
(read-only -- that file is never rewritten)."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("debug/dmrs_doppler_15db")
R = json.loads((OUT / "results.json").read_text())
REF = Path("debug/dmrs_doppler_postfix/results.json")
R25 = json.loads(REF.read_text())["part3"] if REF.exists() else []
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    L.append(s)


def get(iv, df):
    return next(r for r in R if r["iv"] == iv and r["delta_f"] == df)


def get25(iv, df):
    m = [r for r in R25 if r["iv"] == iv and r["delta_f"] == df]
    return m[0] if m else None


DFS = sorted({r["delta_f"] for r in R})
IVS = [(16, 512), (8, 256)]

P("=" * 78)
P("SNR = 15 dB, 16QAM, a=0.2, f_los=1600 Hz, 5575 B, 500 frames/cell")
P("timing_advance=2, full shipped FEC chain, commit 777d4ea")
P("=" * 78)

P("")
P("--- BASELINE GATE: delta_f = 0 (AWGN + static multipath only) ---")
P(f"{'DMRS':>8} {'pass':>11} {'PER%':>7} {'meanEVM':>8} {'worstEVM':>9} "
  f"{'rawBER':>8} {'postVit':>8} {'RSb/cw':>7} {'RSmax':>6} {'>16':>7}")
for iv, us in IVS:
    b = get(iv, 0)
    P(f"{us:6d}us {b['passed']:5d}/{b['trials']:<5d} {b['per']:7.2f} "
      f"{b['evm_mean']:8.3f} {b['evm_worst']:9.3f} {b['raw_ber']:8.4f} "
      f"{b['post_vit_ber']:8.4f} {b['rs_bytes_mean']:7.2f} "
      f"{b['rs_bytes_max']:6.0f} {100*b['rs_frac_over16']:6.2f}%")

P("")
P("--- COMPACT COMPARISON ---")
P(f"{'delta_f':>8} | {'512 pass':>10} {'512 EVM':>8} {'512 rawBER':>11} | "
  f"{'256 pass':>10} {'256 EVM':>8} {'256 rawBER':>11}")
for df in DFS:
    a, b = get(16, df), get(8, df)
    P(f"{df:7d}Hz | {a['passed']:5d}/{a['trials']:<4d} {a['evm_mean']:8.3f} "
      f"{a['raw_ber']:11.4f} | {b['passed']:5d}/{b['trials']:<4d} "
      f"{b['evm_mean']:8.3f} {b['raw_ber']:11.4f}")

P("")
P("--- INCREMENTAL DEGRADATION over the 15 dB delta_f=0 baseline ---")
P("(d-prefixed columns are Doppler-attributable; the rest is AWGN)")
P(f"{'DMRS':>8} {'delta_f':>8} {'PER%':>7} {'dPER%':>7} {'EVM':>7} {'dEVM':>7} "
  f"{'rawBER':>8} {'d rawBER':>9}")
for iv, us in IVS:
    b0 = get(iv, 0)
    for df in DFS:
        r = get(iv, df)
        P(f"{us:6d}us {df:8d} {r['per']:7.2f} {r['per']-b0['per']:7.2f} "
          f"{r['evm_mean']:7.3f} {r['evm_mean']-b0['evm_mean']:7.3f} "
          f"{r['raw_ber']:8.4f} {r['raw_ber']-b0['raw_ber']:9.4f}")

P("")
P("--- RS CORRECTION LOAD (RS(255,223) corrects 16 bytes/codeword) ---")
P(f"{'DMRS':>8} {'delta_f':>8} {'mean':>7} {'max':>6} {'frac>16':>9} "
  f"{'codewords':>10} {'postVitBER':>11}")
for iv, us in IVS:
    for df in DFS:
        r = get(iv, df)
        P(f"{us:6d}us {df:8d} {r['rs_bytes_mean']:7.2f} {r['rs_bytes_max']:6.0f} "
          f"{100*r['rs_frac_over16']:8.3f}% {r['rs_codewords']:10d} "
          f"{r['post_vit_ber']:11.4f}")

P("")
P("--- CFO / CPE / timing (sanity: common Doppler must stay absorbed) ---")
P(f"{'DMRS':>8} {'delta_f':>8} {'cfo_Hz':>8} {'meanCPE':>8} {'maxCPE':>7} "
  f"{'start_index':>18}")
for iv, us in IVS:
    for df in DFS:
        r = get(iv, df)
        P(f"{us:6d}us {df:8d} {r['cfo_hz']:8.0f} {r['cpe_mean']:8.3f} "
          f"{r['cpe_max']:7.3f} {str(r['start_index']):>18}")

if R25:
    P("")
    P("--- SNR COST: 25 dB (debug/dmrs_doppler_postfix, unmodified) vs 15 dB ---")
    P(f"{'DMRS':>8} {'delta_f':>8} | {'25dB pass':>11} {'25dB EVM':>9} | "
      f"{'15dB pass':>11} {'15dB EVM':>9} | {'dEVM':>7}")
    for iv, us in IVS:
        for df in DFS:
            r = get(iv, df)
            o = get25(iv, df)
            if o is None:
                continue
            P(f"{us:6d}us {df:8d} | {o['passed']:5d}/{o['trials']:<5d} "
              f"{o['evm_mean']:9.3f} | {r['passed']:5d}/{r['trials']:<5d} "
              f"{r['evm_mean']:9.3f} | {r['evm_mean']-o['evm_mean']:7.3f}")

(OUT / "report.txt").write_text("\n".join(L) + "\n")
print(f"\n[written to {OUT/'report.txt'}]")
