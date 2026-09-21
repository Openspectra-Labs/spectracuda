#!/usr/bin/env python3
"""Render debug/multipath_stress/phase{2,3,4}.json. Read-only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("debug/multipath_stress")
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    L.append(s)


def load(n):
    f = OUT / f"phase{n}.json"
    return json.loads(f.read_text()) if f.exists() else []


AMPS = (0.2, 0.4, 0.6, 0.8, 1.0)
DELAYS = (50, 100, 200, 500, 1000)

p2 = load(2)
by2 = {r["label"]: r for r in p2}

if p2:
    P("=" * 94)
    P("PHASE 2 -- STRONG TWO-PATH, delta_f = 0 (STATIC frequency-selective only)")
    P("20 MSps, fft=256, cp=64 (CP = 3.2 us), 16QAM, 15 dB, iv=32, 300 frames/cell")
    P("Echo phase randomized per frame. Delays are EXACT (1 sample = 50 ns at 20 MSps).")
    P("=" * 94)
    for title, key, fmt in (("PACKET SUCCESS (of 300)", "passed", "{:>9}"),
                            ("MEAN EVM", "evm_mean", "{:>9.3f}"),
                            ("WORST-SYMBOL EVM", "evm_worst", "{:>9.3f}"),
                            ("RAW DEMAPPER BER", "raw_ber", "{:>9.4f}"),
                            ("POST-VITERBI BER", "post_vit_ber", "{:>9.4f}"),
                            ("RS BYTES/CODEWORD (mean)", "rs_bytes_mean", "{:>9.2f}"),
                            ("RS BYTES/CODEWORD (max)", "rs_bytes_max", "{:>9.0f}"),
                            ("RS CODEWORDS OVER 16 (%)", "rs_frac_over16", "{:>9.2%}"),
                            ("FRAMES NEVER DETECTED (%)", "no_frame_pct", "{:>9.1f}")):
        P("")
        P(f"--- {title} ---")
        P(f"{'echo a':>8} |" + "".join(f"{d:>7}ns" for d in DELAYS))
        for a in AMPS:
            row = f"{a:8.1f} |"
            for d in DELAYS:
                r = by2.get(f"a={a}_d={d}ns")
                row += fmt.format(r[key]) if r else f"{'-':>9}"
            P(row)

    P("")
    P("--- SYNC: start_index distribution (mechanism 3) ---")
    P(f"{'echo a':>8} |" + "".join(f"{d:>16}ns" for d in DELAYS))
    for a in AMPS:
        row = f"{a:8.1f} |"
        for d in DELAYS:
            r = by2.get(f"a={a}_d={d}ns")
            row += f"{str(r['start_index']):>18}" if r else f"{'-':>18}"
        P(row)

p2sc = []
f = OUT / "phase2_subcarrier.json"
if f.exists():
    p2sc = json.loads(f.read_text())

if p2sc:
    P("")
    P("=" * 94)
    P("SUBCARRIER ANALYSIS -- do failures track deep spectral nulls?")
    P("")
    P("Statistics are PER FRAME against that frame's OWN nulls, then averaged.")
    P("Averaging the spectra first would be wrong: the echo phase is redrawn")
    P("every frame, so the nulls move, and the mean |H| of an a=1.0 channel")
    P("comes out as a ~3 dB ripple instead of the true null it really has.")
    P("")
    P("|H_hat| err is a SHAPE comparison (both normalized to unit mean), since")
    P("h_hat also carries the receiver's own scaling and timing phase ramp.")
    P("=" * 94)
    P(f"{'echo a':>7} {'delay':>8} {'null':>8} {'|H| min':>8} {'|H| max':>8} "
      f"{'EVM deep':>9} {'EVM peak':>9} {'corr':>6} {'H err':>7} {'H err deep':>11}")
    for r in p2sc:
        P(f"{r['a']:7.1f} {r['delay_ns']:6d}ns {r['null_db']:6.1f}dB "
          f"{r['h_min']:8.3f} {r['h_max']:8.3f} {r['evm_deep']:9.3f} "
          f"{r['evm_peak']:9.3f} {r['corr']:6.2f} {r['h_shape_err']:7.4f} "
          f"{r['h_shape_err_deep']:11.4f}")
    P("")
    P("Null depth is 20*log10((1+a)/(1-a)) and depends ONLY on the echo")
    P("amplitude -- delay sets how MANY nulls span the band, not how deep.")

p3 = load(3)
if p3:
    by3 = {r["label"]: r for r in p3}
    P("")
    P("=" * 94)
    P("PHASE 3 -- STRONG MULTIPATH + MODERATE DIFFERENTIAL DOPPLER")
    P("Does a faster refresh (iv=16) rescue it, or is it a deep-fade/FEC problem?")
    P("=" * 94)
    P(f"{'a':>5} {'delay':>8} {'delta_f':>8} | {'iv=32 pass':>11} {'EVM':>7} | "
      f"{'iv=16 pass':>11} {'EVM':>7} | {'verdict':>28}")
    for a in (0.4, 0.6, 0.8):
        for d in (200, 500):
            for df in (100.0, 300.0):
                r32 = by3.get(f"a={a}_d={d}ns_df={df:.0f}_iv=32")
                r16 = by3.get(f"a={a}_d={d}ns_df={df:.0f}_iv=16")
                if not (r32 and r16):
                    continue
                gain = r16["passed"] - r32["passed"]
                n = r32["trials"]
                if r32["passed"] >= 0.99 * n:
                    v = "both fine"
                elif gain > 0.15 * n:
                    v = "iv=16 HELPS -> ageing"
                else:
                    v = "iv=16 does NOT help -> fade"
                P(f"{a:5.1f} {d:6d}ns {df:7.0f}H | {r32['passed']:5d}/{n:<5d} "
                  f"{r32['evm_mean']:7.3f} | {r16['passed']:5d}/{r16['trials']:<5d} "
                  f"{r16['evm_mean']:7.3f} | {v:>28}")

p4 = load(4)
if p4:
    P("")
    P("=" * 94)
    P("PHASE 4 -- RANDOMIZED 3-6 TAP CHANNELS (per-channel PER distribution)")
    P("LOS 1.0 + 2-5 reflections, amplitude 0.1-0.8, delay 50-1000 ns on the")
    P("50 ns grid, phase uniform, differential Doppler -300..+300 Hz.")
    P("=" * 94)
    for row in p4:
        ch = row["channels"]
        pers = np.array([c["per"] for c in ch])
        refl = np.array([c["reflected_power"] for c in ch])
        strong = np.array([c["strongest_echo_ratio"] for c in ch])
        P("")
        P(f"  iv={row['iv']}  ({len(ch)} channels x {ch[0]['frames']} frames)")
        P(f"    PER      median {np.median(pers):5.1f}%   p90 {np.percentile(pers,90):5.1f}%   "
          f"p95 {np.percentile(pers,95):5.1f}%   worst {pers.max():5.1f}%")
        P(f"    channels fully clean (PER=0): {(pers==0).sum()}/{len(pers)} "
          f"({(pers==0).mean()*100:.0f}%)")
        P(f"    channels >10% PER:            {(pers>10).sum()}/{len(pers)} "
          f"({(pers>10).mean()*100:.0f}%)")
        P(f"    total reflected power: median {np.median(refl):.2f}, max {refl.max():.2f}")
        P(f"    strongest echo / LOS:  median {np.median(strong):.2f}, max {strong.max():.2f}")
        bad = sorted(ch, key=lambda c: -c["per"])[:5]
        P(f"    worst 5 channels:")
        P(f"      {'PER%':>6} {'EVM':>7} {'taps':>5} {'refl pwr':>9} "
          f"{'strongest':>10} {'max delay':>10}")
        for c in bad:
            P(f"      {c['per']:6.1f} {c['evm_mean']:7.3f} {c['n_paths']:5d} "
              f"{c['reflected_power']:9.2f} {c['strongest_echo_ratio']:10.2f} "
              f"{c['max_delay_ns']:8.0f}ns")
    if len(p4) == 2:
        a32 = np.array([c["per"] for c in p4[0]["channels"]])
        a16 = np.array([c["per"] for c in p4[1]["channels"]])
        P("")
        P(f"  iv=16 vs iv=32 across the SAME channel seeds: median PER "
          f"{np.median(a32):.1f}% -> {np.median(a16):.1f}%, "
          f"clean {(a32==0).mean()*100:.0f}% -> {(a16==0).mean()*100:.0f}%")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "report.txt").write_text("\n".join(L) + "\n")
print(f"\n[written to {OUT/'report.txt'}]")
