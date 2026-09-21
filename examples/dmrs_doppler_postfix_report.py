#!/usr/bin/env python3
"""Render debug/dmrs_doppler_postfix/results.json into the report tables.
Read-only; produces no PHY effect. See dmrs_doppler_postfix_study.py."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("debug/dmrs_doppler_postfix")
R = json.loads((OUT / "results.json").read_text())
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    L.append(s)


def pick(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


IVS = ((4, 128), (8, 256), (16, 512), (32, 1024))
US = {iv: us for iv, us in IVS}

P("=" * 78)
P("PART 9 -- frame geometry, measured from real 5575 B frames")
P("=" * 78)
P(f"{'nominal':>8} {'iv':>4} {'data':>6} {'dmrs':>6} {'total':>6} "
  f"{'overhead':>9} {'airtime':>9} {'true dT':>9}")
for g in R["geometry"]:
    P(f"{g['nominal_us']:6d}us {g['interval']:4d} {g['data_symbols']:6d} "
      f"{g['dmrs_symbols']:6d} {g['total_symbols']:6d} {g['overhead_pct']:8.1f}% "
      f"{g['airtime_ms']:8.2f}ms {g['true_dT_us']:8.0f}us")

P("")
P("=" * 78)
P("PART 1 -- clean baseline, f_los = f_echo = 0 (no Doppler at all)")
P("=" * 78)
P(f"{'a':>5} {'SNR':>5} {'pass':>9} {'meanEVM':>8} {'worstEVM':>9} "
  f"{'rawBER':>8} {'postVit':>8} {'RSb/cw':>7} {'start_index':>14}")
for r in R["part1"]:
    P(f"{r['a']:5.1f} {r['snr']:5.0f} {r['passed']:4d}/{r['trials']:<4d} "
      f"{r['evm_mean']:8.3f} {r['evm_worst']:9.3f} {r['raw_ber']:8.4f} "
      f"{r['post_vit_ber']:8.4f} {r['rs_bytes_mean']:7.2f} "
      f"{str(r['start_index']):>14}")

P("")
P("=" * 78)
P("PART 2 -- COMMON Doppler only (delta_f = 0), nominal 512 us")
P("=" * 78)
P(f"{'f_los=f_echo':>13} {'cfo_norm':>9} {'cfo_Hz':>8} {'pass':>9} "
  f"{'meanEVM':>8} {'worstEVM':>9} {'PER%':>6} {'meanCPE':>8} {'maxCPE':>7}")
for r in R["part2"]:
    P(f"{r['f_los']:11.0f}Hz {r['cfo_mean']:9.5f} "
      f"{r['cfo_mean']*10e6/256:8.0f} {r['passed']:4d}/{r['trials']:<4d} "
      f"{r['evm_mean']:8.3f} {r['evm_worst']:9.3f} {r['per']:6.1f} "
      f"{r['cpe_mean']:8.3f} {r['cpe_max']:7.3f}")

P("")
P("=" * 78)
P("PART 3 -- DMRS interval x DIFFERENTIAL Doppler, f_los = 1600 Hz")
P("=" * 78)
dfs = sorted({r["delta_f"] for r in R["part3"]})
for title, key, fmt in (("PACKET SUCCESS (pass/trials)", None, None),
                        ("MEAN EVM", "evm_mean", "{:>8.3f}"),
                        ("MEAN WORST-SYMBOL EVM", "evm_worst", "{:>8.3f}"),
                        ("RAW DEMAPPER BER", "raw_ber", "{:>8.4f}"),
                        ("POST-VITERBI BER", "post_vit_ber", "{:>8.4f}"),
                        ("RS BYTE ERRORS / CODEWORD (mean)", "rs_bytes_mean", "{:>8.1f}")):
    P("")
    P(f"--- {title} ---")
    P(f"{'DMRS':>9} |" + "".join(f"{int(d):>8}" for d in dfs))
    for iv, us in IVS:
        row = f"{us:7d}us |"
        for d in dfs:
            c = pick(R["part3"], iv=iv, delta_f=d)
            if not c:
                row += f"{'-':>8}"
            elif key is None:
                row += f"{c[0]['passed']:>4}/{c[0]['trials']:<3}"
            else:
                row += fmt.format(c[0][key])
        P(row)

P("")
P("=" * 78)
P("PART 5 -- BEFORE vs AFTER the timing fix")
P("nominal 512 us, crc32, f_los=667 Hz, identical seeds (the OLD config)")
P("=" * 78)
P(f"{'delta_f':>8} | {'BEFORE (adv=0)':>22} | {'AFTER (adv=2)':>22} | {'delta':>16}")
P(f"{'':>8} | {'pass':>9} {'meanEVM':>11} | {'pass':>9} {'meanEVM':>11} | "
  f"{'pass':>7} {'EVM':>8}")
for d in sorted({r["delta_f"] for r in R["part5"]}):
    b = pick(R["part5"], delta_f=d, timing_advance=0)[0]
    a = pick(R["part5"], delta_f=d, timing_advance=2)[0]
    P(f"{int(d):7d}Hz | {b['passed']:4d}/{b['trials']:<4d} {b['evm_mean']:11.3f} | "
      f"{a['passed']:4d}/{a['trials']:<4d} {a['evm_mean']:11.3f} | "
      f"{a['passed']-b['passed']:+7d} {a['evm_mean']-b['evm_mean']:+8.3f}")

P("")
P("=" * 78)
P("PART 6 -- delta_f = 300 Hz focus (f_los=1600, f_echo=1900), 500 frames")
P("=" * 78)
P(f"{'DMRS':>9} {'true dT':>9} {'pass':>11} {'PER%':>7} {'meanEVM':>8} "
  f"{'worstEVM':>9} {'rawBER':>8} {'RSb/cw':>7} {'cw>16':>7}")
for r in R["part6"]:
    g = [x for x in R["geometry"] if x["interval"] == r["iv"]][0]
    P(f"{US[r['iv']]:7d}us {g['true_dT_us']:8.0f}us {r['passed']:5d}/{r['trials']:<5d} "
      f"{r['per']:7.2f} {r['evm_mean']:8.3f} {r['evm_worst']:9.3f} "
      f"{r['raw_ber']:8.4f} {r['rs_bytes_mean']:7.2f} {r['rs_cw_over']:7.2f}")

P("")
P("=" * 78)
P("PART 7 -- same delta_f=300 Hz, very different COMMON Doppler")
P("=" * 78)
P(f"{'f_los':>7} {'f_echo':>7} {'cfo_Hz':>8} {'pass':>11} {'meanEVM':>8} "
  f"{'worstEVM':>9} {'rawBER':>8}")
for r in R["part7"]:
    P(f"{r['f_los']:6.0f}Hz {r['f_echo']:6.0f}Hz {r['cfo_mean']*10e6/256:8.0f} "
      f"{r['passed']:5d}/{r['trials']:<5d} {r['evm_mean']:8.3f} "
      f"{r['evm_worst']:9.3f} {r['raw_ber']:8.4f}")

P("")
P("=" * 78)
P("PART 8 -- |dH| = 2a|sin(pi*delta_f*dT)|, dT = TRUE DMRS-to-DMRS spacing")
P("=" * 78)
P(f"{'nominal':>8} {'true dT':>9} {'delta_f':>8} {'measured':>9} "
  f"{'analytic':>9} {'ratio':>7}")
for r in R["part8"]:
    P(f"{r['nominal_us']:6d}us {r['true_dT_us']:8.0f}us {r['delta_f']:8d} "
      f"{r['measured']:9.4f} {r['analytic']:9.4f} "
      f"{(r['ratio'] if r['ratio'] else float('nan')):7.2f}")

P("")
P("=" * 78)
P("EVM vs TIME SINCE LAST DMRS (f_los=1600; baseline row is delta_f=0)")
P("=" * 78)
for iv, us in IVS:
    for d in (0, 300, 667):
        c = pick(R["part3"], iv=iv, delta_f=d)
        if not c:
            continue
        prof = c[0]["evm_vs_t"]
        pts = [(int((p+1)*32), prof[p]) for p in range(0, len(prof), max(1, len(prof)//6))]
        P(f"{us:6d}us df={d:4d}Hz: " +
          "  ".join(f"{t}us={v:.3f}" for t, v in pts))

(OUT / "report.txt").write_text("\n".join(L) + "\n")
print(f"\n[written to {OUT/'report.txt'}]")
