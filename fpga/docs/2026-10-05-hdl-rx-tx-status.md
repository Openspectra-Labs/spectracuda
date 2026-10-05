# HDL RX / TX status and OpenOFDM comparison (2026-10-05)

Branch `HDL_SPECTRA`. Target part XC7A50T (Artix-7). Clocks: 100 MHz
(time + frequency domain), 125 MHz (bit domain). All resource numbers are
Vivado 2025.2 placed-and-routed, out of context, RX only unless stated.

## 1. Where we are

**Works and tested:** RX and TX in the *old* frame format (Python reference
`ad0a396`), with **4-bit soft decision** now in the RX chain:

- Soft demapper = the cheap "thresh+w" metric as a 2-BRAM lookup table
  (`demapper_soft.v`) plus a per-frame channel-weight unit (`llr_weight.v`),
  feeding the soft Viterbi (`viterbi_dec_soft.v`, `LLR_W = 4` in `rx_top`).
- Soft values are **identical** to the Python reference model
  (`Modem.demodulate_soft_tableq`, 36,000 / 36,000 in the module test).
- FD stage suite 87/87, BD stage suite 44/44, golden fixtures 7/7 bit-exact,
  spot frames (64-QAM EVM 0.12; 16-QAM CFO + noise at 10 clocks/sample)
  bit-exact.

**Not done yet:**

| Item | State |
|---|---|
| Timing, 100 MHz | -0.725 ns: header-queue memory output feeds the demapper's table index in the same cycle. Fix: one input register. |
| Timing, 125 MHz | -0.076 ns, 1 endpoint (soft Viterbi). |
| Soft block size | 728 LUT vs the 500 LUT target (0 DSP and 2 BRAM met). |
| Commit | Soft-decision RTL is uncommitted; the Python references are committed (`521760d`, `8e5c3ce`). |
| TX -> RX loopback | Passed before today's soft change; not re-run since. |

**Parked** on branch `wip/v3-dmrs-soft` (commit `9b7b26e`): new frame
format (protected header, IL2 on, DMRS) for both TX and RX. Not merged
because its RX used a dedicated 4,300-LUT header Viterbi and the full modem
reached 87% LUT with failing timing.

## 2. How the soft demapper went wrong, and the fix

1. First version computed exact max-log soft values for every bit, weighted
   by the exact `|H|^2 / noise`, with wide multipliers:
   **2,858 LUT + 29 DSP**. Built before checking what OpenOFDM does
   (about 150 LUT, 0 DSP).
2. Python comparison (`examples/soft_metric_study.py`, 100 frames per point)
   showed the channel-strength weighting matters on fading channels, but a
   rough power-of-two weight ("thresh+w") performs the same as the exact one:

   | Channel, mod, SNR | ours (exact) | thresh+w | thresh (no weight) | OpenOFDM-style | hard |
   |---|---:|---:|---:|---:|---:|
   | AWGN 16-QAM 12 dB | 0.27 | 0.27 | 0.30 | 0.30 | 1.00 |
   | Multipath 16-QAM 14 dB | 0.22 | 0.25 | 0.75 | 0.78 | 1.00 |
   | Multipath 64-QAM 20 dB | 0.32 | 0.27 | 0.79 | 0.78 | 1.00 |
   | Deep fades 16-QAM 16 dB | 0.48 | 0.53 | 0.97 | 0.97 | 1.00 |

   (frame error rate, lower is better; ±0.05 is noise at 100 frames)
3. Precision study (`examples/soft_thresh_precision_study.py`): 3-5
   fractional bits of x/norm all match full precision; 2 bits loses. Chosen:
   4 bits, i.e. a 2048 x 36 lookup table = 2 BRAM.
4. Table model (`examples/soft_tableq_check.py`) matches full-precision
   thresh+w on every point.
5. Result in hardware: **728 LUT, 0 DSP, 2 BRAM** (demapper 410 LUT + 2 BRAM;
   weight unit 318 LUT). RX DSP back to 57, the same as the hard-decision RX.

## 3. Resources vs OpenOFDM

OpenOFDM numbers from its own report for the same part
(`reference/openofdm/vivado_synth/utilization_report_v2_hierarchical.txt`).

| Block | Ours (LUT / DSP / BRAM tiles) | OpenOFDM | Verdict |
|---|---|---|---|
| Time domain: sync, CFO, FFT | 7,010 / 28 / 5.5 | 5,967 / 48 / 4.5 | LUT +17%, DSP better by 20, BRAM similar. Our FFT is 256-point (theirs 64). |
| Freq domain: chanest, equalizer, CPE, soft demapper | 4,583 / 29 / 13.5 | 6,832 / 21 / 1 | LUT 33% better (their equalizer has 3 dividers of ~1,700 LUT); DSP +8; BRAM +12.5 worse (FD buffers). |
| Bit domain: Viterbi + deinterleave | 4,096 / 0 / 28 | 2,517 / 0 / 7 | LUT +63%, BRAM 4x worse. |
| - Viterbi alone | 3,813 LUT, survivors in LUT RAM | 2,116 LUT + 4 BRAM | ours is the 0.96-bit/clock overlapped decoder for 40 Msps |
| - coded-bit FIFO | 27 BRAM | - | sized for a full 128-symbol frame |
| **Total RX** | **15,757 / 57 / 49** | **15,810 / 69 / 13.5** | LUT equal, DSP better, **BRAM 3.6x worse** |

**Conclusions**

- LUT and DSP are at parity with OpenOFDM or better, even with a 4x larger FFT.
- **BRAM is the problem: 49 vs 13.5.** It is buffering sized for the worst
  case without measurement:
  - coded-bit FIFO: 27 tiles; measured peak use is about 2,750 of 27,648 entries;
  - FD buffers (B1 5 tiles, B2, CPE FIFO, header queue): about 12 tiles;
  - frame-sync replay: about 5 tiles.

  Right-sized from measured worst cases, the estimate is about 15-18 tiles.
- The Viterbi is larger because it is built for 40 Msps. If 20 Msps is
  enough, a smaller BRAM-survivor design like OpenOFDM's would do.
- Rule from now on: every new block gets a resource budget compared against
  the OpenOFDM equivalent **before** it is written.

## 4. Functionality

### RX chain

| Function | Status |
|---|---|
| Preamble detect / frame sync (Schmidl-Cox) | Yes |
| CFO estimate + correction | Yes |
| 256-point FFT, CP removal | Yes |
| LS channel estimate from the training symbol | Yes |
| MMSE equalizer | Yes |
| Pilot phase tracking (CPE) | Yes, per-symbol phase only; **no timing-drift (SFO) slope tracking**. Python lacks it too; it limits long packets. |
| Header: 112-bit scrambled BPSK, 1 symbol, uncoded | Yes (old format) |
| Header: protected (CRC-16 + rate-1/2 code, 2 symbols) | No (parked branch) |
| Demapper QPSK / 16-QAM / 64-QAM | Yes |
| Soft decision: 4-bit soft values + soft Viterbi | Yes (new, uncommitted) |
| Inner interleaver (IL2) | De-interleaver built but off: the old format does not use it |
| Viterbi K=7, rate 1/2 | Yes |
| Outer byte de-interleaver | Yes |
| DMRS (mid-frame channel refresh) | No (parked branch) |
| C2 control region | No |
| Payload geometry from the header | **No: the host still supplies the coded length and interleaver size (`cfg_encoded_bits`, `cfg_di_*`).** A real receiver must read them from the header. Small fix (H8). |
| AGC | No (planned later) |
| RS / CRC / MAC | Host side, by design |

### TX chain

| Function | Status |
|---|---|
| Host bytes in, two packet buffers | Yes |
| Outer byte interleaver, conv rate 1/2 + tail, padding | Yes |
| Header: 112-bit scrambled BPSK, uncoded | Yes (old format) |
| Header: protected | No (parked branch) |
| Mapper QPSK / 16 / 64-QAM, pilots, nulls, training | Yes |
| IFFT (Xilinx core), CP, preamble, Q15 samples out | Yes, verified on the real IFFT netlist |
| Underrun / bad-symbol recovery | Yes |
| Inner interleaver (IL2), DMRS | No (parked branch) |
| AD9361 / radio interface | No |

### Gaps against current Python, and functional gaps

Missing against current Python:

1. protected header;
2. DMRS;
3. IL2 on;
4. C2 region.

Missing functionality:

5. RX reading the payload geometry from the header (H8);
6. SFO tracking (Python too);
7. AGC;
8. radio (AD9361) interface.

Items 1-3 exist on `wip/v3-dmrs-soft` for TX and RX. On RX they would come
back with a small serial header decoder (or the shared payload Viterbi)
instead of the 4,300-LUT dedicated one.

## 5. Next steps, in order

1. Timing fixes (demapper input register; 125 MHz endpoint) and soft block to
   <= 500 LUT:
   - weight unit 318 -> ~120 LUT: fold sqrt(2) into the thresholds as fixed
     shift-adds, compare the top ~16 bits instead of 40;
   - demapper 410 -> ~300 LUT: narrower per-bit round/clamp.

   Then one Vivado run, re-run the TX -> RX loopback, commit.
2. Coded-bit FIFO -> 4,096 entries (27 -> 3 BRAM tiles), proven with a
   rate-limited stress test at 5 / 20 / 40 Msps; size the FD buffers from
   measured peaks.
3. Viterbi: decide 40 vs 20 Msps; if 20 is enough, move to a smaller
   BRAM-survivor design.
4. H8: payload geometry from the header.
5. Bring back the new format (protected header, IL2, DMRS) from the parked
   branch, with a resource budget set first.
