# `hls/` rundown — read this before asking what's left to build

> ## 🔒 RX RECEIVER LOCKED — 2026-09-14
> Bit-exact against Python (QPSK/QAM16/QAM64, to 2048-byte packets),
> timing-closed at 100 MHz on xc7a50t (WNS +0.087 ns, 0 failing
> endpoints, DRC clean), and characterised under AWGN + phase noise
> (99/100 packets at EVM 0.12).
>
> **Before changing ANY RTL, read section 8 — how to re-test.** The
> timing margin is +0.087 ns; any added logic can break it, and
> simulation alone will not tell you.

Status as of 2026-09-14 (sections 11 and 12 are that day's work; the
dated sub-headings below are kept as records of when each thing was
proven). This file exists because the same questions keep
getting re-derived wrong from the code and from stale docs. If you are a
new agent: read sections 0, 1 and 2 before proposing any work.

---

## 0. Goal and context

**The aim is a decent-enough OFDM RX *and* TX chain in fabric** — a
working receiver and transmitter, not a research-grade one. RX is what
exists today; TX is an explicit goal, not yet started.

**Where it ships.** The Spectra M.2 SDR. The synthesized design is:

```
  PCIe + USB 2.0  ──►  ofdmtxrx IP (this work)  ──►  AD9361
```

The surrounding PCIe/USB and AD9361 HDL already exist and are not ours to
design or size.

**The golden model is validated silicon, not theory.** `spectracuda`
(Python) has been proven on a real 2x Raspberry Pi 5 + PlutoSDR link,
including sustained streaming and a ~20-network WiFi interference
environment. That is why it, and not a textbook, is the reference for
every block here.

**This is the first attempt at an RX chain in Verilog.** The governing
rule follows from that: **match the Python model that has been tested on
real hardware, even where a better-known technique exists.** As the
Python model improves, the HDL follows it. Do not "upgrade" a block past
the golden model — that breaks the only reference we can check against.

The blocks were written using `reference/openofdm` for receiver structure
and `spectracuda` for the algorithms and golden vectors.

---

## 1. Status at a glance — the milestone ladder

**A stage is DONE only if it has been run and checked. Code existing is
not DONE.**

| # | Milestone | Status |
|---|---|---|
| 1 | Each block written and verified against Python golden vectors | ✅ **DONE** — 9/9 in-scope |
| 2 | RX decodes a Python-generated frame end-to-end, Python filling the gaps | ✅ **DONE** for everything simulable here — 11/11 chainable RTL stages pass on clean and CFO frames (`rtl/trace.py`, `rtl/check_chain.py`). `cp_fft` still needs xsim. |
| 3 | Remaining datapath blocks in RTL (grid extract, pilot CPE) | ✅ **DONE** — written, chained, passing. Training averaging is a no-op at `N_TRAINING 1`. |
| 4 | Control in RTL: header decode + frame state machine | ✅ **DONE** — `header_decode.v`, `frame_sync.v`, both chained and passing, and both now synthesize and close 100 MHz standalone (`header_decode` +5.997 ns, `frame_sync` +4.064). |
| 5 | `rx_top.v` written — whole chain wired, all-RTL end-to-end sim passes | ✅ **DONE — LOCKED (behavioural FFT).** Bit-exact vs Python on **qpsk / qam16 / qam64**, up to **2048-byte packets** (15/15, 5 seeds each) plus a 64/512/2000-bit x cfo sweep (18/18). Noise + phase noise now tested to 20% EVM (section 7). Still needs the real `xfft_256` under xsim. |
| 6 | Top level synthesized as ONE design (not per-block out-of-context) | ✅ **DONE, and it caught 6 real bugs** (section 6). Final: 41% LUT, 19% FF, **48% DSP**, 25% BRAM, 52% slices. |
| 7 | Place & route, timing closed on the integrated design | ✅ **CLOSED at 100 MHz, out-of-context.** Setup WNS **+0.087 ns**, hold WHS +0.031, pulse +3.750, **TNS/THS/TPWS all 0.000, 0 failing endpoints**, DRC 0. All 9 functional cases still bit-exact. Took four fixes from -22.77 ns; the route and the remaining levers are in **section 12**. **OOC — no I/O, no MMCM, and none of the PCIe/DMA/AD9361 infrastructure. This margin will probably not survive milestone 9.** |
| 8 | Bitstream generated | ❌ not started |
| 9 | Integrated into the Spectra M.2 design (PCIe+USB → ofdmtxrx → AD9361) | ❌ not started |
| 10 | Decodes real over-the-air frames on hardware | ❌ not started |
| 11 | TX chain | ❌ not started |

### Direct answers to the three questions that get asked

- **Is the complete top-level RX written?** **Yes** — `rtl/rtl/rx/rx_top.v`
  wires all 14 blocks. Reed-Solomon and CRC are deliberately absent; they
  run on the host (section 2).
- **Is the RX chain working?** **Yes, in simulation, on an ideal
  channel.** It decodes Python-generated frames bit-exactly on qpsk,
  qam16 and qam64 up to 2048-byte packets. It has NEVER been run with
  noise, and never against the real `xfft_256`.
- **Is the top level synthesized and does it meet timing?** **Yes to
  both, out-of-context.** `rx_top` builds as one design, routes DRC-clean
  at 41% LUT / 19% FF / 48% DSP / 25% BRAM / 52% slices, and **meets
  setup, hold and pulse width at 100 MHz with zero failing endpoints**
  (WNS +0.087 ns). Section 12 records how, and the two levers worth ~2 ns
  each if 125 MHz is ever wanted. **No bitstream exists.** And OOC is
  doing real work in that sentence — see "What closure does NOT mean".
- **What is the throughput ceiling?** The front end sustains **1 sample
  per clock = 100 Msps**, verified block by block; `pilot_cpe` is the
  tightest at 82% duty, so the real front-end limit is ~122 Msps, well
  above the AD9361's 61.44 Msps. **The Viterbi is the only block that
  cannot** — it manages ~24.6 Mbit/s decoded, which fits 10 Msps QAM64 by
  9% and nothing above it. Section 13.

### Blockers — stated plainly

1. **Nothing is blocking.** Vivado is installed (section 12) and no
   licence is needed for our own flow: `xc7a50t` is covered by Vivado ML
   Standard's free built-in licence, and `viterbi_dec.v` was hand-written
   specifically to avoid the licensed Xilinx core.
2. **The licence question is closed.** `rx_top` has now been through
   place & route in full, `xfft_256` included, with no licence
   challenge. The old worry that `xfft_256` might demand one at P&R and
   stop stage 7 dead did not happen.

   What replaces it is an ordinary engineering problem: **the routed
   design fails timing by 4.846 ns.** Section 12 measures it.
3. **The Xilinx Viterbi LogiCORE licence is needed eventually, but not to
   ship.** It gates only the optional swap of our hand-written decoder
   (section 4) and rebuilding openofdm's reference numbers.

### Stage 2 result (2026-09-11)

`python -m fpga.sim.trace [--cfo F]` then `python check_chain.py
--trace-dir build/trace[_cfo]`. Both a clean frame and a cfo=0.05 frame:

**11/11 RTL stages pass**, on a clean frame and a cfo=0.05 frame:

| stage | verdict | evidence |
|---|---|---|
| sc_sync | PASS | start_index 200 = true frame start; peak metric matches |
| cfo_est+corr | PASS | angle 0 LSB error, driven by sc_sync's OWN P; evm 0.18% |
| frame_sync | PASS | start 200 = Python's; 1984/1984 samples replayed, 0 mismatches |
| header_decode | PASS | len=2000 mod=qam64 crc=crc16 fec0=rs_m8 fec1=conv_v27, valid=1 |
| grid_extract | PASS | 4x256 bins -> 864 data + 32 pilots, 0 mismatches (exact) |
| ls_chanest | PASS | 224 pilots -> 256 bins, evm 0.024%, corr 1.000000 |
| mmse_eq | PASS | header[216] + payload[864] + pilots[32], evm <= 0.10% |
| pilot_cpe | PASS | 864/864 out, evm 0.059%, max angle error 0 LSB |
| demapper | PASS | 864 qam64 symbols -> 5184 bits, 0 mismatches |
| viterbi_dec | PASS | 2534 symbols -> 2528 bits, 0 mismatches |
| deinterleaver | PASS | 316 units (M=18 N=18), 0 mismatches |

**No divergence anywhere.** Still carried by Python: `cp_fft` (needs
xsim) and training averaging (a no-op at `N_TRAINING 1` -- it only
becomes a real block if `n_training_symbols` is raised).

**The four new blocks are NOT yet synthesized** -- they have no LUT/DSP/
timing numbers and are unproven against a clock. Vivado IS installed
(section 12); this simply has not been run yet, and it is the next thing
to do.

> **SUPERSEDED 2026-09-14.** All four were synthesized and all four close
> 100 MHz standalone; `rx_top` now places and routes as one design. See
> the milestone table, section 6 and section 12. This paragraph is kept
> as the record of what was true on 2026-09-11.

Two traps this run hit, both worth remembering:
- Standalone `check_*.py` artifacts and chain artifacts share filenames.
  The chain now writes `chain_*` files and its own `*_params.vh`; do not
  point a chain stage at a standalone output path.
- Compile-time `define`s (EQ_N, DM_N, VIT_NSYM) mean a cached simulator
  binary silently runs the PREVIOUS case's length. Rebuild per case --
  `gen_cfo_stimulus.py` documents the same trap for its P value.

### Stage 5 — rx_top.v status (2026-09-11)

`rx_top.v` wires the whole receiver. It RUNS end to end under Verilator
against a behavioural FFT stub (`tb/rx/stubs/xfft_256.v`, simulation only --
never synthesize it).

**What is proven:**
- Detection, replay, CFO, FFT, grid split, channel estimation,
  equalization and BPSK header decode all work in one piece: the header
  comes out **2000 / qam64 / crc16 / rs_m8 / conv_v27**, exactly Python's.
- Every count matches Python: 6 FFT symbols, 4 payload symbols, 864
  demapped symbols, 2534 Viterbi input symbols, 316 deinterleaver units.

**15/15 bit-exact at 2048 bytes** -- 5 packets on each of qpsk (76
payload symbols), qam16 (38) and qam64 (26), every byte matching Python,
`fifo_overflow=0`. Plus **18/18 on a smaller sweep** (`rtl/run_frame.py`): qpsk/qam16/qam64
x 64/512/2000 payload bits x cfo 0.0/0.05, header and every deinterleaved
byte matching Python. That sweep uses an FPGA-ONLY config -- no rs_m8, no
crc16 -- which also makes the decode order `conv_v27 -> deinterleave`,
exactly what the fabric does, rather than the two-stage order of the full
link. Run it with `python run_frame.py --bits N --cfo F --modem M`.

**The bit FIFO must hold a WHOLE FRAME, not a symbol.** Sized at 8192
bits it wrapped silently on a 2048-byte qam64 frame -- 26 x 1296 = 33696
coded bits into an 8192-bit ring, `fifo_wr` landing at exactly
33696 mod 8192 = 928 -- and the receiver emitted 561 of 2048 bytes with
no error reported anywhere. The demapper averages ~4.5 bits/clock against
the Viterbi's 2, so the backlog grows for the whole frame and never
drains between symbols. Now sized from
`MAX_PAYLOAD_SYM * N_DATA * 6` (165,888 bits, ~7 BRAM36 -- expensive, and
only because this model runs 1 sample/clock; at ~10 clocks/sample it
collapses to a small buffer), with a sticky `fifo_overflow` output so the
loss can never be silent again. **Check that flag before trusting any
decode.**

`MAX_PAYLOAD_SYM` is 128, matching `Ofdm.MAX_PAYLOAD_SYMBOLS`. At 64 it
capped QPSK at 1727 bytes, which a 2048-byte packet quietly exceeded.

**The payload is bit-exact.** 316/316 deinterleaved bytes match Python on
both a clean frame and a cfo=0.05 frame, with the header decoding
correctly in both.

**The scaling was derived, not tuned.** `xp.fft.fft()` does no
normalisation and the Xilinx core is configured `scaling_options=
unscaled`, so both sides grow by the same FFT gain; the RTL differs only
by the int16 full scale, giving `rtl = python * 2^15`. Downstream blocks
want Q12, so the shift is 15-12 = **3**. The original 14 left the payload
~700x small with correlation 0.18 -- crushed into quantization noise,
which is why BPSK (sign only) still decoded while QAM64 did not.
openofdm does the same thing at `sync_long.v:148`
(`{fft_out_re[22:7], ...}`): a 23-bit unscaled output from a 16-bit
input, shifted by 7 -- the shift is set by FFT growth, never picked.

**Still on the stub.** `tb/rx/stubs/xfft_256.v` is a behavioural DFT. It
shares the unnormalised property that the derivation relies on, but not
the real core's latency, rounding or AXI behaviour under backpressure.
Re-run under xsim against the real IP before trusting this on hardware.

**Six integration bugs this exercise found**, none visible at block
level, all fixed:
1. Three separate phase domains are needed, not one. `phase` counts input
   samples, but cp_fft holds a whole symbol, so the frequency-domain
   blocks need their own counter, and mmse_eq's latency means the
   bit-level stages need a third. Gating on the wrong one silently
   disabled `ls_chanest` entirely, then cost 5 of the header's 216 bits.
2. The header symbol has no pilot equalization, so a symbol-complete
   condition that waited on pilots stalled forever.
3. Payload symbol count must come from the ENCODED length, not
   `payload_len_bits` -- 2 symbols where the frame had 4.
4. Viterbi `last` must be asserted on the final coded symbol or the last
   traceback group never flushes: 2520 bits instead of 2528, leaving the
   deinterleaver one unit short and emitting nothing at all.
5. `pilot_cpe` needed ping-pong buffering: a single buffer cannot accept
   symbol N+1 while rotating symbol N, and in a real receiver symbols
   arrive back to back. The standalone testbench had left a gap between
   symbols and so never saw it.
6. The ping-pong banks must be based at N_DATA, not at `2**ADDR_W`.
   Indexing as `{bank, addr}` put bank 1 at offset 256 in a 432-entry
   array, so the last 40 writes of every ODD symbol fell off the end --
   correlation 1.000000 on even symbols and 0.91 on odd ones, which is
   what pointed at the bank rather than at the arithmetic.

**Still host-supplied** via `cfg_*` ports: encoded bit length and the
deinterleaver grid. Deriving them in fabric needs the rs_m8/conv_v27
geometry, and that is a real open item -- a receiver decoding a peer's
arbitrary FEC choice from the header cannot ask the host first.

### Stage 2 in detail — how it was done

The point of this stage is to test the nine existing blocks *against each
other* before writing four more against untested assumptions. You do not
need a complete RTL chain to do it: Python fills the gaps. The existing
per-block pattern (Python → `.hex` → xsim → `.txt` → Python score)
already supports chaining; this is a driver script, not new
infrastructure.

1. Generate one frame with the Python TX — fft=256, cp=32, qam64,
   rs_m8+conv_v27, crc16 — **clean: no CFO, no noise**. Capture Python's
   intermediate array at every block boundary as the golden trace.
2. Run the existing RTL blocks in sequence, Python performing the missing
   stages in between, comparing against the trace **at every boundary**.
3. Success = correct payload bits out. A clean frame isolates plumbing
   and convention bugs (Q-format, bit order, valid timing, symbol
   boundaries) from DSP bugs — one hard problem at a time.
4. Only then add CFO and AWGN, at the levels the real Pluto link sees.

Then replace Python stages with RTL one at a time, re-running this same
harness after each. Order matters: **datapath first (stage 3), control
last (stage 4)** — a control bug and a datapath bug at the same time are
very hard to separate.

This harness is not throwaway. It is what proves a block is still correct
when the hand-written Viterbi is later swapped for the Xilinx core.

---

## 2. The scope boundary (this is the part that keeps getting missed)

**Reed–Solomon and CRC are NOT going in the FPGA. Neither is the MAC.**
This is a settled decision. Do not propose an RS decoder, a CRC checker,
or a Berlekamp–Massey/Chien/Forney block. Do not "notice" that RS is
missing from the RX chain and file it as a gap — it is not a gap.

The fabric implements the PHY from antenna through the deinterleaver.
Everything past that runs on the host (Pi-5).

```
  ── FPGA (XC7A50T) ──────────────────────────────────────┐
  sync → CFO est/corr → CP-strip/FFT → grid extract →     │
  chan est → equalize → pilot CPE → demap →               │
  Viterbi (conv_v27) → deinterleave                       │
  ───────────────────────────────────────────────────────┘
        │  PDU bits over PCIe-DMA / USB
        ▼
  ── host (Pi-5) ─────────────────────────────────────────┐
  Reed–Solomon (rs_m8) → CRC → MAC (ARQ/timers/UM/AM)     │
  ───────────────────────────────────────────────────────┘
```

**Note the order: Viterbi comes BEFORE the deinterleaver, not after.**
This is a concatenated code, not an 802.11-style intra-symbol bit
interleaver. The interleaver sits between the two FEC codes so that
Viterbi's output burst errors are spread across many RS codewords before
RS sees them. `deinterleaver.v`'s header comment records the measured
case: bursts concentrated in a narrow byte range blow rs_m8's
per-codeword error budget, and `unit_bits=8` matches rs_m8's byte symbol
size (bit-granularity interleaving was measured to make it *worse*).

**Why the line is drawn there.** Everything before Viterbi runs at sample
rate (10 Msps target, 20 Msps long-term) and has no choice but to be in
fabric. After Viterbi the data rate collapses to PDU rate — ~1.7 MB/s at
10 Msps QPSK r=1/2 — which the host handles comfortably, and where
spectracuda already has fast, hardware-validated implementations (batched
RS via one C call per PDU, CRC numba kernel). Artix-7 has no hard
processor, so anything stateful and control-shaped is the wrong shape for
this part anyway.

The same principle already shows up inside a block: `sc_sync` emits P and
R and lets the host form |P|²/R², because a divider costs ~3,900 LUT in
fabric and the value is needed once per frame. See `rtl/check.py`'s
docstring. When in doubt, ask "does this run at sample rate?" — if not,
it belongs on the host.

**`rtl/rtl/rx/crc_gen.v` exists anyway.** It was built and verified before
the boundary settled. It is not part of the RX scope. Leave it; don't
wire it in, and don't treat its existence as evidence CRC is in scope.

**Only RX is built so far.** TX is a stated goal (section 0) but no TX
chain exists yet, and section 5 tracks RX work only.

---

## 3. Verilog, not HLS

The HLS C++ track was tried and dropped. `src/sc_sync.cpp` is the only
HLS block that was ever written; `rtl/src/` is the real work. Read
`src/sc_sync.cpp` only if you care about the comparison below — it is not
part of the build.

Measured on the same part (xc7a50tcsg325-1) at the same 100 MHz, both
meeting timing:

| | HLS post-synth | hand Verilog |
|---|---|---|
| LUT | 5,028 | **596** |
| FF | 6,251 | 541 |
| DSP | 32 | 17 |
| WNS | +1.144 ns | +1.171 ns |

8.4x on LUT. **But 78% of the HLS design (3,944 LUT) is a single divider**
inferred from the one `/` at `src/sc_sync.cpp:134`, which is inside the
per-candidate loop and doesn't need to be — peak detection needs only
ordering, and `a/b > c/d` is a cross-multiply when the denominators are
positive. Excluding it the ratio is ~1.8x. Quote 8.4x only with that
caveat attached; the honest reading is that HLS built exactly what the
C++ asked for and the C++ asked for the wrong thing.

Do not quote the older **13.6x** figure. It divided a Vitis *csynth
estimate* (8,081) by a real post-synth number. csynth over-estimated LUT
by 61%.

---

## 4. What is implemented

Fourteen blocks, all scored against spectracuda Python golden vectors
(`golden/`, `rtl/check_*.py`) and all exercised together through
`rx_top.v`. Numbers are post-synth from `rtl/build/<top>_util.txt`. All
four new blocks now synthesize AND close 100 MHz standalone
(`frame_sync` +4.064 ns, `pilot_cpe` +2.395, `grid_extract` +6.511,
`header_decode` +5.997) — see section 6 for what getting there cost.

| Block | File | LUT | FF | DSP | BRAM | Checker |
|---|---|---|---|---|---|---|
| Schmidl–Cox sync | `sc_sync_rtl.v` | 596 | 541 | 17 | 2 | `check.py` |
| CFO estimate | `cfo_estimate.v` | 1477 | 907 | 0 | 0 | `check_cfo.py` |
| CFO correct | `cfo_correct.v` | 1003 | 963 | 2 | 0 | `check_cfo.py` |
| CP-strip + FFT | `cp_fft.v` | 2063 | 3708 | 9 | 1.5 | `check_fft.py` |
| LS channel est | `ls_chanest.v` | 90 | 175 | 6 | 2.5 | `check_ce.py` |
| MMSE equalizer | `mmse_eq.v` | 287 | 114 | 10 | 0.5 | `check_eq.py` |
| Demapper | `demapper.v` | 81 | 22 | 2 | 0 | `check_dm.py` |
| Deinterleaver *(after Viterbi)* | `deinterleaver.v` | 92 | 81 | 0 | 1 | `check_deint.py` |
| Frame sync + replay | `frame_sync.v` | 1942 | 2036 | 0 | 4 | `check_chain.py` |
| Header decode | `header_decode.v` | 31 | 243 | 0 | 0 | `check_chain.py` |
| Grid extraction | `grid_extract.v` | 18 | 139 | 0 | 0 | `check_chain.py` |
| Pilot CPE | `pilot_cpe.v` | 2178 | 2175 | 2 | 1 | `check_chain.py` |
| Viterbi | `viterbi_dec.v` | 3078 | 745 | 0 | 0 | `check_viterbi.py` |
| CORDIC rot/vec | `cordic_*.v` | — | — | — | — | `check_rot.py`, `check_cordic.py` |
| **Top level** | `rx_top.v` | see below | — | — | — | `run_frame.py` |
| *(out of scope)* | `crc_gen.v` | 40 | 18 | 0 | 0 | `check_crc.py` |

**Naive sum of the per-block out-of-context synths: 8,767 LUT, 46 DSP.**
**Do not use it.** It was always a floor — no glue, no backpressure, no
DMA plumbing — and it is now superseded by a real routed measurement of
the whole design:

**Integrated post-route: 13,315 LUT (41%), 12,247 FF (19%), 58 DSP
(48%), 18.5 BRAM (25%), 4,228 slices (52%).** The per-instance split,
which differs from the per-block table above because the tool optimises
across boundaries, is in **section 12**.

The per-block numbers above are still useful for answering "how big is
this block on its own"; they are not a budget and they do not add up to
the design. DSP is the resource that binds — 48% with no TX chain
written, and an IFFT alone is another ~9.

**Only `cp_fft.v` wraps vendor IP** (Xilinx `xfft_256`). Everything else
is hand-written datapath.

**`viterbi_dec.v` is hand-written, and that was a licensing decision, not
a technical one.** The Xilinx Viterbi LogiCORE is licensed, and the
license available here is `License_Type:Hardware_Evaluation`, which
self-expires in hardware — unusable for a shipping design. openofdm
instantiates that core (`verilog/viterbi.v` -> `viterbi_v7_0`), so it is
a reference for how the block sits in a receiver, not for the decoder
itself. **The intent is to swap in the Xilinx core once the license
situation is resolved**, but the hand-written decoder is worth keeping
either way as a fallback and as a known-good reference. It is a
hard-decision rate-1/2 K=7 decoder, a direct port of
`spectracuda/fec/viterbi.py`'s `conv_v27`.

---

## 5. What remains

The RX datapath is complete and decodes bit-exactly in simulation
(section 1). Everything below is what stands between that and hardware,
roughly in the order it should be done.

1. ~~**Close timing on the INTEGRATED design.**~~ **DONE** at 100 MHz
   out-of-context, WNS +0.087 ns, 0 failing endpoints — section 12.
   What replaces it: **close timing again after milestone 9
   integration**, where the PCIe/DMA/AD9361 logic joins it on a device
   already at 52% slices. Section 12's two hard-macro levers (BRAM
   output register, DSP MREG) are the headroom in hand, worth ~2 ns
   each, and are also what a 125 MHz target would use.
2. **Run against the real `xfft_256` under xsim.** Everything so far used
   the behavioural stub in `tb/rx/stubs/`. The scaling was derived rather
   than tuned so it should hold, but the core's latency, rounding and AXI
   backpressure are unexercised.
3. ~~**Test with noise.**~~ **DONE** (section 7) — QPSK and QAM16
   bit-exact to 20% EVM at 2048 bytes, QAM64 to ~0.12 in the fabric
   config. Fixed-point measured NOT to be the limiter (flat 0.024
   quantisation floor). Above ~0.14 Python and RTL fail together — that
   is the no-RS config's limit, not an RTL defect.
4. **Place & route, then bitstream** (stages 7-8).
5. **Pilot slope / SFO tracking** — the long-packet limiter. Python
   first, then HDL. See section 6; it is a restructure of `pilot_cpe.v`,
   not an addition.
6. **Derive the frame geometry in fabric.** `cfg_encoded_bits`,
   `cfg_di_units`, `cfg_di_rows`, `cfg_di_cols` are host-supplied ports
   today. A receiver decoding a peer's arbitrary FEC choice from the
   header cannot ask the host first, so this has to move into the
   fabric eventually — it needs the rs_m8/conv_v27 length geometry.
7. **Shrink the bit FIFO.** ~7 BRAM36 today, and ~2.6 of that is pure
   waste from rounding the ring up to a power of two. It only exists
   because this model runs 1 sample/clock; folding to the real
   cycles-per-sample budget should mostly remove it.
8. **TX chain** — not started (stage 11).
9. **Refactor the top level** — `rx_top.v` carries 14 `always` blocks of
   glue, seven of them position counters that exist only because the
   inter-block streams carry no `sof`/`last`. Sideband plus two grouping
   wrappers; see section 11. Lowest priority of this list, but it
   removes the bug class behind stage 5's integration bugs 1-3.

**Use `reference/openofdm` where it helps.** It is a working 802.11
receiver, so it has proven implementations of several of these:

| Topic | openofdm reference |
|---|---|
| pilot slope / phase tracking | `equalizer.v:708`, `phase.v` |
| top-level frame FSM (for comparison) | `dot11.v` |
| fine timing off the long preamble | `sync_long.v` |
| FFT output scaling | `sync_long.v:148` |

Structure only — the algorithms stay spectracuda's (section 0). openofdm
is 802.11a/g/n, so its framing, SIGNAL field and scrambler are NOT ours.

---

## 6. Synthesis — what it caught that simulation could not

Everything in section 5 passed bit-exact simulation for days before any
of this was found. **Lint-clean under Verilator and bit-exact against
Python say nothing about whether a design is buildable.** Six bugs, all
in the blocks written here, none visible in simulation.

### Bugs that stopped the build

1. **Vivado could not parse five files.** They use SystemVerilog sized
   casts, `WIDTH'(expr)`; Vivado reads `.v` as Verilog-2001 and rejects
   them. Verilator accepted all 36 silently. Fixed by marking those
   files `file_type SystemVerilog` in `synth_any.tcl` /
   `synth_rx_top.tcl` rather than rewriting verified RTL.

2. **`header_decode` synthesized to 0 LUT — the whole block vanished.**
   Its ROMs were declared `reg mask [0:N-1]` with no explicit width.
   Vivado's `$readmemh` rejects that ("malformed $readmem task: invalid
   memory name"), silently left them unloaded, `sel` had no driver, and
   the optimiser deleted everything. **It looked like an impressively
   small block.** Declare 1-bit memories as `reg [0:0]`. In hardware this
   receiver would never have decoded a header.

3. **The bit FIFO was not a memory.** 1 bit wide, written 6 bits per
   cycle — a six-write-port RAM. Verilator simulated it happily; Vivado
   refused outright. Restructured to one 6-bit demapper word per entry.
   `n_bits` is always even (2/4/6), so a word holds a whole number of
   rate-1/2 symbols and no symbol straddles a word: the read side needs
   a sub-counter, not a barrel shifter.

### Bugs that built but would not run

4. **A divider, in a project whose whole premise is not having one.**
   `n_pay_sym = ceil(encoded_bits / bits_per_sym)` — one `/` became a
   96-deep CARRY4 chain, a 56.19 ns path, **WNS -46.18 ns**, and most of
   the top level's LUTs. Replaced by an accumulator: add `bits_per_sym`
   until it reaches the encoded length, at most 128 clocks, started at
   `hdr_done` and needed ~288 clocks later.

   This is the third time a divider has cost this project dearly --
   `sc_sync`'s metric divide went to the host for the same reason, and
   78% of the abandoned HLS design was a divider inferred from one `/`.

5. **Squaring to answer an ordering question** -- in BOTH new DSP-heavy
   blocks. See below; this was the big one.

6. **A missing pair of parentheses.** `r >>> 1 + r >>> 2` parses as
   `r >>> (1 + r) >>> ...` in Verilog: binary `+` binds TIGHTER than
   `>>>`. The threshold became a data-dependent shift, i.e. zero.
   **Every functional test still passed** -- with the gate stuck open,
   the argmax on |P| alone still found the right peak. The only symptom
   was 10.6 ns of unexplained delay. Parenthesise every shift.

### The big one: stop squaring

`|P|^2 / R^2 > T` is the same decision as `|P| > sqrt(T) * R`, and
`|mean|^2 > T^2` is the same as `|mean| > T`. Squaring buys nothing and
costs multipliers, width and delay -- and because a squared value swings
much harder with signal level, it also forces a dynamic normaliser.

**openofdm never squares** (`sync_short.v:254`):

```verilog
prod_thres <= mag_sq_avg[31:1] + mag_sq_avg[31:2];   // 0.75 x avg, shifts
if (delay_prod_avg_mag > prod_thres)                  // registered compare
```

A pipelined magnitude, a threshold built from shifts, compared
register-to-register.

What ours cost before and after:

| | LUT | DSP | WNS |
|---|---|---|---|
| `frame_sync` squared | 903 | **31** | -1.848 |
| `frame_sync` magnitude | 1942 | **0** | **+4.064** |
| `pilot_cpe` squared | 2441 | 4 | -1.390 |
| `pilot_cpe` magnitude | 2178 | 2 | **+2.395** |

`frame_sync` had been doing three 48x48 squarings, a 96-bit leading-one
search, a barrel shift and two cross-multiplies IN ONE CYCLE.
`pilot_cpe` was squaring a vector whose magnitude its own CORDIC had
already computed -- `cordic_vec` now exposes `out_mag` (always computed
internally as `x[STAGES]`, just never brought out), so the gate is a
compare against a generated constant with the CORDIC gain folded in.

**Trading DSP for LUT is the right direction on this part**: DSP was at
76% and binding, LUT at 55%.

### Two more things worth keeping

- **Vivado merges registers.** It collapsed `pilot_cpe`'s buffer-output
  register into `cordic_rot`'s input register, so BRAM clock-to-out fed
  the CORDIC's quadrant pre-rotation -- a 10.6 ns path. Needed two
  genuinely separate stages, not one.
- **Validate index alignment against real IQ, not by reasoning.** The
  rewritten `frame_sync` came out with `start_index` 199 against Python's
  200: the delay line matching the CORDIC was `STAGES+2` deep when
  `cordic_vec` is 1 load cycle + STAGES. Nothing but a comparison against
  real frames would have caught a one-sample error.

### Timing closure — the whole-design story

Synthesis and P&R of `rx_top` (out-of-context, xc7a50t, 100 MHz). Each
row is one fix; the WNS column is post-route.

| what was fixed | LUT | DSP | WNS | TNS | failing |
|---|---|---|---|---|---|
| (first run) | 17,950 | 91 | -22.770 | -16,896 | 5,211 |
| divider -> accumulator; squaring -> magnitude | 18,592 | **58** | -4.846 | -399 | 374 |
| FIFO -> Viterbi skid buffer | 13,698 | 58 | -4.412 | -399 | 374 |
| `cfo_estimate` normaliser pipelined | 13,318 | 58 | -0.340 | -2.17 | 14 |
| `frame_sync` -> `cfo_correct` registered | 13,315 | 58 | **+0.087** | **0.000** | **0** |

**TIMING CLOSES: WNS +0.087 ns, TNS 0.000, 0 failing endpoints of
27,747, DRC clean.** Resources are comfortable: 41% LUT, 19% FF, 48% DSP,
24% BRAM. DSP was the binding resource at 76% until the squaring went.

**But the margin is 0.087 ns, which is nothing.** A placement seed
change, a different Vivado version or any added logic can push it
negative. The remaining worst path is `sc_sync`'s own sample buffer into
its accumulator DSP cascade -- the same memory-output shape as the three
below, so applying that rule there would buy real margin instead of
scraping past. Treat this as "closes", not "closed comfortably".

### THE standing rule this produced

Three of the failing paths were the same shape:

| memory | consumer |
|---|---|
| coded-bit FIFO RAM | `viterbi_dec` traceback SRAM |
| `pilot_cpe` symbol buffer | `cordic_rot` input |
| `frame_sync` sample buffer | `cfo_correct`'s `cordic_rot` input |

**A memory output that feeds a consumer gets its own register stage.**
Writing `out <= mem[addr]` is NOT enough -- Vivado merges that register
with the consumer's input register, so BRAM clock-to-out ends up driving
the consumer's combinational front end. It needs two genuinely separate
stages. Applying this rule exhaustively (there are only a handful of RAMs
here) is cheaper than finding the paths one at a time.

### Read the logic/route split before deciding what to do

The worst path at -4.412 ns was:

    Data Path Delay: 14.357ns   logic 4.125ns (29%)   route 10.232ns (71%)

**71% routing.** "Split the long combinational path" was the wrong mental
model -- there was only 4.1 ns of logic. Registers still helped, but by
breaking a long WIRE into two hops, and because 96 wires (`p_re` +
`p_im`, 48 bits each) crossed from `sc_sync` into `cfo_estimate` only to
be truncated to 20 bits inside. Narrowing an interface can beat
pipelining it.

Corollary: do not conclude "the top level is under-pipelined" from WNS
alone. TNS and the failing-endpoint count said this design was
CONVERGING (5,211 -> 374 -> 14), not systemically broken, which is why
targeting paths one at a time stayed the right call.

### How openofdm avoids all of this

`phase.v` is not a CORDIC -- it is abs -> max/min -> coarse scale ->
divider -> arctan LUT, and every step is registered. Its "normaliser" is
ONE comparison:

```verilog
assign dividend = (max > 4194304) ? min : {min[...], 12'b0};
assign divisor  = (max > 4194304) ? max[31:12] : max[19:0];
```

A single test against 2^22 choosing between two fixed bit-slices --
constant delay, a couple of LUT levels. Ours ran a 48-bit priority
encoder plus a variable barrel shift. openofdm also spends latency
freely where it is free: a 36-cycle divider with a 37-cycle `delayT` to
realign the quadrant. `cfo_estimate` runs once per frame, so its latency
is equally free -- that is why pipelining it cost nothing.

Note openofdm DOES use a divider here. It is a 36-cycle sequential
module, not combinational. "No dividers" in this project really means
**no combinational dividers in the sample path**.

### A test weakness this exposed

Bug 6 passed every test with the threshold disabled. `check_chain.py`
verifies WHERE the peak is, never that the gate REJECTS anything -- so a
`frame_sync` with no threshold at all scores full marks, and would then
false-trigger on noise, which is the one thing the threshold exists to
prevent. **Detection-on-noise is untested.** Feeding a frameless noise
buffer and asserting `detected` stays low is the missing case.

---

## 7. Noise and phase noise — the EVM envelope

Run 2026-09-14. Until this, every result in this file was an ideal
channel: no AWGN, no phase noise.

### READ THIS FIRST: the fabric config is not the link

The RTL stops at the deinterleaver -- **Reed-Solomon and CRC run on the
host** (section 2). So any end-to-end test of the fabric must switch them
off, and that chain is **weaker than the real link by about one EVM
step**. Measured, QAM64 at 2048 bytes:

| EVM | full link (`rs_m8`+`conv_v27`+`crc16`) | fabric config (`fec=none`) |
|---|---|---|
| 0.10 | 4/4 | 4/4 |
| 0.12 | 4/4 | 4/4 |
| 0.15 | **4/4** | 2/4 |
| 0.20 | 2/4 | 0/4 |

`fec=none` means **Reed-Solomon off, Viterbi still on** -- `fec1` stays
`conv_v27`. CRC only detects, it never corrects, so it contributes
nothing here. The whole gap is RS repairing the residual *burst* errors
Viterbi leaves, which is what the block interleaver exists to spread --
the mechanism `abhi/pluto_rx_standalone_v2.py` and `deinterleaver.v`
both record from the real 1024-byte field losses.

**The validated hardware config is `qam64` + `rs_m8` + `conv_v27` +
`crc16` + block interleaver** (`abhi/pluto_rx_standalone_v2.py`), proven
on two Plutos + Pi at 2048 bytes, tx -10 dBm / rx gain 60. Simulation
reproduces it: 4/4 through EVM 0.15.

### Results — 400 packets, 2048 bytes, QAM64, fabric config

`rtl/exp_evm_100pkt.py`, 100 packets at each EVM, 2026-09-14:

| target EVM | measured | python OK | rtl==python | rtl OK |
|---|---|---|---|---|
| 0.12 | 0.1075 | **99/100** | **99/100** | **98/100** |
| 0.14 | 0.1187 | 79/100 | 83/100 | 73/100 |
| 0.16 | 0.1279 | 20/100 | 31/100 | 13/100 |
| 0.18 | 0.1357 | 1/100 | 3/100 | 0/100 |

**The RTL tracks Python, and both fail together after 0.14.** The falling
`rtl==python` rate at high EVM is not the RTL drifting away -- once
Python is failing too, both produce garbage and two garbage streams
rarely match. Judge agreement only where the link works.

**Usable edge for this configuration: measured EVM ~0.12** (99% of
packets), marginal at 0.14 (79%), collapsed by 0.16.

At 0.12 exactly **1 packet in 100** had Python decode correctly while the
RTL did not. That is the quantisation floor at the decision boundary --
consistent with the flat 0.024 RMS measured below. Small, but not zero.

### Smaller sweep, all three MCS

| MCS | packets | both OK | both failed | RTL diverged |
|---|---|---|---|---|
| QPSK | 5 | 5 | 0 | **0** |
| QAM16 | 5 | 5 | 0 | **0** |
| QAM64 | 12 | 6 | 5 | 1 |

- **QPSK**: bit-exact to measured EVM **0.199**; cliff never reached.
- **QAM16**: bit-exact to **0.188**; cliff never reached.

**So: 2048 bytes works in Python and the RTL matches it, up to EVM 0.12
with RS and CRC turned off. Above ~0.14 both fail together.** With RS on
-- which is how the system actually runs -- the same frames decode
through 0.15.

### Fixed-point is NOT the limiter -- measured

Controlled experiment, `rtl/exp_len_vs_precision.py`: ONE frozen channel
realisation (noise and phase arrays drawn once and sliced, sigma fixed,
NOT regenerated per length) reused across packet lengths, so length is
the only variable. QAM64, fabric config, measured EVM ~0.127:

| bytes | payload symbols | py==tx | rtl==py |
|---|---|---|---|
| 256 | 4 | YES | **YES** |
| 512 | 7 | YES | **YES** |
| 1024 | 13 | YES | **YES** |
| 2048 | 26 | NO | NO |
| 4096 | 51 | NO | NO |

Bit-exact at 4, 7 and 13 symbols at the same EVM that fails at 26. And
the quantisation floor is flat:

    constellation |py - rtl| rms = 0.024
      256 B by quarter: 0.0246 0.0242 0.0239 0.0263
     1024 B by quarter: 0.0239 0.0242 0.0239 0.0242

It does not grow through a frame and does not grow with length -- so it
is quantisation, not leaked phase drift. Against a channel EVM of 0.12
it is ~4% of the error power.

**Correction to an earlier reading in this session.** The length effect
above was first written up as evidence for the missing SFO/timing-slope
correction (section 9). That was wrong: **RS is what handles residual
burst errors over long frames**, and it had been switched off. The
pilot-slope gap in section 8 is real and still unmeasured, but this
experiment is not evidence for it.

### Phase-noise model -- read before reusing it

spectracuda has **no phase-noise model**, so it lives in the harness
(`run_frame.py`), not the library. It is a **bounded AR(1)** process,
steady-state RMS = target, `--pn-alpha` default 0.9995.

The first attempt used a Wiener random walk, which was wrong: a random
walk's variance grows without bound, so 0.035 rad/sample accumulated to
~1 rad across a frame and destroyed the header at a nominal "5% EVM".
Real oscillator phase noise is bounded and strongly correlated -- which
is also what lets per-symbol CPE correction track it.

Measured EVM lands below target (0.15 -> ~0.12) because the receiver's
own CPE correction removes part of the phase-noise contribution. Always
quote the MEASURED figure.

### Harness trap worth remembering

`cfg_di_units` must come from the **deinterleaver's actual input size**,
not from `encoded_length()`. The deinterleaver sits AFTER the Viterbi, so
it works on the DECODED stream. Sizing it from the encoded length made it
wait for 513 units when 256 were coming, emit nothing, and look exactly
like an RTL failure under noise -- the debug counters (`det=1`,
`hdr_done=1`, `vit_out=2054`, `di_in=256`, units out 0) are what showed
every stage was actually fine.

---

## 8. HOW TO RE-TEST THIS — read before changing any RTL

The receiver is **locked** as of 2026-09-14: bit-exact against Python,
timing-closed, and characterised under noise. If you change RTL, re-run
these in order. Each one catches a class of bug the others cannot.

    export XILINX_VIVADO=/home/abhi/work/xilinx/2025.2/Vivado
    export PATH=$XILINX_VIVADO/bin:$PATH      # Vivado is NOT on PATH
    cd fpga
    ./setup_ref.sh                            # ONCE: pinned Python reference

**The Python reference is PINNED (2026-10-03).** The RTL implements the
frame format of spectracuda `ad0a396` (uncoded 1-symbol header). Later
commits changed it (e374fdf: CRC-16 + conv_v27 header, then DMRS, C2), so
every harness script imports spectracuda from a worktree of `ad0a396`
via `golden_ref.py` and refuses to run against anything else. Before this
pin existed, run_frame.py silently used the working tree and failed with
"header never decoded" / "2000 payload bits -> 268 encoded" (the header's
conv_v27 call mistaken for the payload's).

**0a. Frequency-domain stage alone (~10 min).** Standalone, against the
committed reference dumps -- no Python:

    python run_fd_stage.py          # 87 scenarios; --quick for 14

**0b. Bit-exact vs the reference dumps, integrated (~10 min).**

    ./check_golden.sh               # fixtures/frames/: all 7 frames at C=1 and C=10

**0. Rate invariance, every C (~25 min).**

    ./rate_matrix.sh      # 9 configs x C = 1, 2.5, 5, 10, 20; exit 0 = all pass

Each run bit-exact vs Python AND FFT output identical to C=1.

**1. Per-block, against golden vectors (seconds).**

    python check_chain.py --trace-dir build/trace_cfo   # 11 RTL stages

Regenerate the traces first if the config changed:
`python -m fpga.sim.trace [--cfo 0.05]`.

**2. End-to-end, ideal channel (~40 s per frame).**

    python run_frame.py --bits 2000  --modem qam64
    python run_frame.py --bits 16384 --modem qpsk        # 2048 bytes

Expect `VERDICT: PASS -- bit-exact`. Sweep `--modem qpsk|qam16|qam64`
and `--bits 64|512|2000|16384`.

**3. Under noise (~40 s per frame).**

    python run_frame.py --bits 16384 --modem qam64 --evm 0.12

Read TWO verdicts, never one:
- `python==tx` NO means the LINK is dead at that noise -- physics, not
  an RTL fault.
- `rtl==python` NO *while* `python==tx` is YES is the only real RTL
  failure.

**4. Statistical, 400 packets in parallel (~25 min).**

    python exp_evm_100pkt.py

100 packets at each of EVM 0.12/0.14/0.16/0.18. Builds 8 worker binaries
and runs them across 8 processes. Compare against the table in section 7.

**5. Timing (~25 min).**

    vivado -mode batch -source impl_rx_top.tcl
    grep -m1 -A2 'WNS(ns)' build/rx_top_impl_timing.txt | tail -1

Expect WNS >= 0. Margin is only +0.087 ns, so ANY added logic can break
it -- always re-run this after an RTL change, not just the simulations.

### Traps that have already cost time here

- **`cfg_di_units` comes from the deinterleaver's ACTUAL input size**, not
  from `encoded_length()`. It sits AFTER the Viterbi, so it works on the
  DECODED stream. Getting this wrong makes it emit nothing and looks
  exactly like an RTL failure under noise.
- **The fabric config is NOT the link.** RS and CRC are host-side, so
  fabric tests run `fec=none` -- about one EVM step weaker than the real
  system. Never quote a fabric cliff as the system's cliff.
- **Parallel runs need per-PROCESS file paths.** The testbench bakes
  stimulus paths in as compile-time defines, so each worker needs its own
  binary. Tying the slot to a job index instead of the process let two
  processes clobber one file.
- **Verilator accepts things Vivado rejects** -- SystemVerilog casts,
  width-less `$readmemh` arrays, multi-write-port memories. Lint-clean
  and bit-exact says NOTHING about synthesisability (section 6).
- **Quote MEASURED EVM, not the target.** CPE removes part of the phase
  noise, so measured runs ~12% below target.

---

## 9. Pilot tracking — half done, and it is the long-packet limiter

`pilot_cpe.v` is bit-exact against Python. The gap is in the MODEL, so
the fabric inherits it. **Agreed plan: fix this in Python first, then
bring the HDL up to match** — same rule as section 0, the HDL never gets
ahead of the hardware-validated model.

### What we do vs what openofdm does

| | ours | openofdm |
|---|---|---|
| common phase (CPE) | ✅ scalar rotation per symbol | ✅ `cpe` |
| timing slope (SFO) | ❌ measured in debug only, never applied | ✅ `cpe + pilot_idx*peg`, `equalizer.v:708` |
| channel estimate refresh | ❌ training symbols only, never updated | partial (smoothing) |
| pilot values | constant `1+0j` every symbol | BPSK ±1, **127-length polarity sequence** |

openofdm's signed pilot indices (+8, +22, −20, −6) and its `Sxy`
accumulator (`equalizer.v:152`) are the tell for a least-squares slope
fit — signed indices are only needed to fit a ramp across frequency.
802.11 varies pilot polarity specifically to avoid a spectral line at the
pilot subcarriers; a receiver must know the sequence to un-rotate them
before measuring phase. Ours being a constant is a deliberate
simplification, not an oversight, but it is a real difference if anyone
compares spectra.

### Why this is the long-packet limiter

`MAX_PAYLOAD_SYMBOLS = 128` exists for exactly this reason. From its own
comment in `pipeline/ofdm.py:246`:

> *"beyond roughly this many OFDM symbols, the RF channel has likely
> changed enough (coherence time) that the single channel estimate taken
> from the training symbol(s) at the start of the frame no longer
> applies, and/or sync has drifted — there's no per-symbol tracking to
> compensate"*

Scale, from the measured 2000 bits → 5068 encoded → 4 symbols:

| PDU | payload symbols |
|---|---|
| 64 B | ~1 |
| 1024 B | ~16 |
| 2048 B | ~32 |

**This matches the field observation: 64-byte PDUs decoded fine while
1024/2048-byte packets did not.** One symbol gives drift no time to
accumulate; thirty-two gives it plenty.

**Careful with attribution, though.** The 1024-byte losses were ALSO
root-caused to the interleaver (burst Viterbi errors blowing rs_m8's
per-codeword budget — see `deinterleaver.v`'s header). Both very likely
contributed. Measure before designing: Python already computes
`pilot_timing_slope_per_symbol` under `debug_payload_symbols`, so a long
capture will show how much is slope versus stale channel estimate.

### What is still missing, in priority order

1. **Timing slope (SFO) correction.** `ofdm.py:1234-1240` says outright
   *"This is NOT corrected by the CPE fix above (that's a per-symbol
   scalar rotation only)"*. Measured, never applied.
2. **Channel-estimate refresh across the burst.** The estimate comes
   entirely from the training symbols at frame start and is reused for
   every payload symbol. The per-symbol pilots could track it and
   currently do not. The stale class docstring at `ofdm.py:168-172` still
   claims pilots are *"not yet used for anything"* — that predates the
   CPE work and should be fixed when this is.

### ARCHITECTURAL WARNING for the HDL side, when the time comes

`pilot_cpe.v` sums the pilots BEFORE the CORDIC, because
`angle(sum) == angle(mean)` and that saved a divider. **That optimisation
destroys exactly the information a slope fit needs** — the individual
pilot phases. Adding slope means one CORDIC per pilot (8 angles), a
least-squares fit, and a per-subcarrier rotation ramp instead of a
constant angle. Budget it as a restructure of the block, not an addition
to it.

---

## 10. Deliberate deferrals — do not "fix" these

- **Hard-decision Viterbi, not soft-decision.** openofdm uses
  soft-decision and it is worth real coding gain, so this will look like
  an obvious improvement. It is deferred on purpose: **the Python golden
  model is hard-decision too**, and this is the first Verilog RX chain, so
  the HDL tracks the validated model rather than getting ahead of it.
  Soft-decision moves when spectracuda moves. Note the same item is open
  on the Python side for LDPC (the soft-LLR step), so improving the model
  first benefits both.
- **The `sc_sync` metric divide stays on the host.** The RTL emits P and
  R; |P|²/R² is formed off-chip. See section 2.

---

## 11. Gotchas for a new agent

- **`fec0`/`fec1` naming is inverted from textbook convention.** This
  repo follows liquid-dsp: **`fec0` is called the "inner" code and is
  `rs_m8`; `fec1` is called the "outer" code and is `conv_v27`**
  (`examples/drone_air_unit.py:73`). `packetizer.py` and
  `header.py:102` agree on this. But note that `fec1`/"outer" is the code
  *closest to the channel* — it is encoded last and decoded first — which
  is the opposite of the usual concatenated-coding meaning of "outer".
  Decode order is `fec1` (Viterbi) → deinterleave → `fec0` (RS) → CRC.
  **Prefer naming the codec ("Viterbi", "RS") over the field number**;
  convention mismatches of exactly this kind already caused two real bugs
  in the AFF3CT bridge work, and cost this doc a wrong chain order on its
  first draft.
- **Report filenames are inconsistent.** Most blocks write
  `<top>_util.txt` / `<top>_timing.txt` via `synth_any.tcl`, but
  `synth_sc_sync.tcl` writes generic `rtl_utilization.txt` /
  `rtl_timing.txt`. This has already cost one session an hour of hunting.
- **A Vitis csynth number is an estimate, not a measurement.** It said
  8,081 LUT where Vivado measured 5,028.
- **Check whether a utilization report had black boxes before quoting
  it.** `reference/openofdm/vivado_synth/utilization_report.txt` reports
  5,564 LUT because `xfft` and `viterbi` were left unresolved; the real
  figure is 15,810 (`utilization_report_v2.txt`). Only the
  `create_project` + `launch_runs` flow resolves the encrypted IP; a raw
  `synth_design` call does not.
- **Vivado is installed but invisible to `which`** -- see section 8. Do
  not conclude it is missing; source its `settings64.sh`.
- **openofdm is read-only reference.** `reference/openofdm/` is there for
  structure. Never modify it.
- **Regenerating openofdm is currently blocked** — the Xilinx Viterbi
  LogiCORE license fails to check out (FlexNet -5,357). Our own
  `xc7a50t` synthesis is unaffected (built-in free license).
- **`docs/vitis-hls-ofdm-ip-plan.md` is stale in two places.** It was
  written for the HLS track, and its section 1.1 sizes a Reed–Solomon
  decoder in fabric. Both are superseded by sections 2 and 3 above. The
  plan doc is still useful for the folding/cycles-per-sample analysis and
  the fixed-point word-length study.

---

## 12. Reference

### Toolchain -- Vivado is installed, but NOT on PATH

```
/home/abhi/work/xilinx/2025.2/Vivado/bin/{vivado,xsim,xvlog,xelab}   v2025.2
source /home/abhi/work/xilinx/2025.2/Vivado/settings64.sh
```

**It is not on PATH and not in /opt, /tools or ~/Xilinx.** A `which
vivado` returns nothing and a shallow `find` misses it -- one session
concluded from exactly that "there is no Vivado on this machine" and
planned around a blocker that did not exist. Source `settings64.sh`
first, then everything works: synthesis, implementation and xsim.

No licence is needed for our own flow: `xc7a50t` is covered by Vivado ML
Standard's free built-in licence and `xfft_256` is a no-charge core (its
generated `build/ip/xfft_256/xfft_256.dcp` is proof it built fine). See
section 7 for what the Viterbi LogiCORE licence does and does not gate.


- Golden model: `spectracuda.pipeline.ofdm.Ofdm` — `rx_process()`,
  `_decode_header_from_sync()`, `_decode_payload_from_header()`.
- Config in use: fft=256, cp=32, QPSK/16QAM, `rs_m8`+`conv_v27`, crc16.
- Target: Artix-7 XC7A50T — 32,600 LUT6, 65,200 FF, 120 DSP48E1, 75
  BRAM36, no hard processor.
- The IP sits between the AD9361 HDL and the PCIe-DMA + USB HDL in an
  existing Vivado design. That surrounding infrastructure is not ours to
  size.

---

## 13. Structure — why `rx_top.v` draws busier than `dot11.v`

Recorded 2026-09-14, from a block-diagram comparison against openofdm.
The question that prompted it: *our diagram looks complex and clumsy next
to theirs — is that because we split into more granular blocks?*

Partly. But granularity is the smallest of three causes, and only the
third is a real problem. **Nothing here says the design is wrong** — it
is bit-exact and it now synthesizes (section 6). This is about structure,
and it is recorded because the same "why is ours messier" question will
be asked again.

### 1. Most of it is a drawing mismatch, not a design difference

The comparison that produced the impression drew **our instances plus
glue** against **their module list**. That is not like for like.
openofdm's boxes are big:

| openofdm box | lines | what is inside it |
|---|---|---|
| `equalizer.v` | 880 | channel est, smoothing, CPE, SFO/LVPE, pilot polarity, 2 RAMs, rotator, 3 dividers |
| `sync_long.v` | 570 | LTS fine timing, rotator, **the FFT itself** |
| `ofdm_decoder.v` | 216 | demod, deinterleave, Viterbi, descramble, bits_to_bytes — a pure wrapper |

Collapse ours to that level and it is *smaller* than theirs:

```
ours, grouped their way                openofdm
─────────────────────────              ──────────────────────────
sync + CFO ............ 4 blocks       sync_short
CP strip + FFT ........ 1 block        sync_long      (FFT inside)
equalize .............. 6 blocks  ──►  equalizer      (880 L monolith)
demap → decode → deint  3 blocks  ──►  ofdm_decoder   (216 L wrapper)
header + frame FSM .... 1 block        dot11 FSM + phy_len + rate_to_idx

        5 boxes                                6 boxes
```

Their *source* is no cleaner: `dot11.v` is 1124 lines of which **628 are
one `always` block**. It draws as one tidy box because a diagram draws an
FSM as one box.

**Do not conclude from a diagram that openofdm is better factored.** It
is better *grouped*.

### 2. We have no intermediate hierarchy layer — real, and cheap

openofdm has a middle tier; we do not. `ofdm_decoder.v` is a 216-line
wrapper that exists for no reason except to make five blocks into one
box. `rx_top.v` goes straight from top to leaf, so all 14 leaves and
every piece of glue surface at once.

### 3. The top level does datapath work — real, and it has already cost bugs

This is the one worth acting on. Compare what each top level *does*:

| | `rx_top.v` | `dot11.v` |
|---|---|---|
| instantiations | 15 | 10 |
| `always @(posedge clk)` blocks | **14** | **2** |
| `reg` declarations | 33 | — |
| datapath at top level | H-hold RAM, bit FIFO (write + read sides), bit→byte packer | none |

`rx_top.v` is not a wiring file. It is an unnamed 15th block. Of its 14
always blocks: 4 are real datapath, 3 are control (phase FSM,
`dm_scheme`, the `n_pay_sym` accumulator), and **7 are bare position
counters** —

```
fft_sof/fft_bin · out_sym · h_bin · d_rd/p_rd ·
eq_dcnt/eq_pcnt/eq_sym · hdr_cnt/hdr_sof · push_cnt
```

Those seven exist for one reason: **the inter-block streams carry
`valid` and nothing else.** No `sof`, no `last`, no symbol-type tag. So
the top level has to independently re-derive "where are we in this
symbol" at three different points in the pipeline. That is precisely the
three-phase-domain problem the stage 5 notes describe, and integration
bugs 1, 2 and 3 recorded there are all symptoms of it — gating on the
wrong counter silently disabled `ls_chanest` entirely, then cost 5 of the
header's 216 bits.

openofdm's blocks are self-locating: strobes carry position and each
block tracks itself, so `dot11.v` only has to hand out enables.

**Not a cause:** the doubled instances (`grid_extract` ×2, `mmse_eq` ×2)
add four boxes to the drawing but were deliberate and are documented in
`rx_top.v`'s own header — H is re-split through a second `grid_extract`
so no third index table is needed, and Python equalizes data and pilots
against genuinely different estimates. openofdm serialises the
equivalent inside one module. Leave both alone.

### What to do about it

In this order, and **not while timing closure is in flight**:

1. **Add a sideband to the inter-block streams** — `sof`, `last`, and a
   2-bit symbol type (train / header / payload). Each block then tracks
   its own position and the seven counters collapse.
2. **Group into two wrappers:** `rx_freq_domain.v` (both `grid_extract`,
   `ls_chanest`, both `mmse_eq`, `pilot_cpe`, the H hold) and
   `rx_bit_decoder.v` (demapper, bit FIFO, `viterbi_dec`, packer,
   `deinterleaver`). `rx_top.v` drops to ~6 instances and near-zero glue.
3. The H-hold RAM, the bit FIFO and the bit→byte packer move inside
   those wrappers with it.

**This is a refactor of working, bit-exact, now-synthesized RTL.** Run
`run_frame.py` after every single step, and re-check area and timing at
the end — moving logic across a module boundary changes what the tool
can flatten and retime.

**Priority: below finishing timing closure and below the noise sweep.**
It buys legibility and it removes a class of bug that has already fired
three times; it does not buy a working receiver, which is what the items
in section 5 buy.

---

## 14. Timing closure — how it was done, and what is left

**TIMING CLOSES.** `rx_top` meets setup, hold and pulse width at 100 MHz
on `xc7a50tcsg325-1`, DRC clean, with all 9 functional cases still
bit-exact. Final run 2026-09-14 05:40. Evidence:
`rtl/build/rx_top_impl_{timing,util,util_hier}.txt` and
`rx_top_routed.dcp`.

**Read the caveat in "What closure does NOT mean" at the end of this
section before quoting this anywhere.**

### Final numbers

```
Setup   WNS  +0.087 ns    TNS 0.000        0 failing / 27,747
Hold    WHS  +0.031 ns    THS 0.000        0 failing / 27,747
Pulse   WPWS +3.750 ns    TPWS 0.000       0 failing / 13,760
DRC     0 errors
```

| resource | used | available | % |
|---|---:|---:|---:|
| Slice LUTs | 13,315 | 32,600 | 40.8% |
| &nbsp;&nbsp;as logic | 12,287 | | 37.7% |
| &nbsp;&nbsp;as memory | 1,028 | 9,600 | 10.7% |
| Slice Registers | 12,247 | 65,200 | 18.8% |
| **Slices (occupancy)** | **4,228** | **8,150** | **51.9%** |
| &nbsp;&nbsp;SLICEM | 1,305 | 2,400 | 54.4% |
| **DSP48E1** | **58** | **120** | **48.3%** |
| Block RAM tile | 18.5 | 75 | 24.7% |
| F7 / F8 muxes | 51 / 0 | | ~0% |

**DSP binds, not LUT.** 40.8% LUT reads comfortable; 48.3% DSP with no TX
chain written is the number to plan against — an IFFT alone is another
~9. Note also the LUT-vs-slice gap: 40.8% of LUTs but **51.9% of
slices**, because packing strands LUTs. **Quote slice occupancy, not
LUT%, when asking whether there is room.**

### Where it goes, by instance

| instance | LUT | LUTRAM | FF | BRAM36/18 | DSP |
|---|---:|---:|---:|---|---:|
| `u_vit` | 3,060 | 172 | 741 | — | 0 |
| `u_cpe` | 2,089 | 0 | 2,137 | 0/2 | 2 |
| `u_fft` | 1,993 | 31 | 3,664 | 0/3 | 9 |
| `u_fs` | 1,810 | 0 | 2,008 | 4/0 | 0 |
| `u_cfo_est` | 1,125 | 0 | 1,135 | — | 0 |
| `u_cfo_cor` | 1,023 | 0 | 948 | — | 2 |
| `u_sync` | 793 | 0 | 539 | 0/4 | **17** |
| `u_eq_pilot` | 383 | 0 | 112 | 0/1 | 10 |
| `u_eq_data` | 330 | 0 | 110 | — | 10 |
| `(rx_top)` glue | **295** | 216 | 247 | 6/0 | 0 |
| `u_dm` | 98 | 0 | 21 | — | 2 |
| `u_ce` | 93 | 0 | 170 | 0/5 | 6 |
| `u_di` / `u_hdr` / `u_grid` / `u_grid_h` | 85 / 78 / 42 / 29 | 0 | | 1/0 | 0 |

DSP concentration: `u_sync` 17 + `u_eq_data` 10 + `u_eq_pilot` 10 +
`u_fft` 9 + `u_ce` 6 = **52 of 58**. Those are the only meaningful
folding targets if DSP ever has to come down.

### The route from -22.77 ns to +0.087 ns

| step | WNS | TNS | failing | LUT |
|---|---:|---:|---:|---:|
| first integrated P&R | -22.770 | -16,896.9 | 5,211 | 17,950 |
| after the sc_sync squaring fix | -4.846 | -4,075.0 | 2,121 | 18,592 |
| after the bit-FIFO synchronous read | -4.412 | -399.0 | 374 | 13,698 |
| after pipelining `cfo_estimate` | -0.340 | -2.2 | 14 | 13,318 |
| after registering `cfo_correct`'s input | **+0.087** | **0.000** | **0** | 13,315 |

Four fixes. Note the LUT column: the FIFO change alone removed **4,894
LUTs** and took the top-level glue from 3,775 to 295, LUTRAM from 3,875
to 419, and F8 muxes from 162 to 0.

### The one rule that would have caught three of the four

Three of the four worst paths were the same shape:

```
  memory output  ──►  a consumer's INPUT ARITHMETIC  ──►  register
                      (nothing registered in between)
```

1. `bit_fifo` (LUTRAM) → the Viterbi's ACS
2. `pilot_cpe`'s sample buffer → `cordic_rot`'s 1/K prescale multiply
3. `frame_sync`'s BRAM → `cfo_correct` → **the same** `cordic_rot`
   prescale multiply

**Cases 2 and 3 are one defect at two call sites.** `cordic_rot.v:82-83`
multiplies its INPUT PORT combinationally:

```verilog
wire signed [DATA_W+16:0] px = in_x * `CORDIC_INV_K;
```

and `cfo_correct.v:69` hands it `.in_x(in_i)` straight from its own port
with no register anywhere in the block. So the multiply starts at
whatever drives `cfo_correct` — a BRAM on the far side of the die.
Fixing the pilot_cpe *link* left the `cordic_rot` *block* alone, and it
resurfaced at the other instantiation four hours later.

**So the useful rule is not "register every interface", and not even
"register every memory output". It is: a block with arithmetic on its
input port must not be fed from a memory across a boundary.**

`sc_sync_rtl` already does this correctly and is the in-house pattern to
copy — `s0_m_i <= buf_i[rp_mid]` fetches, *then* multiplies.

**One latent instance remains.** `rx_top.v:413` and `:420` pass
`.h_re(h_data_re[d_rd])` — an asynchronous array read straight into
`mmse_eq`'s complex multiply. It does not fail today because those
arrays are small LUTRAM placed beside the equalizers, but it is the same
shape. Register it when you are next in there.

**Caveat on the rule as a prescription.** It would have *flagged* the bit
FIFO but given the wrong fix: registering the output would not have
helped, because the address-decode mux sat *before* the register. Making
the read synchronous did. Good detector, bad prescription — see the
detail below.

### Why the bit FIFO was the big one

`rx_top.v:592` used to read asynchronously:

```verilog
reg  [5:0] bit_fifo [0:FIFO_DEPTH-1];   // 128*216 = 27,648 deep
wire [5:0] rd_word = bit_fifo[f_rd];    // <-- asynchronous
```

**Block RAM cannot do an asynchronous read**, so Vivado had no option but
distributed RAM, and built a 27,648-deep LUTRAM with a combinational
address-decode mux tree. The worst path's primitive list was the
signature — `RAMD64E=1 MUXF7=3 MUXF8=1 LUT6=4 LUT4=2 LUT3=1 CARRY4=2`,
14 levels, 14.142 ns of which **74% was routing**, because distributed
RAM only lives in SLICEM columns and 3,672 of them pinned 78% of the
device's SLICEM.

A synchronous read moved it to ~6 BRAM36 and deleted the mux tree. The
projection recorded here before the run was ~14,900 LUT and ~19.5 BRAM;
actual was **13,318 LUT and 18 BRAM** — better than projected.

### The remaining path is at the silicon floor

```
Source:       u_sync/buf_i_reg_2/CLKBWRCLK        (RAMB18E1)
Destination:  u_sync/s1_aout_re_reg/PCIN[0]       (DSP48E1 cascade)
Slack: +0.087 ns   Data path 8.429 ns   logic 6.305 (75%)  route 2.124 (25%)
Logic Levels: 1  (DSP48E1=1)

  RAMB18E1 CLKBWRCLK -> DOBDO[9]     2.454 ns   <- no output register
  route -> s0_m_i[9]                 2.122 ns   <- BRAM column to DSP column
  DSP48E1 B[9] -> PCOUT[0]           3.851 ns   <- no MREG; mult straight to cascade
  DSP48E1 PCIN setup                 1.400 ns
```

**Logic Levels = 1.** This is not deep logic, it is two hard macros. It
will not improve with placement seeds or router effort, and it will not
degrade from congestion either. The margin is stable and
un-improvable *as structured*.

### Two levers worth ~2 ns each, if 125 MHz is ever needed

1. **Enable the BRAM output register.** 2.454 ns is the latency-1
   figure; with DOREG it is roughly 0.5 ns. `s0_m_i <= buf_i[rp_mid]`
   currently packs as the latency-1 output, so one more register placed
   directly on it lets Vivado absorb it into DOREG. **~1.9 ns.**
2. **Enable MREG on the first DSP.** B->PCOUT at 3.851 ns is the
   no-pipeline-register figure; with MREG the multiply splits into two
   hops of roughly 2 ns. **~1.8 ns.**

Each costs one cycle in `sc_sync`, and that is an existing parameterised
knob: `frame_sync`'s `SC_LATENCY = 3` exists precisely to compensate for
sc_sync's depth, and section 6 records what getting it wrong looks like
(`start_index` 203 or 199 against Python's 200).

Arrival today is 9.402 ns. An 8 ns period (125 MHz) needs about 7.5.
**Either lever alone roughly gets there** — 125 MHz does not need a
rearchitecture.

### What closure does NOT mean

- **It is OUT-OF-CONTEXT.** No I/O buffers, no MMCM (the report shows
  `Total Input Jitter 0.000` — a real clock source adds some), and none
  of the PCIe hard block, DMA, USB or AD9361 interface that milestone 9
  puts on the same die. At **51.9% slices for the RX alone**, integration
  adds both area and congestion, and congestion is exactly what the
  LL=0/LL=1 path families were about earlier. **+0.087 ns will probably
  not survive it.** Closed OOC is a real milestone; it is not closed in
  the product.
- **No TX chain exists.** Whatever it costs comes out of the same 120
  DSPs.
- **Setup and hold are both inside 100 ps.** Nothing is wrong with that,
  but every future change needs a full re-run, not a judgement call.
- **Still unaddressed and unchanged by this run:** the integrated xsim
  against the real `xfft_256` (section 5), the noise sweep, the two
  `rx_top` defects (`frame_sync`'s missing `in_valid` guard and
  `fields_valid` computed but never used), and the Viterbi throughput
  ceiling in section 13.

### This is also the measurement behind section 11

Section 11 argued from source structure that `rx_top.v` is an unnamed
15th block. The first P&R proved it: the glue was the #2 consumer of area
(3,775 LUT) and families starting or ending in it were 1,234 of the 2,121
failing endpoints. Moving the FIFO out took the glue to **295 LUT**,
which is what a wiring file should look like. The rest of section 11's
refactor (the sideband and the two wrappers) is still unbuilt, but the
single largest piece of it is done.

---

## 15. Throughput — how far 1 sample/clock holds

Now that the fabric runs at 100 MHz, "1 sample per clock" means a real
**100 Msps**. This section records how far up the chain that actually
holds, because the answer is not "all of it" and not "hardly any of it".

Work each block must do in one 288-clock slot at 1 sample/clock:

| block | work / slot | clocks | duty | |
|---|---|---:|---:|:--:|
| `sc_sync_rtl` | 288 correlate | 288 | 100% | OK |
| `frame_sync` | 288 replay | 288 | 100% | OK |
| `cfo_estimate` | once per frame | — | — | OK |
| `cfo_correct` | 288 rotate | 288 | 100% | OK |
| `cp_fft` | 256 in / 256 out | 256 | 89% | OK |
| `grid_extract` x2 | 256 bins | 256 | 89% | OK |
| `ls_chanest` | once per frame | — | — | OK |
| `mmse_eq` x2 | 224 | 224 | 78% | OK |
| **`pilot_cpe`** | 17 CORDIC + 216 rot | **235** | **82%** | OK, tightest |
| `demapper` | 216 | 216 | 75% | OK |
| **`viterbi_dec`** | **648 ACS @ QAM64** | **~2,640** | **917%** | **NO** |
| `deinterleaver` | PDU rate | — | — | OK |

**Everything through the demapper sustains 1 sample/clock.** Verified by
initiation interval, not assumed: `mmse_eq` has a 5-deep valid shift
register (`e1_v`..`e5_v`), the CORDICs are unrolled `generate` pipelines
with no feedback, and `cp_fft` is configured
`implementation_options {pipelined_streaming_io}`.

The binding block is `pilot_cpe` at 235 of 288 clocks, so true front-end
headroom is 288/235 = **1.22x, about 122 Msps**.

### Which means the front end is not the limit — the converter is

The AD9361 tops out at **61.44 Msps**. At 60 Msps the fabric runs 1.67
clocks per sample, the slot is 480 clocks, and `pilot_cpe` drops to 49%
duty. **The front end has ~1.6x margin over the fastest the radio can
feed it.**

### The Viterbi is the only block that cannot, and by how much

Required decode rate is a clean formula:

```
  decoded bits/s   = fs x (N_DATA / SLOT_LEN) x bps / 2  =  0.375 . fs . bps
  ACS per clock    = 0.375 x bps / (clocks per sample)      <- no f_clk term
```

`viterbi_dec` pauses ACS during traceback. Per 42-bit burst at
`TB_DISCARD = TB_GROUP = 42`: 42 ACS + 43 `S_TB_DISC` + 43 `S_TB_EMIT` +
43 `S_POP` = ~171 clocks, i.e. **~4.07 clocks per decoded bit, ~24.6
Mbit/s at 100 MHz**.

**FIXED 2026-09-14.** The header comment in `viterbi_dec.v` claimed
"~3 clocks per decoded bit, ~33 Mbit/s" -- it omits `S_POP` and is 34%
optimistic on the decoded figure. The comment now carries the
cycle-by-cycle count, the per-MCS ceilings and both upgrade paths.

| operating point | needed | vs 24.6 Mbit/s |
|---|---:|---|
| 10 Msps QPSK | 7.5 | OK 3.3x |
| 10 Msps QAM16 | 15.0 | OK 1.6x |
| **10 Msps QAM64** | **22.5** | **marginal, 1.09x** |
| 50 Msps QPSK | 37.5 | NO 0.66x |
| 50 Msps QAM16 | 75.0 | NO 0.33x |
| 50 Msps QAM64 | 112.5 | NO 0.22x |

### Can a small change reach 40 Msps? No.

Asked 2026-09-14. Coded demand at 40 Msps is 60 / 120 / 180 Mbit/s
(QPSK / QAM16 / QAM64); in decoded terms 30 / 60 / 90.

Raising `TB_GROUP` **cannot get there**. With ACS paused the ingest duty
is `2G / (3G + D + 3)`, whose asymptote is 2/3 -- so 66.7 Mbit/s coded
(33.3 decoded) however large the group:

| | coded | decoded |
|---|---:|---:|
| today, G=42 | 49.1 | 24.6 |
| G=168 | 61.2 | 30.6 |
| G -> infinity | **66.7** | **33.3** |
| ACS/traceback overlapped | 200.0 | 100.0 |

G=168 buys QPSK at 40 Msps with **2% margin**, which is not a margin.

**The only route to 40 Msps is overlapping ACS and traceback**
(ping-pong the survivor RAM) for a continuous 2 bits/clk. Max sample
rate then becomes:

| decoder | QPSK | QAM16 | QAM64 |
|---|---|---|---|
| today (49.1 coded) | 32.7 | 16.4 | **10.9** |
| G=168 (61.2) | 40.8 | 20.4 | 13.6 |
| overlapped (200) | 133 | 66.7 | **44.4** |

That is a MEDIUM job, not a small one: a second survivor RAM bank,
traceback pointers on the idle bank, and the four-state sequential FSM
split into two concurrent processes -- in a block that is bit-exact
against libcorrect and would need careful revalidation. Radix-4 ACS is
only needed above ~44 Msps at QAM64.

**Nothing else blocks 40 Msps.** At 2.5 clocks/sample the slot is 720
clocks and `pilot_cpe`, the tightest front-end block at 235-249 clocks,
drops to ~35% duty.

Side effect: with an overlapped decoder the **bit FIFO collapses from
165 kbit to ~1 kbit**. Its current size exists only because the demapper
outruns a pausing Viterbi for a whole frame at 1 sample/clock.

**The cheap fix, if the 10 Msps QAM64 margin ever bites:** `TB_GROUP`
42 -> 168 takes 1.09x to 1.36x for one extra BRAM (RAM_AW 7->8) and
latency, with **no BER change** -- quality is set by `TB_DISCARD`, not by
how many bits a pass harvests. Not worth doing pre-emptively.

**The Viterbi has always been the limiter.** 10 clocks/sample is simply
the only budget where it fits — the target was set where the existing
decoder happened to land, and even there QAM64 has 9% margin on an
overstated figure.

### What fixing it would take

1. **Stop pausing ACS during traceback** (ping-pong survivor RAM). This
   is exactly what `viterbi_dec.v:56` defers: *"If a later config needs
   more, THAT is when to overlap them."* Gives 1 decoded bit/clock =
   **100 Mbit/s** — covers 50 Msps QPSK and QAM16, still short of QAM64
   (112.5).
2. **Radix-4 ACS**, two trellis steps per clock. **200 Mbit/s** — covers
   50 Msps QAM64 with 1.8x margin.

Raising the clock does not help at a fixed clocks-per-sample ratio: the
`ACS per clock` formula has no `f_clk` term. 100 -> 125 MHz at 2
clocks/sample buys sample rate (50 -> 62.5 Msps), not decoder headroom.

**Swapping in the Xilinx Viterbi LogiCORE removes the traceback pause,
but check its throughput config before assuming it closes the gap** —
the standard single-channel architecture retires one symbol per clock =
1.0 ACS/clock, which covers QPSK (0.375) and QAM16 (0.75) at 2
clocks/sample but is still short of QAM64's 1.125. The
`Hardware_Evaluation` licence remains the gating issue (section 4).

### Fix #2 would also delete the bit FIFO

Coded bits per slot at 50 Msps / 100 MHz (576 clocks), QAM64 — 1,296
produced per slot:

| Viterbi | consumed/slot | backlog over 128 symbols |
|---|---:|---|
| today, 0.25 ACS/clk | 283 | ~130 kbit — a whole-frame FIFO |
| overlap, 1 ACS/clk | 1,152 | ~18 kbit |
| radix-4, 2 ACS/clk | 2,304 | **none — a few entries** |

**Correction to a claim that appears in `rx_top.v`'s header and in
section 5 item 7: the bit FIFO is NOT a 1-sample/clock artefact.** It is
a Viterbi-throughput artefact. It shrinks at 10 clocks/sample because the
*decoder* finally keeps up, not because the modelling changed. Both
places should be fixed — as written they point the next person at the
wrong fix.

### Three caveats on "the front end does 1 sample/clock"

- **FIXED 2026-10-03 -- see docs/rx_modular_architecture.md H11/H12.**
  The receiver is now rate-invariant (identical FFT output at C = 1, 2.5,
  5, 10, 20) and the CFO estimate is actually applied (it never was).
  Original note kept below for history.
- **`frame_sync` only works at 1 sample/clock.** Its `S_REPLAY` branch
  emits one sample per clock with no `in_valid` guard, while writes
  advance only on `in_valid`. So this is the single rate it handles
  correctly. At 1.67 clocks/sample — the actual 60 Msps case — the reader
  outruns the writer and replays samples that have not arrived. **Fixing
  this is a prerequisite for every real operating point.**
- **The demapper's output has nowhere to go at that rate.** The front end
  can do 1 sample/clock; the receiver cannot, because of the Viterbi.
- **Never measured with continuous input.** Every simulation stops the
  stream and drains — `run_frame.py` allows `RXT_DRAIN 200000` clocks. A
  2048-byte QAM64 frame needs ~66,700 clocks to decode after arriving in
  ~37,700, i.e. the receiver runs at ~0.55x real time and catches up in
  the dead air.

### Forward-looking: the SFO fix collides with 1 sample/clock

Section 7's pilot-slope/SFO correction — the long-packet limiter — needs
**one CORDIC per pilot** instead of the single shared one, because
summing pilots before the CORDIC destroys the per-pilot phases a slope
fit needs. Serialised that is 8 x 17 = 136 clocks of angle instead of 17:

```
  136 + 216 = 352 clocks  >  288
```

**Adding pilot slope tracking breaks 1 sample/clock in `pilot_cpe`.** It
needs 8 parallel vectoring CORDICs, not a reuse of the existing one.
Budget that when the work is scoped, not after.
