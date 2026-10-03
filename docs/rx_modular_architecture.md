# RX modular architecture — time / frequency / bit domain

Status: **INTERFACES FROZEN (I1, I2, C1, symbol-type enum), 2026-10-03.**
Implementation has not started. No RTL has been changed to match this
document yet. See §5 H11: the baseline fails at realistic input rates,
which blocks the C=10 regression until it is resolved.

Baseline it refers to: branch `HDL_SPECTRA`, commit `c180a3a`
(`hls/rtl/src/`). That commit is bit-exact against spectracuda at
`ad0a396` and closes timing post-route at WNS **+0.139 ns** on XC7A50T
(13,402 LUT / 41%, 18.5 of 75 BRAM36, 58 of 120 DSP). Its four-wrapper split
(`rx_time_domain` / `rx_freq_domain` / `rx_header` / `rx_bit_decoder`) is a
verbatim move of the old flat `rx_top.v`, **not** the architecture
described here. This document replaces that split.

Read `hls/rundown.md` first for scope (sections 2 and 10 especially): RS,
payload CRC and the MAC stay on the host. Nothing here changes that.

---

## 0. The rule

> **Data flows forward. Configuration flows backward through one explicit
> configuration interface. A stage's internal FSM state never leaks into
> another stage.**

Every design choice below is a consequence of that rule. Two corollaries:

- No stage may rely on "N clocks after X, Y will have happened" about
  another stage. If it needs to know something, it must be carried as
  metadata on the stream or published on the config interface.
- A counter that rebuilds *another stage's* position ("which symbol is
  this?") is a bug waiting to happen. A counter that is part of *this
  stage's own* algorithm (Viterbi push count, FIFO pointers) is fine.

### Why three stages, and why these boundaries

The partition follows the future ASIC so the FPGA RTL can serve as its
architectural reference:

```
  AD9361 → DSP-TD → FFT/IFFT accel → DSP-FD → FEC accel → lower-MAC CPU
           └──── rx_time_domain ───┘ └ rx_freq_domain ┘ └ rx_bit_domain ┘ └ host ┘
```

On the FPGA the FFT stays inside `rx_time_domain` (`cp_fft.v` wraps
`xfft_256`). The FFT-output → frequency-domain boundary is the cleanest one
in the design: interface **I1** below.

---

## 1. Current RX processing sequence (as built, `c180a3a`)

Frame on air: `PREAMBLE (256, no CP) | TRAINING ×1 | HEADER ×1 | PAYLOAD ×n`.
Slot = 288 samples (CP 32 + FFT 256). 216 data, 8 pilot and 32 null bins.

| # | Operation | RTL today | State it uses |
|---|---|---|---|
| 1 | Schmidl-Cox correlation (P, R) | `sc_sync_rtl` | sliding sums |
| 2 | Peak detect, sample replay from frame start | `frame_sync` | sample BRAM, detector FSM |
| 3 | CFO estimate (once per frame, from P) | `cfo_estimate` → `cordic_vec` | — |
| 4 | CFO correction | `cfo_correct` → `cordic_rot` | phase accumulator |
| 5 | Frame phase tracking (PRE/TRAIN/HDR/PAY) | inline FSM in `rx_time_domain` | `phase`, `ph_cnt`, `slot_idx`; reads `n_pay_sym` |
| 6 | CP strip + FFT | `cp_fft` (+ `xfft_256`) | slot counter, **stype tag FIFO** (added on this branch) |
| 7 | Bin classification data/pilot/null | `grid_extract` (u_grid) | `bin` counter |
| 8 | LS channel estimate (training) | `ls_chanest` | — |
| 9 | Re-split H into data/pilot order | `grid_extract` (u_grid_h) + `h_bin` counter | `h_bin` |
| 10 | H hold for the whole frame | inline RAMs `h_data_*`, `h_pil_*` | `hd_wr`, `hp_wr` |
| 11 | Equalize data / pilots | `mmse_eq` ×2 | `d_rd`, `p_rd` (H read address) |
| 12 | Track symbol position after the equalizer | inline | `eq_dcnt`, `eq_pcnt`, `eq_sym` |
| 13 | CPE (payload only) | `pilot_cpe` | 1-symbol queue |
| 14 | Header BPSK hard decision | inline (`~eqd_re[MSB]`) | `hdr_cnt` |
| 15 | Header despread / descramble / parse | `header_decode` | bit counter |
| 16 | Payload length in symbols | inline accumulator in `rx_header` | `sym_acc`, `n_pay_sym`, `counting` |
| 17 | Demapper scheme select | inline in `rx_header` | `dm_scheme` |
| 18 | Demap QPSK/16/64 (hard) | `demapper` | — |
| 19 | Coded-bit FIFO (rate mismatch) + skid buffer | inline in `rx_bit_decoder` | `f_wr`, `f_rd`, `sub`, `rd_nbits`, skid regs |
| 20 | Viterbi (conv_v27, hard, sliding TB) | `viterbi_dec` | trellis/TB FSM; `push_cnt` for `last` |
| 21 | Bit → byte packer | inline | `pack`, `pack_n` |
| 22 | Block deinterleave (unit_bits=8) | `deinterleaver` | own counters |
| — | → host: RS, CRC, MAC | (host) | — |

Note on the order of 20 and 22: Viterbi comes **before** the deinterleaver
(concatenated code, rundown §2). This is not the 802.11 order.

---

## 2 + 3. Ownership: what belongs to which stage

| Operation (# above) | Proposed owner | Change vs today |
|---|---|---|
| 1–4 sync, replay, CFO | **TD** | none |
| 5 frame phase FSM | **TD** | stops reading `n_pay_sym`; reads `cfg_body_syms` (§9) instead, and only knows PREAMBLE/TRAIN/HEADER/BODY |
| 6 CP strip + FFT | **TD** | emits interface I1 |
| — symbol index generation | **TD** | NEW: `sym_idx` counter (the single source of truth) |
| 7 grid classification | **FD** | driven by `bin` from I1, not by its own counter |
| — body symbol classification DATA/DMRS/C2 | **FD** | NEW `fd_classify` (trivial today: everything is DATA) |
| — body buffer while waiting for config | **FD** | NEW `fd_ingress` buffer B1 (§9) |
| 8–10 chan est, H re-split, H hold | **FD** | metadata-driven; becomes the channel-estimate sub-block |
| 11 equalizers | **FD** | metadata passed through the pipeline |
| 13 CPE | **FD** | `sym_end` metadata replaces `eq_sym_done` |
| 18 demapper | **FD** (moves from bit stage) | gains BPSK; emits an LLR group (I2) |
| 14 header hard decision | **FD → removed** | header goes through the demapper as BPSK |
| — egress FIFO under bit-domain backpressure | **FD** | NEW buffer B2 (§8 of the brief); replaces the coded-bit FIFO's role |
| 15 header despread/descramble/parse | **BD** | consumes I2 items with `stype=HEADER` |
| 16, 17 payload length, scheme | **BD → config publisher** | published on interface C1, never wired ad hoc |
| 19 coded-bit unpack | **BD** | reads `n_llr` per item instead of latching `rd_nbits` at `hdr_done` |
| 20–22 Viterbi, packer, deinterleave | **BD** | started by stream metadata, not `hdr_done` |
| RS, payload CRC, MAC | **host** | unchanged (rundown §2) |

**Header-path scope (agreed).** In fabric: header descramble, header FEC
(Viterbi, new format), header CRC-16 (new format), header parse. On the
host: payload RS, payload CRC and above. During modularization the header
stays in the **current** format (uncoded BPSK, 1 symbol, `header_decode.v`).
The new format comes after the refactor (§11).

---

## 4. Signals crossing the proposed boundaries today

From the current wrapper split (`rx_top.v` at `c180a3a`):

| Signal | From → To | Problem |
|---|---|---|
| `fft_re/im/valid/sof/stype` | TD → FD | fine in principle; becomes I1 |
| `frame_start` | TD → FD, header, BD | **broadcast synchronous reset** at TD timing (H1) |
| `hdr_bit`, `hdr_valid_bit`, `hdr_bit_sof` | FD → header | special-purpose header interface; removed (header uses I2) |
| `cpe_re/im/valid` | FD → BD | boundary is before the demapper; moves to after it (I2) |
| `n_pay_sym` | header → TD | TD FSM depends on a value that is still counting (H2) |
| `hdr_done` | header → BD | BD starts Viterbi and deinterleaver on an event from another path (H4) |
| `hdr_bps`, `dm_scheme` | header → BD | latched once per frame on `hdr_done` (H3) |
| `cfg_encoded_bits` | host → header, BD | host-supplied geometry (H8) |
| `cfg_di_units/rows/cols` | host → BD | host-supplied geometry; stays (BD-only, fine) |

---

## 5. Hidden timing dependencies to remove

**H1. `frame_start` is a broadcast reset.** `rst || frame_start` clears
state in FD and BD at the moment TD detects a new frame, whatever those
stages are still doing. With back-to-back frames, BD can still be draining
the previous frame's Viterbi when the next frame is detected, and that
decode gets wiped. *Fix:* frame boundaries travel **in-band**
(`frame_start` on the stream). Each stage closes its own frame when the
marker reaches it.

**H2. `n_pay_sym` race.** `n_pay_sym` is reset to 1 at frame start, then
to 0 at `hdr_done`, and only then counts up over up to 128 clocks. TD's
`slot_idx >= n_pay_sym - 1` (which underflows at 0) is right only because
the first payload `slot_last` comes ≥288 clocks later. If the header never
decodes, the payload phase silently ends after one symbol. *Fix:* TD
reads `cfg_body_syms` only when `cfg_valid` is set and `cfg_fseq` matches.
Until then it keeps capturing, up to `MAX_BODY_SYMS`.

**H3. Demap scheme latched at `hdr_done`.** `dm_scheme` and `rd_nbits`
assume no payload symbol reaches the demapper before the header is parsed.
That holds today only because `pilot_cpe` adds about one symbol of latency.
*Fix:* FD does not process BODY until config is valid (buffer B1).
Every LLR group carries its own `n_llr`.

**H4. BD started by `hdr_done`.** `viterbi_dec.start` and
`deinterleaver.start` come from the header path, not from the payload
stream. *Fix:* BD starts payload decode on the first `stype=DATA` item
of a frame.

**H5. Header special case in equalizer bookkeeping.** `eq_sym_done`
"knows" that the header has no pilot equalization (a bug fixed once
already). *Fix:* the equalizer pipeline carries `sym_end` and `stype`.
`pilot_cpe` acts on `sym_end` of DATA symbols only.

**H6. H read address reset by `g_symdone`.** `d_rd`/`p_rd` rely on
`grid_extract`'s `sym_done` landing exactly between symbols at the
equalizer input. *Fix:* `grid_extract` emits the data/pilot ordinal
(`sc`) with each item. The H RAM is read at that address.

**H7. `h_bin` counts `ls_chanest` outputs** to fake a `sof` for the second
`grid_extract`. *Fix:* `ls_chanest` emits its own `h_bin` / `h_sof`. That
counter is local to the channel-estimate sub-block.

**H8. Host-supplied `cfg_encoded_bits`** must match the received header.
That's a known open item (rundown, `rx_top.v` port comment). It stays
host-supplied during the refactor but enters BD as a config input, not
as a wire into several stages.

**H9. Coded-bit FIFO sized for a whole frame**, because the testbench runs
1 sample/clock (see B2, §7).

**H10. Header start from `hdr_cnt == 0`** counted per frame. *Fix:* the
header is identified by `stype=HEADER` with `sym_start`. The bit count
inside the header parser is that parser's own state.

**H11. The baseline only works at 1 sample/clock (found 2026-10-03).**
With `run_frame.py --cps 10` the baseline (`c180a3a`) detects the frame at
sample 182 instead of 200 and never decodes the header. Two clock-vs-sample
assumptions in `frame_sync.v`:
- `S_REPLAY` emits one replayed sample **every clock**, whatever the input
  rate. At C>1 the reader overtakes the ring writer and replays slots that
  have not been written yet.
- `cand_idx_now = wr − (2·LAG + SC_LATENCY)` subtracts `sc_sync`'s 3-*clock*
  pipeline as 3 *samples*. Those are equal only at C=1.

Nothing downstream of TD has ever been exercised at C>1 either. The C=10
functional regression cannot start until this is resolved (§11).

---

## 6. Counters and state: fate of each

| State | Today | Verdict | Replaced by |
|---|---|---|---|
| `fft_bin`, `fft_sof` | rx_top | **gone** (done on this branch) | `cp_fft.out_sof` |
| `out_sym`, `op_*` | rx_top | **gone** (done on this branch) | `stype` through `grid_extract` |
| `h_bin` | rx_top/FD | **move into chanest sub-block** | `ls_chanest` emits `h_bin`/`h_sof` |
| `hd_wr`, `hp_wr` | rx_top/FD | local to H store | H store writes at `sc` from metadata |
| `d_rd`, `p_rd` | rx_top/FD | **gone** | `sc` ordinal carried with each item |
| `eq_dcnt`, `eq_pcnt`, `eq_sym` | rx_top/FD | **gone** | `sym_start/sym_end/stype` carried through `mmse_eq` |
| `hdr_cnt`, `hdr_sof` | rx_top/FD | **gone** | `stype=HEADER` + `sym_start` on I2 |
| `sym_acc`, `n_pay_sym`, `counting` | rx_header | **moves to BD config publisher** | `cfg_body_syms`, `cfg_valid` |
| `dm_scheme` | rx_header | **gone** | FD demapper picks the scheme from `stype` + `cfg_mod` |
| `rd_nbits` | BD | **gone** | `n_llr` per I2 item |
| `phase`, `ph_cnt`, `slot_idx` | TD | **stays, local** | — (plus new `sym_idx`) |
| `f_wr`, `f_rd`, `sub`, skid regs | BD | **stays, local** | becomes BD's I2 unpack |
| `push_cnt` | BD | **stays, local** | — (Viterbi's own `last`) |
| `pack`, `pack_n` | BD | **stays, local** | — |
| `bin` in `grid_extract` | FD | **gone** | `bin` from I1 indexes the type ROM |

---

## 7. Stage interfaces

### Conventions (all interfaces)

- One clock, synchronous active-high `rst`. `rst` is power-on/global only.
  It is **never** used as a per-frame reset again.
- A **transfer** is a cycle with `valid` (and `ready` where present) high.
  All other signals are defined only during a transfer.
- **No interface carries a timing promise.** A consumer may stall, buffer
  or run at any latency. Only the order of transfers is guaranteed.
- `fseq` is a 2-bit frame sequence number assigned by TD at each frame
  start, modulo 4. Data and config are both tagged with it, so a consumer
  never applies frame N's config to frame N+1's data.
  **Wrap rule:** four values do NOT mean four frames may be outstanding.
  Before TD reuses an `fseq` value, the previous frame carrying that value
  must have fully retired from every stage and buffer (TD, B1, FD pipeline,
  B2, BD, and the C1 bundle). Every stage testbench and the end-to-end
  testbench carry a **simulation assertion** that fires if an `fseq` value
  enters a stage while an older frame with the same value is still
  resident anywhere. Widening `fseq` later is a width change only.
- One symbol-type enum for the whole receiver, `rx_stype.vh` (3 bits):

| Code | Name | Who assigns it | Appears on |
|---|---|---|---|
| 0 | `TRAIN` | TD | I1 |
| 1 | `HEADER` | TD | I1, I2 |
| 2 | `BODY` | TD (not yet classified) | I1 only |
| 3 | `DATA` | FD (refines BODY) | I2 |
| 4 | `DMRS` | FD (refines BODY) | inside FD only; produces no LLRs |
| 5 | `C2` | FD (refines BODY) — future | I2 |
| 6–7 | reserved | — | — |

### I1: TD → FD, FFT bin stream (no backpressure)

| Signal | Width | Definition |
|---|---|---|
| `fb_valid` | 1 | one FFT output bin is presented |
| `fb_re`, `fb_im` | FFT_W (32; 25 significant, unscaled) | bin value, signed |
| `fb_bin` | 8 | natural-order bin index 0..255. All 256 bins of a symbol arrive in increasing order. **Symbol start ≡ `fb_bin==0`, symbol end ≡ `fb_bin==255`**, so there are no separate start/end signals that could disagree. |
| `fb_sym_idx` | 8 | OFDM symbol index within the frame. 0 = first TRAINING symbol (the preamble is never FFT'd and not emitted). +1 per symbol. Constant over a symbol. |
| `fb_stype` | 3 | `TRAIN`, `HEADER` or `BODY`. A pure function of `fb_sym_idx` and the fixed frame format, carried explicitly so FD needs no copy of `N_TRAINING`/`N_HDR`. |
| `fb_fseq` | 2 | frame sequence number |
| `fb_frame_start` | 1 | high on bin 0 of `fb_sym_idx==0` only |

There's no `frame_end` on I1. TD may emit BODY symbols past the real end
of the frame before it learns `cfg_body_syms` (§9). FD discards those,
because it knows the count. **Contract:** FD accepts one transfer per
clock, unconditionally, forever.

### I2: FD → BD, LLR stream (valid/ready)

One transfer = one data subcarrier's group of LLRs.

| Signal | Width | Definition |
|---|---|---|
| `ll_valid`, `ll_ready` | 1, 1 | standard handshake; transfer when both high |
| `ll_llr` | 6×LLR_W | `llr[0..5]`. **`llr[0]` is the first bit in transmission order.** `llr[i]` for `i ≥ ll_n` is driven 0. |
| `ll_n` | 3 | number of valid LLRs: 1 BPSK, 2 QPSK, 4 16-QAM, 6 64-QAM |
| `ll_sc` | 8 | data-subcarrier ordinal within the symbol, 0..215 (for interleaver2, so BD needs no position counter) |
| `ll_sym_start` | 1 | first group of an OFDM symbol (`ll_sc==0`) |
| `ll_sym_end` | 1 | last group of that symbol |
| `ll_sym_idx` | 8 | same index as `fb_sym_idx` for this symbol |
| `ll_stype` | 3 | `HEADER`, `DATA` (later `C2`) |
| `ll_fseq` | 2 | frame sequence number |
| `ll_frame_start` | 1 | first group of the frame (= first header group) |
| `ll_frame_end` | 1 | last group of the last DATA/C2 symbol of the frame. FD can assert it because it knows `cfg_body_syms`. |

**I2 is frozen in this soft-ready form.** The refactor instantiates
`LLR_W=1`. Moving to LLR_W=4/6 later changes only the parameter, never the
protocol (fields, order, handshake, `ll_n` semantics).

**LLR convention.** LLR = log P(b=0)/P(b=1), signed two's complement,
LLR_W bits. **The hard decision is the sign bit (MSB) in every width.**
With LLR_W=1 the LLR is 0 (bit 0) or −1 (bit 1), so hard-decision RTL is
just the 1-bit case of the same interface. Soft decision is a parameter
change, not an interface change. During the refactor LLR_W=1, which is
bit-exact with today's hard demapper.

### O1: BD → host (unchanged in substance)

`out_unit[7:0]`, `out_unit_valid`, `out_last`; header fields; status
(`fifo_overflow`, `hdr_crc_fail` later); `fseq` added.

### C1: configuration interface (BD → FD, TD, host)

Published by BD's config publisher, once per frame, **after** the header
is parsed. A registered bundle, not a stream:

| Field | Width | Meaning | Consumers |
|---|---|---|---|
| `cfg_valid` | 1 | bundle valid for frame `cfg_fseq`; high until the next frame's bundle replaces it | all |
| `cfg_err` | 1 | header rejected (CRC fail, illegal field) for frame `cfg_fseq`; that frame is aborted | all |
| `cfg_fseq` | 2 | frame this bundle belongs to | all |
| `cfg_body_syms` | 8 | **body_symbol_count**, defined below | TD, FD |
| `cfg_mod` | 3 | payload modulation (header code) | FD |
| `cfg_c2_syms` | 8 | C2 symbols at the start of BODY (0 until new header) | FD |
| `cfg_dmrs_period` | 2 | header `dmrs_period` code (0 = off; 0 until new header) | FD |
| `cfg_payload_len_bits`, `cfg_fec0`, `cfg_fec1`, `cfg_crc` | … | header fields | BD, host |

**Frame-length quantity, frozen as `body_symbol_count`:**

> `cfg_body_syms` = the number of OFDM symbols the frame carries **after
> its last header symbol**: every C2 symbol, every main-payload symbol and
> every DMRS symbol. It excludes the preamble, training and header symbols.
> The frame's last symbol index is
> `N_TRAINING + N_HDR + cfg_body_syms − 1`.

Why this one: the fixed part (training, header) is already known to TD.
The body count is the only variable part, and it's the one thing TD needs.
TD never learns how the body is split into C2, data and DMRS.

---

## 8. HEADER, DATA and future DMRS / C2 through one pipeline

```
 symbol class    TD tags   FD does                                      I2 out          BD does
 ────────────    ───────   ──────────────────────────────────────────   ─────────────   ───────────────────
 TRAINING        TRAIN     → channel estimate → H store                 nothing         —
 HEADER          HEADER    → equalize (no CPE) → demap BPSK            HEADER, n=1     header parse → C1
 BODY: DATA      BODY      → B1 until cfg → classify=DATA → eq → CPE    DATA, n=bps     payload Viterbi → deint → host
                              → demap(cfg_mod)
 BODY: DMRS*     BODY      → B1 → classify=DMRS → channel estimate      nothing         —
                              → H store refresh
 BODY: C2*       BODY      → B1 → classify=C2 → eq → CPE → demap QPSK   C2, n=2         C2 Viterbi → deint → host
                                                                                        (*future)
```

The only stage with symbol-class knowledge beyond TRAIN/HEADER/BODY is FD.
BD sees only HEADER, DATA and C2 items, and never DMRS.

### Future DMRS change: what each stage touches

| Change | TD | FD | BD | C1 |
|---|---|---|---|---|
| Add DMRS (first time) | **nothing** | `fd_classify` (DMRS slot rule), chanest sub-block accepts DMRS symbols, H store gets ping-pong banks | header parser extracts `dmrs_period`; `cfg_body_syms` adds `n_dmrs` | `cfg_dmrs_period` (field already reserved) |
| Change DMRS period | nothing | nothing (reads `cfg_dmrs_period`) | nothing | value only |
| Change DMRS placement rule | nothing | `fd_classify` only | `n_dmrs` formula in config publisher | — |
| Change DMRS sequence | nothing | known-symbol ROM (`.mem`, generated) | nothing | — |
| Change refresh/interpolation | nothing | chanest sub-block only | nothing | — |

The DMRS slot rule in `fd_classify` mirrors `spectracuda/framing/dmrs.py`.
The interval counts **data** symbols, so a DMRS follows every `interval`
data symbols, and a trailing DMRS is suppressed. Body ordinal `j`
(0-based, after the C2 symbols) is DMRS iff
`(j+1) mod (interval+1) == 0`. The suppression rule needs nothing extra:
`cfg_body_syms` simply doesn't include a trailing DMRS.

The H store needs **ping-pong banks**: a DATA symbol right after a DMRS
must use the refreshed H, while the previous DATA symbol may still be
equalizing against the old one. The bank flip is triggered by the DMRS
symbol's `sym_end` plus channel-estimate completion. That is FD-internal
metadata, invisible to TD and BD.

---

## 9. Where decoded configuration lives, and the buffers it forces

```
                       ┌──────────────── C1: cfg bundle (cfg_valid, cfg_fseq, …) ────────────────┐
                       │                                                                           │
                       ▼ cfg_body_syms only                 ▼ cfg_mod, cfg_dmrs_period, …           │
 ┌──────────────────────────┐  I1   ┌────────────────────────────────────────┐  I2   ┌───────────┴──────────┐
 │ rx_time_domain           │ ────► │ rx_freq_domain                         │ ────► │ rx_bit_domain        │
 │ sync · CFO · FSM · FFT   │ no    │ B1 ─► classify ─► CE/EQ/CPE ─► demap ─► B2 │ v/r │ hdr parse ─► config  │
 └──────────────────────────┘ ready └────────────────────────────────────────┘       │ payload FEC ─► host  │
                                                                                      └──────────────────────┘
```

Config lives in **one place**: a register bundle owned by BD's config
publisher. TD and FD hold no copies beyond latching the bundle whose
`cfg_fseq` matches the frame they are processing.

### B1, FD ingress buffer (waiting for config)

I1 has no backpressure, so FD can't stall the FFT while the header is
decoded. BODY symbols are written into B1 as they arrive. They are read
out (at up to 1 bin/clk) only once `cfg_valid && cfg_fseq == fb_fseq`.
TRAIN and HEADER symbols bypass B1 (they need no config). If config is
already valid, B1 is a pass-through FIFO.

**Worst-case latency, last header bin out of FFT → `cfg_valid`:**

| Term | Current header (refactor) | New header (CRC-16 + conv 1/2, 2 symbols) |
|---|---|---|
| FD pipeline to the header's last LLR (grid + eq + demap) | ≤ 12 clk | ≤ 12 clk |
| header collect (new format spreads 268 coded bits over 2 symbols, so decoding starts after the last one) | — | 0 (in parallel with arrival) |
| header decode | `header_decode`: ≈ 5 clk after its last bit | dedicated header Viterbi, 134 pairs: 3 groups × 171 + final flush ≈ 110 → **≈ 625 clk** (from `viterbi_dec`'s FSM) |
| descramble + CRC-16 + parse (serial, inline) | — | ≈ 5 clk |
| `cfg_body_syms` derivation (accumulator, ≤ 128 iterations; the DMRS term is a shift) | ≤ 130 clk | ≤ 130 clk |
| **L_cfg** | **≈ 150 clk** | **≈ 775 clk** |

The first BODY bin reaches FD ≈ `32·C + 1` clocks after the last header bin.
C = clocks per input sample, so T_sym = 288·C clocks per symbol.
Required depth:
`B1_syms = ceil((L_cfg − 32·C) / T_sym) + 1` (the +1 is the symbol arriving
while the first drains).

| C (clk/sample) | Operating point | B1, current header | B1, new header |
|---|---|---|---|
| 1 | testbench today | 2 symbols | **4 symbols** |
| 5 | 20 Msps @ 100 MHz | 1 | 2 |
| 10 | 10 Msps @ 100 MHz | 1 | 2 |

**Frozen:** parameter `BODY_BUF_SYMS`, default 4 (never hard-coded),
storing all 256 bins at 2×25 bits: 1024 × 50 bits = **2 BRAM36**. This covers every operating point with the
new header, so B1 never needs resizing for DMRS. All header-path terms
above are **calculated**. The stage testbenches **measure** the actual
latency from the last header item leaving FD to `cfg_valid`, and assert
that `BODY_BUF_SYMS` is still enough at the tested C:
`ceil((L_meas − 32·C) / (288·C)) + 1 ≤ BODY_BUF_SYMS`. If a later header
decoder changes the latency, this assertion catches it.

The bound assumes the header's LLRs are not queued behind a backlog of the
previous frame's payload in B2 (next section). That holds at C ≥ the
rate-contract minimum. At C=1 it holds only with a guard gap between
frames.

**Header FEC requirement (frozen as architecture, not implementation).**
Header FEC decoding is independent of payload FEC decoding and can never
wait for the payload Viterbi. The implementation is NOT frozen: a copy of
`viterbi_dec` may be used first because it is easy to verify, but a smaller
header-specific decoder (fixed 134-pair length, known tail) must be
evaluated before accepting ~3.1k LUT.

### B2, FD egress FIFO (bit-domain backpressure)

```
 I1 ─► B1 ─► classify ─► CE · EQ · CPE ─► demap ─► [B2 FIFO] ─► I2 (valid/ready)
       ▲ no stall anywhere in here: every block before B2 runs at input rate ▲
```

Everything before B2 is free-running at the input rate. Only B2's read
side sees `ll_ready`.

**No finite B2 depth is a substitute for the sustained-rate contract.**
B2 absorbs burstiness and temporary backpressure only. The two
requirements are separate and both must hold:

1. **Average-rate requirement:** BD's sustained consumption ≥ FD's average production at the
   configured C. The payload Viterbi ingests 84 coded bits per 171 clocks
   (0.491 bit/clk). FD produces 216·bps bits per 288·C clocks. So it's
   sustainable iff C ≥ 216·bps / (0.491·288): **QPSK C ≥ 3.1, 16-QAM
   C ≥ 6.1, 64-QAM C ≥ 9.2.** At 10 Msps / 100 MHz (C=10) every MCS fits.
   64-QAM has a 1.09× margin, the tightest number in the receiver
   (rundown §15).
2. **Burst-depth requirement:** B2 absorbs the burstiness within a symbol (216 groups in 256
   clocks, then a gap). At C=10 that is **≤ 1 symbol** (216 entries).

At C=1 the average-rate requirement fails for every MCS. Today's
coded-bit FIFO holds a **whole frame** (128 × 216 entries), which only
postpones overflow to the next frame. It does not fix the mismatch. For
the refactor B2 keeps that depth in the C=1 build (bit-exact, same memory
as today), as a parameter `B2_DEPTH`. `fifo_overflow` stays as a sticky
hardware check that must never fire.

**Cost warning for soft decision.** A whole-frame B2 at LLR_W=4 is
27,648 × 24 bits ≈ 18 BRAM36, double today's total BRAM. Proposal: make
**C=10 the default regression rate** (realistic, B2 ≈ 1 symbol), and keep
C=1 + whole-frame B2 as a stress build only.

---

## 10. Testbench strategy

Two references, used for different questions:

- **Python (spectracuda at `ad0a396`)**: is the algorithm right? Exact at
  bit boundaries; EVM/correlation at float boundaries (FFT, H, equalized
  symbols). This is what `check_chain.py` already does.
- **Frozen RTL (`c180a3a`)**: is the refactor equivalent? Vectors are
  captured from the pre-refactor RTL **at the new boundaries I1 and I2**,
  using a probe testbench on the baseline commit. Refactored stages must
  match these **bit-exactly**, including fixed-point values. This is the
  proof the brief asks for in step 6.

| Testbench | Stimulus | Compared at | Against | Simulator |
|---|---|---|---|---|
| `tb_rx_time_domain` | IQ samples (`trace.py` frame, ± CFO) | I1 (`fb_*`) | frozen-RTL I1 dump (bit-exact); Python `fft_*.out` (EVM/corr) | **Verilator + FFT stub** for fast regression; **xsim + real `xfft_256` is authoritative** (the stub's latency and scaling are not the core's) |
| `tb_rx_freq_domain` | I1 dump + a scripted C1 bundle | I2 (`ll_*`) | frozen-RTL I2 dump (bit-exact); Python `demod_bits/demod_stats` (exact for hard bits) | Verilator |
| `tb_rx_bit_domain` | I2 dump, with random `ll_ready` stalls inserted by the testbench | O1 bytes + C1 bundle | Python `fec:conv_v27.out`, `deinterleave.out`, header fields (exact) | Verilator |
| `tb_rx_end_to_end` (`run_frame.py`) | Python frame | O1 | Python (exact) | Verilator; sweep as today (6 frames + EVM 0.12) |

Each stage testbench also checks the **interface contract**, not just the
data. For example: I1 bins 0..255 in order within each symbol;
`ll_sym_start`/`ll_sym_end` pair up; `ll_n` matches `ll_stype`/`cfg_mod`;
`fseq` is consistent; B2 never overflows under the stall pattern; the
header-to-`cfg_valid` latency is within budget.

Failure localization is then direct:
TD test fails → sync / CFO / CP / FFT;
FD test fails → channel estimate / equalizer / CPE / demapper;
BD test fails → header / FEC / deinterleaver.

### Three rate regimes, kept separate

| Regime | C | Purpose | Rules |
|---|---|---|---|
| Functional (default) | 10 | modularity + bit-exact regression | must pass on every step |
| Stress / burst | 1 | burst handling, B1/B2 corner cases | **single frame, or inter-frame guard ≥ B2/BD drain time.** C=1 is not a valid continuous-rate test: the Viterbi cannot sustain it. |
| `PERFORMANCE_C` | from the PHY target clock / sample rate (e.g. 100 MHz / 10 Msps = 10; 100 MHz / 20 Msps = 5) | throughput: back-to-back frames, worst MCS, sustained | asserts the average-rate requirement (§9) and no overflow; **reports** margin |

Throughput optimization and modular refactoring are separate problems.
Functional regression at C=10 must not be read as a throughput result,
and a `PERFORMANCE_C` failure is a throughput finding, not a
modularity regression.

---

## 11. Order of work

Each step ends with the stage testbench plus the end-to-end regression,
bit-exact. No algorithm, width or rounding changes until step 6 is done.

1. ✅ Mapping (this document §1–6).
2. ✅ Interfaces frozen (§7). Remaining in this step: `rx_stype.vh` (3-bit
   enum) + `rx_if.vh` (interface widths); frozen-RTL dumps at I1/I2 from
   `c180a3a` (C=1, the only rate the baseline works at).
   **Blocked on H11 for C=10.**
3. **FD**: I1 in, I2 out; B1 + classify (all BODY → DATA); metadata
   through `grid_extract`/`ls_chanest`/`mmse_eq`/`pilot_cpe`; demapper moves
   in with BPSK; B2. `tb_rx_freq_domain`.
4. **BD**: I2 in; header parser on `stype=HEADER`; config publisher (C1);
   payload chain started by metadata. `tb_rx_bit_domain`.
5. **TD**: I1 out with `sym_idx`/`fseq`; FSM reads C1; no broadcast
   resets. `tb_rx_time_domain` (Verilator + xsim).
6. Full regression; per-stage equivalence vs frozen RTL; Vivado timing
   (margin is small) and utilization delta.
7. New header format (descramble, conv 1/2, CRC-16, `dmrs_period`) in BD,
   verified on its own. Python reference moves to current `main` for the
   header.
8. DMRS in FD.

Expected latency changes (data values unchanged): B1 and B2 each add a few
pipeline cycles. Nothing in the design depends on absolute latency after
this refactor, which is the point.

---

## 12. Open points to confirm before step 2

1. `fseq` at 2 bits (up to 4 frames in flight across the receiver).
2. Default regression rate C=10, with C=1 kept as a stress build.
3. LLR_W=1 during the refactor (hard, bit-exact). Soft later is a
   parameter change.
4. `BODY_BUF_SYMS = 4` (2 BRAM36), sized now for the new header.
5. A dedicated header Viterbi in BD for the new format (≈ +3.1k LUT if it
   copies `viterbi_dec` as-is; a header-only instance could be smaller.
   Measure in step 7).
