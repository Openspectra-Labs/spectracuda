# Protected C2 region — MAVLink survives when video does not

**Status: planned 2026-09-20, not started.**

Follows the DMRS work (`docs/2026-09-20-dmrs-periodic-channel-refresh-plan.md`,
steps 1-6 shipped on `feat/dmrs-periodic-channel-refresh`).

## Goal

One sentence: **split the payload into a small, fixed-profile critical
region and the existing adaptive-MCS region, so that losing the video
does not lose the control link.**

The property worth having:

```
main payload fails CRC   =/=>   C2 fails
```

Today a single CRC covers the whole payload, so one bad video frame
takes MAVLink down with it. That is the wrong failure mode for a UAV
datalink.

## Frame layout after this

```
[ Preamble ] [ Training ] [ BPSK Header ]
                                |
        +-----------------------+-----------------------+
        |                                               |
  [ C2 region ]                                  [ Main payload ]
  QPSK, fixed FEC                                header-selected MCS/FEC
  MAVLink / command                              video / IP / bulk
        |                                               |
        +-----------------------+-----------------------+
                                |
              pilots every symbol (CPE), DMRS every ~1ms (H[k])
```

## Decided

### The C2 profile is FIXED, not signalled

```
QPSK  +  crc32  +  fec0=rs_m8  +  fec1=conv_v27  +  block interleaver (unit_bits=8)
```

No `c2_mcs` or `c2_fec` header fields. The receiver does not need to be
told, cannot be told wrong, and a corrupted header cannot cause a
mis-decode of C2 by selecting the wrong scheme. This is the same
reasoning that makes the DMRS interval safe to carry in 2 bits: fewer
representable states, fewer ways to be wrong.

It is also the exact profile already measured on real hardware
(`rs_m8 + conv_v27`, ~3-4 dB coding gain, see
`docs/fec-c-lib-acceleration.md`), so it is not a new unknown.

Cost, at `n_data=216` (432 QPSK coded bits per symbol):

| C2 bytes | encoded bits | OFDM symbols | % of 128-slot budget | airtime @10 MSps |
|---|---|---|---|---|
| 16 | 844 | 2 | 1.6% | 58 us |
| 64 | 1612 | 4 | 3.1% | 115 us |
| **72 (typical MAVLink)** | **1740** | **5** | **3.9%** | **144 us** |
| 128 | 2636 | 7 | 5.5% | 202 us |
| 256 | 5196 | 13 | 10.2% | 374 us |
| **320 (MAX)** | **6220** | **15** | **11.7%** | **432 us** |

`C2_MAX_BYTES = 320`, enforced in `generate_frame()` and on decode.
`c2_len_bytes` is a uint16 on the wire so it can express more; the cap
is a policy limit, checked separately, so an over-large decoded value
is rejected as corruption rather than acted on.

A typical 72-byte C2 burst costs 5 symbols (3.9%); the worst case costs
15 (11.7%).

### Header layout: bytes 6-7 become C2_LEN

`user_data` is currently 8 bytes (6-13) and is a pure pass-through --
nothing in the library, examples or `abhi/` reads it. Take two:

```
bytes 6-7 : c2_len_bytes, uint16   (0 = no C2 region in this frame)
bytes 8-13: user_data, 6 bytes     (was 8)
```

Header stays 112 bits. `PROTOCOL_VERSION` goes 1 -> 2, and
`tests/test_framing_header.py` + `tests/test_ofdm_class.py` need their
8-byte `user_data` expectations updated.

Deliberately NOT added yet: `c2_flags`, `c2_seq`. The MAC already owns
sequencing and dedup; adding wire fields nothing consumes is how header
space gets wasted. Bytes 8-13 remain available if that changes.

**The layout is fixed, not conditional.** Bytes 6-7 are always C2_LEN
even when it is 0. A header whose field positions depend on another
field's value is harder to parse and much harder to debug.

### Who decides what is C2 — the PHY never does

The PHY has no concept of MAVLink, C2 or priority, and should not
acquire one. It is **told**:

```python
ofdm.generate_frame(payload_bits, c2_bits=..., user_data=...)
```

Two separate arguments. The caller fills both. `c2_bits=None` produces
today's frame exactly.

That keeps the PHY honest: it does not sniff payloads, does not parse
MAVLink, and has no policy embedded in it. The policy lives one layer
up, where it belongs.

### What the MAC does (NEXT project, not this one)

The MAC currently has no priority concept at all -- `send_iq()`
segments an SDU and calls `generate_frame()` once per PDU
(`mac/mac.py:211`). The design worth building on top of this PHY work:

1. `Mac.queue_c2(bits)` enqueues control data on a small, separate
   queue -- it is NOT segmented or merged into the normal SDU stream.
2. **Every frame the MAC emits, for any reason** -- video, bulk data,
   an ACK, a link report -- drains up to `C2_MAX_BYTES` of pending C2
   into that frame's C2 region.
3. If C2 is pending and there is nothing else to send, emit a C2-only
   frame. Verified legal today: a zero-length main payload produces a
   valid 1-symbol frame that round-trips with a passing CRC, so this
   needs no special case in the PHY.

The point of (2) is that **C2 rides along on any transmission
opportunity**. Its latency is bounded by the next frame out rather than
by its own scheduling slot, and it costs no extra airtime beyond the
region itself. That is a much better property than giving C2 its own
periodic slot, and it is only possible because C2 is a region of an
ordinary frame rather than a frame type of its own.

None of that is in this plan. This plan gets the wire format and the
PHY right so the MAC work has something correct to sit on.

### DMRS counts across BOTH regions

The channel does not care which region a symbol belongs to, so the DMRS
interval runs over all payload-carrying symbols:

```
dmrs_slot_map(n_c2_symbols + n_main_symbols, interval)
```

`framing/dmrs.py` needs **no change at all**. The data slots it returns
are simply partitioned: the first `n_c2_symbols` are C2, the rest are
main payload. Per-segment `H[k]` expansion also works unchanged, because
it is already expressed over "all data symbols".

### The 128-slot budget now covers three things

```
n_c2_symbols + n_main_symbols + n_dmrs_symbols  <=  MAX_PAYLOAD_SYMBOLS
```

Same rule as before, one more term. C2 comes out of the budget like DMRS
does -- it does not extend airtime.

---

## The one hard problem

Everything else is plumbing. The hard part is **the region boundary in
the receiver**.

Today `_decode_payload_from_header` works on one homogeneous run of
symbols: one `n_payload_symbols`, one `bits_per_ofdm_symbol`, one modem,
one packetizer. After this there are two regions with **different bits
per symbol** (QPSK C2 vs possibly 64QAM main), interleaved with DMRS
slots that belong to neither.

The boundary is *derived*, never signalled: `c2_len_bytes` plus the
fixed profile gives the C2 encoded length, hence `n_c2_symbols`. If the
receiver computes that even slightly differently from the transmitter,
both regions decode as garbage, and the CRCs will say so — but only
after wasting the frame.

It gets built and verified on its own, with the main region left at
QPSK first (so a boundary error shows up as a clean failure rather than
being masked by a modulation mismatch), then re-verified with the main
region at 64QAM.

---

## Steps

### 1. C2 profile + header field

`C2_PROFILE` as a module-level constant (one place, not scattered
literals). `HeaderCodec` gains `c2_len_bytes` encode/decode, `user_data`
shrinks to 6, `PROTOCOL_VERSION` -> 2.

Verify: round-trip every `c2_len_bytes` 0..65535 boundary case, and that
`user_data` still round-trips at its new length. Update the two existing
tests that pin 8 bytes.

### 2. C2 sizing helper

`n_c2_symbols(c2_len_bytes)` -- the derived boundary, as a pure function
next to `framing/dmrs.py`'s arithmetic, testable with no OFDM object.
This is the number both sides must agree on, so it lives in exactly one
place and both call it.

### 3. TX: emit the C2 region

`generate_frame(payload_bits, c2_bits=None, user_data=None)`. Encode C2
through its own `Packetizer` and QPSK `Modem`, concatenate its symbols
ahead of the main payload symbols, then interleave DMRS over the
combined sequence exactly as now.

Verify TX-side only: frame length, and that the C2 symbols demodulate
back to the C2 bits when pulled out by hand.

### 4. RX: decode the C2 region — **the hard step**

Split the equalized data slots at `n_c2_symbols`, run the two regions
through their own modem/packetizer, return both.

Verify, in order:
- `c2_len_bytes=0` is **bit-exact** with today's output. Regression gate.
- C2 round-trips with the main region also at QPSK.
- C2 round-trips with the main region at 64QAM (different bits/symbol
  across the boundary -- the case a boundary bug actually breaks).
- C2 round-trips with DMRS enabled (slots interleaved across both
  regions).
- **The headline test: corrupt the main region only, and C2 still
  delivers with a valid CRC.** If this does not pass, the feature has
  no reason to exist.

### 5. Budget + MAC capacity

Extend the total-slot guard to three terms. `Ofdm.max_data_symbols`
becomes a function of `c2_len_bytes` too, and `mac/capacity.py` must
size against it -- same failure mode as the DMRS case, and the same fix.

### 6. Result shape

`rx_process()` gains `c2_bits` / `c2_crc_valid`, always present, `None`
when the frame carries no C2 region -- matching the existing
fully-enumerated-dict contract. `rx_streaming` likewise.

NOT in this step: delivering C2 upward before the main region finishes
decoding. That is a genuinely useful latency win for UAV control, but it
is an API and MAC change (a second callback or a two-phase return), not
a PHY change, and it should not be bundled with getting the wire format
right.

### 7. Measure

PER of C2 vs main across SNR, on the corrected harness from
`docs/2026-09-20-dmrs-static-channel-cost.md` (trailing capture samples,
length-independent paired noise, bounds failures counted separately).
The number that matters: **C2 delivery at SNRs where main payload
delivery has already collapsed.**

---

## Explicitly out of scope

- **Header CRC/FEC.** Reviewed and queued as the next item after this.
  The header is the weakest link and this feature makes it more
  load-bearing. C2 does degrade safely without it -- a corrupted
  `c2_len_bytes` mis-slices the region and the C2 CRC32 fails closed
  rather than delivering garbage upward -- but that is containment, not
  protection.
- Early C2 delivery (see step 6).
- MAC priority queueing for C2, and the piggyback policy sketched
  above. That is the layer above and the natural next project.
- Per-frame adaptive C2 profile. The fixed profile is the point.

## Open question

Should C2 be placed before or after the main payload in symbol order?
Before, as planned here, means the C2 symbols arrive earliest and are
available soonest -- which matters if step 6's early delivery ever
happens. It also means C2 sits closest to the training symbol, so it
uses the freshest `H[k]` in the frame. Both argue for "before", which is
why the plan says before, but it is worth stating that it was a choice.
