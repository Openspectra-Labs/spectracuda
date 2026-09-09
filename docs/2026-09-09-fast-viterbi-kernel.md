# 2026-09-09 session: "fast" Viterbi decoder (branch-metric-broadcast, 8-bit metrics)

**Why:** after the sync/CFO/modem kernels landed the same day, Viterbi
decode was 60-65% of the whole RX on the Pi-5 (4.55 ms of a 7.2 ms
QAM16/QAM64 frame at ~142 ns/bit) -- the one lever left there. The
existing NEON kernel is a port of libcorrect's inner loop, whose per-
state branch-distance *gather* (`pair_lookup`) has no NEON gather
instruction and stays scalar. This is a different decoder formulation
that has no gather at all.

## Design (`spectracuda/fec/_native_src/libcorrect/src/convolutional/fast/`, `include/correct-fast.h`)

- Rate-1/2 K=7 only (create() returns NULL otherwise); 64 states.
- **Branch metrics precomputed per received 2-bit symbol**: for each of
  the 4 possible symbols, per-predecessor constant vectors
  `popcount(table[...] ^ out)` for the 4 (low/high predecessor x new
  bit 0/1) transition kinds. Per step: load 8 constant vectors, then
  add / min / compare / interleave -- no data-dependent lookup.
- **8-bit path metrics**, 16 states per 128-bit vector (the portable
  build uses 16-bit). Renormalized by subtracting the minimum every 32
  steps; the K=7 survivor-metric spread is bounded (<= 12), so nothing
  wraps, and subtracting a common constant changes no decision.
- **Warmup and tail are literal scalar ports** of libcorrect's (~130
  state updates total), which is what keeps the phase-specific tie rules
  (inner: `low <= high` -> low wins; tail: `low < high` -> high wins ties)
  and the no-history-in-warmup behavior exact without SIMD special
  cases. Only the inner loop is vectorized.
- **Traceback schedule mirrored exactly**: 140-slice ring, first
  traceback after 140 steps emitting the oldest 105 bits (walk back 35
  first), then every 105, flush from state 0; libcorrect's own
  `bit_writer` is reused for output, so its "withheld trailing bits"
  quirk and `_native.py`'s `_DECODE_PAD_PAIRS` workaround carry over
  unchanged.
- One C source, three backends selected at compile time behind ~30
  lines of helpers: SSE2 (x86_64 baseline), NEON (AArch64 baseline),
  generic GCC/clang vector extensions elsewhere. No `-m` flags, no
  runtime CPU check needed. So the file that is bit-exact-tested on the
  x86 dev box is byte-for-byte what the Pi-5 compiles.

## The libcorrect quirk this had to reproduce

First version was bit-exact on every real codeword up to 6% BER at
every length, but disagreed with libcorrect on ~20% of *pure-random*
inputs at every length. A scalar Python port of libcorrect's source
agreed with the new kernel 300/300 and with libcorrect only 232/300 --
so the source as read was not what the binary does. A step-by-step dump
harness around libcorrect's own warmup found it: `error_buffer_reset()`
leaves `read=errors[0], write=errors[1], index=0`, and
`error_buffer_swap()` assigns `read = errors[index]` *before* toggling
`index`, so after warmup step 0 the read buffer is still the zeroed
`errors[0]` and step 1 overwrites step 0's results. **libcorrect's
decoder never uses the first received symbol pair.** Portable, SSE and
NEON builds all share `error_buffer.c`, so this is the reference
behavior everywhere; the fast kernel reproduces it (skips warmup step
0) because bit-exactness is the contract. A future decoder that used
that symbol would be marginally stronger on the first few bits, at the
cost of no longer being bit-exact with libcorrect.

## Verification (`tests/test_fec_fast_viterbi.py`, 16 tests)
- Byte-exact with the portable `correct_convolutional_decode()` at the
  raw C level for every `sets` in 12..79 plus long ones, at 0/5/30%
  corruption -- including the withheld-bits quirk.
- Bit-exact through the Python wrappers at 18 lengths (every T mod 8
  residue, 1 bit .. 32032 bits) x 7 error rates (0 .. 50%).
- **0 mismatches / 5790 pure-random inputs** (sets 12..300 x20 plus
  long) against a fresh portable instance -- the worst case for
  tie-breaking and truncated-traceback paths.
- Dispatch: fast is the default on x86_64; NEON stays the default on
  aarch64 until measured; `SPECTRACUDA_VITERBI_BACKEND` overrides and
  refuses to silently fall back.
- Full suite: see the commit message.

## Measured (x86 dev box, pinned core, 3% BER, median of 60; `examples/benchmark_viterbi_backends.py`)

| PDU | portable | SSE4.1 (previous x86 default) | **fast** |
|---|---|---|---|
| 24040 bits (QPSK) | 5.1-7.4 ms | 1.15 ms (48 ns/bit) | **0.39 ms (16 ns/bit)** |
| 32032 bits (QAM16/64) | 6.8-7.7 ms | 1.50 ms (48 ns/bit) | **0.50 ms (16 ns/bit)** |

End-to-end (`benchmark_x86_stages_v3.py 32000`, interleaved A/B against
`SPECTRACUDA_VITERBI_BACKEND=sse`, three rounds each, same machine state):
the "FEC decode -- Viterbi" line went 1.51-1.89 ms (SSE4.1) -> 0.48-0.78 ms
(fast), and RX 5.1-5.8 -> 2.6-4.4 ms/frame. The box was in a noisy state
for that run (all absolutes ~2x the morning's), but the pairing held in
every round; round 1's 2.62 ms/frame RX is the best figure of the day.

3x over the SSE4.1 kernel on x86.

## Measured on the Pi-5 (aarch64, `examples/benchmark_viterbi_backends.py`, same day)

| PDU | portable | NEON (previous ARM default) | **fast** |
|---|---|---|---|
| 24040 bits | 4.69 ms (195 ns/bit) | 2.95 ms (123 ns/bit) | **0.58 ms (24 ns/bit)** |
| 32032 bits | 6.24 ms (195 ns/bit) | 3.92 ms (122 ns/bit) | **0.76 ms (24 ns/bit)** |

**5.1x over the NEON kernel** (bit-exact-checked by the script before
timing), so `fast` is now the default on aarch64 as well -- the
dispatch's `platform.machine()` gate and
`test_dispatch_prefers_fast_on_measured_architectures` were flipped
together with this number. Predicted effect on the Pi-5's RX (Viterbi
was 4.55 ms of a ~7.2 ms QAM16/QAM64 frame): ~3.9 ms/frame.
**Confirmed** with `benchmark_x86_stages_v3.py 32000` on the Pi-5 right
after the flip:

| Pi-5, 32000-bit SDU | RX ms/frame, start of 2026-09-09 | after modem kernel | **after fast Viterbi** | vs 4 Msps budget | real RX Mbps |
|---|---|---|---|---|---|
| QAM16 | 10.1-11.7 | 7.5 | **3.88** (Viterbi line 4.55 -> 0.93) | OK (6.33 ms) | 2.7-3.2 -> **8.25** |
| QAM64 | 9.3 | 6.9-7.2 | **3.43** (Viterbi line 4.55 -> 0.91) | OK (4.31 ms) | 3.4 -> **9.33** |

The Pi-5 now clears the 4 Msps real-time budget single-core at every
modulation. Remaining RX ranking there (QAM16): "everything else"
1.13 ms, Viterbi 0.93, Reed-Solomon 0.69, sync+CFO 0.45, OFDM decode
0.33, chanest+eq 0.31. `SPECTRACUDA_VITERBI_BACKEND=neon` still forces
the old kernel for any later comparison.

## If resuming here
- The kernel is ~16 ns/bit on x86 with plenty of headroom: the inner
  step is ~40 vector ops and the decision slices are written
  byte-per-state (64 B/step); packing decisions to bits and unrolling
  two steps per iteration are the obvious next moves if Viterbi ever
  matters again.
- Encoders (TX) are still the plain portable shift-register loop, ~33
  ns/bit on the Pi-5 for something that should be ~1 ns/bit -- TX is
  inside budget so it wasn't touched, but it is the largest TX line
  item there.
