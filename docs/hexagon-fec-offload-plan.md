# Hexagon DSP FEC offload -- design + scaffolding (2026-08-30)

**Goal:** move the FEC block (`spectracuda/fec`, sitting between MAC and
OFDM: `MAC <-> FEC <-> OFDM`) to run on the Hexagon DSP on a QCS6490
SoC, target board Radxa Q6A. Starting point: this repo's own
`reference/qc6490/` groundwork and the Pi5-proven NEON Viterbi kernel
(`docs/2026-08-27-neon-viterbi-and-rx-throughput.md`).

**Session constraints (stated up front, not discovered mid-work):** no
Hexagon SDK, no Q6A/Radxa hardware. Nothing in this doc or the files it
describes has been compiled with `hexagon-clang`, run on
`hexagon-sim`, or run on real silicon. Everything here is design +
CPU-side scaffolding that is safe to merge because it is provably inert
until the SDK and hardware are both in hand -- see "What's actually
wired in" below.

---

## 1. Why FEC is the right place to cut, architecturally

`FEC(scheme)` (`fec/fec.py`) already sits behind a clean seam:
`Packetizer` (`framing/packetizer.py`) calls `FEC.encode()`/`.decode()`
knowing nothing about how they're implemented; `FEC` itself dispatches
to `ConvolutionalCode`/`ReedSolomonCode`/`LDPCCode`; and
`ConvolutionalCode` (`fec/viterbi.py`) already dispatches AGAIN, to
whichever of `NativeConvolutionalSSE` / `NativeConvolutionalNEON` /
`NativeConvolutional` / the pure-`xp` fallback is fastest and available
(`fec/_native.py`). **MAC and OFDM never see any of this** -- they call
`Ofdm`/`Packetizer`, which call `FEC`, and everything below that line is
free to change backends without touching either.

That means "move FEC to Hexagon" is, mechanically, "add one more
backend to that existing dispatch chain" -- not a new architecture.
This doc's scaffolding follows that exact existing shape
(`hexagon_available()` / `NativeConvolutionalHexagon`, mirroring
`neon_available()` / `NativeConvolutionalNEON` one-for-one) rather than
inventing a new integration pattern.

## 2. What's actually wired in this session (safe, inert, real)

- `spectracuda/fec/_native_hexagon.py` -- `hexagon_available()` (real
  gate: aarch64 + a loadable FastRPC client lib + a deployed skel file
  at one of a set of placeholder paths -- see its own docstring for why
  each of those paths is a guess, not a verified fact) and
  `NativeConvolutionalHexagon` (constructor raises `NotImplementedError`
  unconditionally right now -- unreachable anyway since
  `hexagon_available()` is False everywhere this has run).
- `spectracuda/fec/viterbi.py` -- dispatch chain now checks
  `hexagon_available()` FIRST, ahead of SSE/NEON. This is provably a
  no-op today (that check is always False), kept first only as a
  placeholder for the *hypothesis* that HVX's width should win once
  real, and flagged in-line as unverified -- **do not** read this
  ordering as an actual recommendation until it's benchmarked.
- DSP-side C skeleton (mirrors `fec/_native_src/libcorrect/src/
  convolutional/neon/` one-for-one): `include/correct-hexagon.h`,
  `include/correct/convolutional/hexagon/convolutional.h`,
  `src/convolutional/hexagon/{convolutional,encode,decode}.c`. None of
  these are added to any build in `fec/_native.py` (that module's
  `_C_FILES`/`_NEON_C_FILES` lists are untouched) -- they cannot
  accidentally get compiled by the existing local-cc pipeline, because
  Hexagon code needs `hexagon-clang`, not `cc`/`gcc`/`clang`, and
  produces a DSP-side skel, not a CPU-loadable `.so` -- fundamentally
  a different build+deploy path (see `_native_hexagon.py`'s own
  docstring, "WHY THIS IS ITS OWN MODULE").
- `spectracuda/fec/_native_src/hexagon/fec_hexagon.idl` -- a FastRPC
  interface sketch, not run through `qaic` (no local IDL/FastRPC sample
  material exists in this repo to check syntax against -- see the
  file's own header comment).

**Nothing above changes behavior on any machine this project currently
runs on.** Full test suite should still pass unmodified; if it doesn't,
that's a real regression to fix, not an expected side effect of this
plan.

## 3. The load-bearing problem found *while writing the sketch*

This is the most important finding from this session, more important
than any single intrinsic name: **at this project's actual code
(K=7, `conv_v27`), one HVX vector is WIDER than the algorithm's own
per-codeword parallelism.**

libcorrect's Viterbi ACS processes one codeword's 64-state trellis in
two 32-state halves per time step (`highbase = 32`). NEON's 8-lane
`uint16x8_t` kernel divides evenly into that 32 (4 iterations per
step) and that's exactly why it works. HVX's 128-byte vector is 64
lanes of `uint16_t` -- **wider than the entire 32-wide half-trellis**.
A mechanical "just widen the NEON kernel's lane count" port (what
`hexagon/decode.c`'s first draft did, before this was caught) either:
  - reads `pair_lookup.keys[]` out of bounds for the extra 32 lanes
    that don't correspond to real base_offsets (a real bug, present in
    an earlier draft of `hexagon/decode.c` in this session, fixed by
    bounding the loop on `base + LANES <= highbase` instead of
    `high <= num_iter` -- see that file's own "OPEN PROBLEM" comment
    for the exact before/after), or
  - once correctly bounded, simply **never enters the HVX path at
    all** for K=7 (`0 + 64 <= 32` is false) and silently falls through
    to plain scalar ACS for every trellis step -- correct, but zero
    speedup. This is what `hexagon/decode.c` actually does right now.

**Two ways to actually fill an HVX vector, neither built yet:**

1. **Batch 2+ independent codewords' 32-wide half-trellises into one
   64-lane vector**, interleaving their ACS math lane-wise. Keeps
   libcorrect's per-codeword C struct shape but needs real surgery: its
   `error_buffer`/`history_buffer`/`pair_lookup` are each sized and
   indexed for exactly one trellis, not two side by side.
2. **Don't port libcorrect's C struct at all -- port `fec/viterbi.py`'s
   own numpy ACS step instead.** That Python path already vectorizes
   the ACS math over `(n_batch, 64 states)` jointly, every trellis time
   step (see that file's own module docstring: "every operation *within*
   a time step is vectorized across the full (n_batch, 64 states) at
   once"). Flatten that to `n_batch * 64` elements per time step and an
   HVX kernel over THAT axis trivially exceeds 64 lanes for any
   `n_batch >= 2` -- filling HVX regardless of K, and matching this
   codebase's own batch-shape-first design (`block.py`'s stated rule)
   instead of fighting libcorrect's single-codeword assumption.

**Recommendation: try (2) first.** It sidesteps the width mismatch
entirely, composes naturally with the batching this project already
does everywhere else (MAC's `receive_iq_batch`, every `Block`'s
batch-shape contract), and only needs (1)'s harder struct surgery if
(2) turns out to have its own problems once real hardware is in hand.
`hexagon/decode.c` as committed is the libcorrect-mirroring attempt
(1)-adjacent path, kept for the record and because it's still the
simpler thing to finish IF (2) stalls -- not because it's the
recommended starting point.

## 4. Why batching the FastRPC call itself is non-negotiable

Separate from the ACS-width problem above: `NativeConvolutional`/`NEON`/
`SSE` (`fec/_native.py`) each loop over the batch **in Python**, one
`ctypes` call per row (`_encode_one`/`_decode_one`). That's fine
in-process -- a `ctypes` call into an already-`dlopen`'d `.so` costs
nanoseconds. A FastRPC call crosses into the DSP's own execution domain
through the kernel and costs real, fixed overhead per call -- on the
order of the entire NEON decode time for one small PDU, plausibly more.
Copying the per-row pattern here would be a **guaranteed** regression
before a single HVX instruction ever ran, the same "hidden host
round-trip" trap this project already hit once with the CUDA LDPC work
(full-pipeline GPU run regressed to 0.56 Msps from host round-trips
despite the kernel itself measuring 2-3.7x faster standalone -- see
this project's own GPU/LDPC exploration notes).

Fix, already reflected in this session's scaffolding: `correct_
convolutional_hexagon_decode_batch()` (`hexagon/decode.c`) and
`fec_hexagon.idl`'s `decode_batch()` both take the WHOLE `(n_batch, k)`
array in ONE call and loop over `n_batch` **inside the DSP-side C**, not
in the Python wrapper. `NativeConvolutionalHexagon.decode()` (once
implemented) must call the generated FastRPC stub exactly once per
`ConvolutionalCode.decode()` invocation, regardless of `n_batch` --
this is the one design rule most worth re-checking if anyone
"simplifies" this code later.

## 5. Open risks -- genuinely unknown, not yet answered

- **Does a stock Radxa Q6A Linux image expose FastRPC to an
  unprivileged userspace process at all, and via which domain (ADSP vs
  CDSP)?** Real Qualcomm boards vary in whether/how this is locked down
  by the OEM's own BSP; Radxa's own Q6A documentation (not present in
  `reference/qc6490/`) needs to be checked, not assumed, before writing
  a single line of real FastRPC glue.
- **Signing.** Some Qualcomm platforms require DSP skels to be signed
  by a vendor-issued key before the ADSP/CDSP loader will run them;
  whether Radxa's image enforces this, and if so how a third party
  signs a custom skel for it, is unknown.
- **HVX generation mismatch risk.** `reference/qc6490/docs/
  hexagon_v68_hvx_programmers_reference.pdf` is the one that matches
  QCS6490's Hexagon rev per its product brief; `hexagon_v73_...pdf` is
  a later chip's HVX rev, useful only for cross-checking, not as the
  primary intrinsic reference -- verify every intrinsic against v68
  specifically, and against the actual silicon rev reported by
  Radxa's own board (`hexagon-sim`'s own `rev_id` output in the local
  `Hexagon_DSP_programming/06-hvx-example/README.txt` transcript shows
  a *different* board reporting `v65a_512` for its own `-mv65` build --
  a reminder that the target flag must match the REAL chip, not be
  assumed from a product brief alone).
- **Every HVX intrinsic name in `hexagon/decode.c`** (`Q6_Vh_vadd_VhVh`,
  `Q6_Q_vcmp_gtu_VhVh`, `Q6_V_vmux_QVV`, `Q6_Vh_vsplat_R`) follows
  Qualcomm's documented naming convention but is individually
  unconfirmed against the v68 PDF -- see that file's own top comment.
- **The FastRPC IDL syntax** in `fec_hexagon.idl` is a best-effort
  sketch, not checked against `qaic` or any real Qualcomm IDL sample
  (none exist locally) -- see that file's own header comment.

## 6. Verification discipline once SDK + hardware exist

Same two-gate rule this project has held every other accelerated
backend to (SSE, NEON, the multi-core `receive_iq_batch` work) --
**correctness first, unconditionally; only then benchmark; only wire
into default dispatch if it's a measured, unconditional win**:

1. Cross-compile the portable `fec/_native_src/libcorrect` C
   (unmodified) with `hexagon-clang` first, as a sanity check that the
   toolchain/include paths work at all, before touching any HVX code.
2. Get `hexagon/decode.c` actually compiling (fix every `VERIFY` marker
   against the v68 HVX reference), and correct on `hexagon-sim` against
   the SAME k=1, 6, 39, 194, 4001, 4002 sweep
   `tests/test_fec_native_acceleration.py` already uses for SSE/NEON,
   bit-exact against the pure-Python decoder as ground truth.
3. Resolve the ACS-width problem (section 3) -- try approach (2)
   (port `viterbi.py`'s own batched numpy ACS step) before approach (1)
   (interleave 2 codewords per vector), per that section's reasoning.
4. Get the real FastRPC path working end to end on actual Q6A hardware
   (section 5's risks resolved), confirm `decode_batch()` is called
   ONCE per `Mac`-level decode regardless of `n_batch` (section 4).
5. ONLY THEN benchmark, at REAL Mac/Ofdm-level batch sizes (not a
   synthetic single-row microbenchmark -- the whole speed claim here
   IS the batching granularity), against both the portable AND NEON
   builds on the same hardware.
6. Wire into `fec/viterbi.py`'s dispatch priority based on what step 5
   actually shows -- not the a priori "HVX is 8x wider" theory this
   doc's own section 3 already showed doesn't trivially hold.

## 7. "Do I have what I need to pick this up?" checklist

- [ ] Hexagon SDK (`hexagon-clang`, `qaic`, HVX headers) -- gated
  behind a Qualcomm Developer Network account + license acceptance,
  needs a person, not an agent (see `reference/qc6490/README.md`).
- [ ] A Radxa Q6A (or other QCS6490 board) reachable for build+deploy,
  with FastRPC access confirmed per section 5's first two risks.
- [ ] `reference/qc6490/docs/hexagon_v68_hvx_programmers_reference.pdf`
  open, to check every `VERIFY`-marked intrinsic in `hexagon/decode.c`.
- [ ] Radxa's own Q6A/QCS6490 BSP documentation (not currently in this
  repo) for the FastRPC domain/signing questions in section 5.

If any of those are missing, the next useful increment is still design
work (resolving section 3's ACS-width problem on paper, e.g.) or
Python-side work that doesn't need the SDK at all -- not attempting to
compile the C sketches here.
