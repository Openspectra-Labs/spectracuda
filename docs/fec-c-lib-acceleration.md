# FEC C Lib Acceleration

How and why spectracuda's FEC codecs (`ConvolutionalCode`/`conv_v27`,
`ReedSolomonCode`/`rs_m8`, `LDPCCode`) reach out to native code instead of
staying pure Python/NumPy, what each native backend actually is, and how the
same problem is solved three different ways across x86, ARM, a Qualcomm DSP,
and (for LDPC) an entirely separate external decoder.

Companion to [`docs/architecture.md`](architecture.md) (the whole-system
layering) and [`docs/hexagon-fec-offload-plan.md`](hexagon-fec-offload-plan.md)
(the Hexagon design in full) — this document is the FEC-native-code slice of
both, in one place.

## Why this exists

spectracuda's own Python/NumPy `ConvolutionalCode.decode()` and
`ReedSolomonCode.decode()` are interpreter-loop-bound: Viterbi's
add-compare-select and RS's Berlekamp-Massey are both "many small sequential
steps, each individually cheap" — the shape that a Python interpreter loop
handles badly and a native compiled loop handles well (batching across
`(n_batch, ...)` with `xp` array ops, spectracuda's usual GPU-parallelism
trick, doesn't help here — these are inherently sequential per-codeword
algorithms, not batch-parallel ones). Measured on x86, one full-length
message/codeword:

| Codec | Pure Python/NumPy | Portable C | Speedup |
|---|---|---|---|
| Viterbi decode (conv_v27) | 38.6 ms | 1.78 ms | ~22x |
| Reed-Solomon decode (rs_m8, 16-symbol-error worst case) | 4.0 ms | 0.03 ms | ~130x |

That portable-C win alone (`libcorrect`, below) is the baseline every other
backend in this document is measured against.

LDPC is architecturally different enough that it gets its own section further
down: its bottleneck isn't a Python loop, it's the belief-propagation
algorithm itself, and the fix wasn't writing SIMD but bridging to an existing
mature decoder (AFF3CT).

## The three codecs, three different native strategies

| Codec | Native strategy | Where |
|---|---|---|
| Viterbi (`conv_v27`) | Vendored C (`libcorrect`), compiled on demand, with x86 SSE4.1 / ARM NEON / (future) Hexagon variants | `spectracuda/fec/_native.py`, `_native_hexagon.py` |
| Reed-Solomon (`rs_m8`) | Same vendored C, portable build only — no SIMD variant exists upstream or was written for it | `spectracuda/fec/_native.py` |
| LDPC (all 12 variants) | Bridge to AFF3CT (external, mature C++ decoder), opt-in, subprocess-based | `spectracuda/fec/_native_aff3ct.py` |

Not one architecture reused three times — three genuinely different problems
(a hot inner loop worth hand-vectorizing; the same loop with nothing to
vectorize; an entire algorithm class worth not reimplementing at all), so
three genuinely different solutions. The common thread is *why* each choice
was made, which is what the rest of this document is about.

## `libcorrect`: vendored C for Viterbi + Reed-Solomon

[`libcorrect`](https://github.com/quiet/libcorrect) (BSD-licensed) is
vendored as portable C99 source under
`spectracuda/fec/_native_src/libcorrect/` — not a pip dependency, not
installed system-wide, just source files shipped in the repo. `_native.py`
compiles it **on demand**, once per (source hash, platform), into a `.so`
cached under `~/.cache/spectracuda` (override with
`SPECTRACUDA_CACHE_DIR`) — not recompiled on every process start, and never
touched again unless the vendored source itself changes (the cache key is a
SHA-256 of the actual `.c`/`.h` files, so an edit to the source invalidates
it automatically, no manual cache-bust needed).

Activation is **fully automatic and transparent** — not a constructor
argument. `ConvolutionalCode`/`ReedSolomonCode`'s public API is unchanged
either way; a `backend="numpy"` instance silently uses the fastest available
native path, a `backend="cupy"` instance never does (forcing a
device→host→device round trip to reach CPU-only native code would defeat
the point of choosing cupy). If no C compiler exists, or compilation fails
for any reason, this fails **silently and permanently** back to pure
Python/NumPy — deliberately not a loud failure, since nothing was
explicitly requested here to be loud about (contrast with
`decoder="aff3ct"` below, which *is* an explicit request and *does* fail
loud).

### Dispatch chain

`ConvolutionalCode.__init__` picks the first of these that's actually
available, checked in this order:

```
hexagon_available()  →  sse_available()  →  neon_available()  →  native_available()  →  (pure Python)
```

`ReedSolomonCode` only has the last rung (`native_available()`) — no SIMD RS
build exists, on any architecture (see "Why Reed-Solomon has no SIMD
variant" below).

Every one of these is a **runtime** check, not a compile-time or
platform-name guess — the reasons differ by rung and matter enough to be
worth walking through individually, because getting this wrong doesn't fail
softly on this particular class of bug (see the SSE section).

### x86_64: SSE4.1

`libcorrect` ships an SSE4.1-accelerated Viterbi decode upstream (vendored
unmodified) — the same base `convolutional.c`/`decode.c` machinery, with
just the add-compare-select inner loop replaced by SIMD. Measured a further
**~2.5x** over the portable build on top of that build's own ~22x. No
Reed-Solomon SSE build exists upstream.

`sse_available()` gates on **two independent conditions**, both required:

1. `platform.machine()` is `x86_64`/`AMD64` — SSE4.1 doesn't exist anywhere
   else.
2. A **runtime** feature check (`/proc/cpuinfo`, Linux-only — declines on
   any other OS rather than guessing): compiling with `-msse4.1` only
   proves the *compiler* can target that instruction set, not that the
   *machine that will eventually run this process* has it. This is
   load-bearing, not a nicety — a compiled `.so` can outlive the machine it
   was built on (a shared `SPECTRACUDA_CACHE_DIR`, a container image
   copied to different hardware), and executing an SSE4.1 instruction on a
   CPU that lacks one is `SIGILL`: an immediate, uncatchable process crash,
   not a Python exception this module could quietly absorb the way it
   absorbs a missing compiler. So the check has to happen *before* the
   `.so` is ever loaded, not be discovered by trying it and catching a
   failure.

The compiled artifact's own cache key includes `platform.machine()` (not
just `sys.platform`) for the same reason — an x86_64-compiled SSE `.so`
handed to an ARM process by a shared cache directory shouldn't just fail to
load, and this makes sure the filename itself never collides across
architectures in the first place.

### ARM: NEON, and why the first attempt made things worse

NEON (mandatory in the AArch64 base ISA — unlike SSE4.1, no separate
runtime CPUID-style check is needed here, only the `platform.machine()`
gate) is spectracuda's own port, not vendored — no NEON build exists
upstream. This is the most instructive part of the whole document, because
the first attempt at it was a real, measured regression, and the reason it
regressed is exactly the kind of thing that's invisible until you actually
run it on the target hardware:

| Build | Pi 5, ~24000-bit PDU | vs. portable |
|---|---|---|
| Portable (no SIMD) | 3.72–3.80 ms | baseline |
| NEON, 1st attempt (4-lane, `uint16x4_t`) | 7.92 ms | **~2.1x slower** |
| NEON, 2nd attempt (8-lane, `uint16x8_t`) | 2.38 ms | ~1.6x faster |

The first attempt was *correct* (bit-exact, passed every correctness test)
and still shipped as a genuine regression — because it gathered table
lookups through small stack scalar arrays before and after each narrow
4-lane vector op, and that memory round-tripping cost more than the ALU
parallelism bought back at that width. `viterbi.py`'s dispatch deliberately
did **not** prefer it over the portable build once that was measured, even
though it was "done" and tested — a working SIMD port that's slower than
scalar code doesn't get used just because it exists.

The second attempt fixed the actual problem, not just the symptom: widened
to 8 lanes, and replaced the stack-array round-trips with direct
`vld1q_u16` loads off the already-contiguous error buffer, a `vld2q_u16`
de-interleave for the gathered distance pairs, and `vst2q_u16`/`vst2_u8`
interleaved stores straight back into the working buffers. *That* version
measured faster than portable, on the same hardware, same PDU size, same
day — and that's the one actually wired into the dispatch chain.

The lesson this leaves behind, stated in `_native.py`'s own dispatch-order
comment: **measured, not assumed** — every rung in that chain earned its
position by a real benchmark on real hardware, not by "this should be
faster" reasoning, and the same standard is explicitly held over whatever
gets promoted into that chain next (Hexagon, below, is deliberately *not*
promoted ahead of SSE/NEON despite a plausible-sounding "1024-bit HVX
vectors should beat 128-bit NEON, scaled up" argument — that's a
hypothesis, not yet a measurement).

### Why Reed-Solomon has no SIMD variant

Simply: nobody has vendored or written one. `libcorrect` upstream never
shipped an SSE Reed-Solomon build, and no NEON port for it exists in this
codebase either — RS's Berlekamp-Massey decode gets the ~130x portable-C
win and stops there. It's a real (if smaller, in absolute terms — 0.03 ms is
already fast) gap relative to Viterbi's further 2-2.5x, not a considered
decision that RS doesn't need it.

## Hexagon: a fundamentally different kind of "native"

`spectracuda/fec/_native_hexagon.py` is the Qualcomm-silicon counterpart to
SSE/NEON, targeting QCS6490-class SoCs (e.g. a Radxa Q6A) that carry a
Hexagon DSP with HVX (1024-bit vectors — 8x NEON's 128-bit) alongside their
Kryo/Cortex CPU cores. It's in its own module, not another branch of
`_native.py`, because it's categorically not the same kind of thing:

| | SSE / NEON (`_native.py`) | Hexagon (`_native_hexagon.py`) |
|---|---|---|
| Toolchain | System `cc`/`gcc`/`clang` | Qualcomm's own `hexagon-clang` cross-toolchain |
| Where it runs | Same process, same CPU, same address space | A **separate processor** (the DSP), not the CPU Python runs on |
| How it's reached | `ctypes.CDLL` + a direct function call | FastRPC — a kernel-mediated RPC transport |
| Per-call cost | Nanoseconds (already-loaded code, local call) | A fixed, real overhead per crossing — easily more than an entire NEON decode for one small PDU |

That last row is the actual design constraint, not a footnote: batching
isn't optional here the way it's a nice-to-have elsewhere. `_native.py`'s
existing pattern (one `ctypes` call per batch row, looped in Python) would
make a Hexagon backend a *certain* regression before a single instruction of
HVX code had a chance to help — so the Hexagon design passes the **whole**
`(n_batch, k)` array across in one FastRPC call and loops over `n_batch`
inside the DSP-side C, not in the Python wrapper (the same batch-shape
contract every other `Block` in this codebase already uses, for the same
underlying reason: dispatch overhead per call is the thing to amortize,
whichever accelerator is on the other end of it).

**Status**: design scaffolding, not a working accelerator.
`hexagon_available()` unconditionally returns `False` on every machine
this has actually run on — no Hexagon SDK, no Q6A hardware in hand yet.
It's checked *first* in the dispatch chain (ahead of SSE/NEON) on the a
priori theory that HVX's wider vectors should win the same kind of speedup
NEON proved out, scaled up — but per the "measured, not assumed" rule
above, that ordering is provably inert until real hardware says otherwise,
and re-benchmarking against both SSE and NEON (not just assuming the
ordering is right) is the explicit condition for that to change. Full
design/rationale/open risks: [`docs/hexagon-fec-offload-plan.md`](hexagon-fec-offload-plan.md).

## LDPC: bridging to AFF3CT instead of hand-vectorizing

LDPC's decode is normalized min-sum belief propagation — a fixed iteration
count over a fixed sparsity pattern, genuinely batch-parallel (unlike
Viterbi/RS), so spectracuda's own `LDPCCode.decode()` is already vectorized
across `(n_batch, ...)` with `xp` gather ops, GPU-capable via cupy with no
custom kernel. The problem isn't a Python loop to escape — the algorithm
itself, run this way, is just slow in absolute terms on CPU (hundreds of ms
for a 12-codeword decode), and hand-writing a SIMD or Hexagon BP kernel to
fix that is real, hard, low-leverage work: BP over a sparse irregular
parity-check graph is gather/scatter-heavy and branchy, nothing like
Viterbi's fixed trellis or RS's clean Galois-field arithmetic that made SSE/
NEON worth writing by hand in the first place.

[AFF3CT](https://github.com/aff3ct/aff3ct) (MIT-licensed) already is that
work: a mature, portable-SIMD (via its own MIPP wrapper — SSE/AVX/AVX-512/
NEON dispatched at compile time), multi-algorithm (MS/NMS/AMS), multi-
schedule (flooding/horizontal-layered/vertical-layered) LDPC decoder with
syndrome-based early termination — the one real advantage spectracuda's own
decode doesn't have (it always runs the full `max_iterations`; AFF3CT stops
as soon as a codeword's parity checks are satisfied). So instead of a fourth
from-scratch SIMD port, spectracuda bridges to it.

### Architecture: a persistent process, not a library call

AFF3CT is a large external project (~850MB checked out + built) with **no
Python API and no LLR-in/bits-out entry point** by design — its normal use
is as a standalone C++ simulation binary with its own internal source/
channel/encoder/monitor loop. Getting a real decode path out of it meant
writing spectracuda's own driver, not just calling into AFF3CT as-is:

```
spectracuda/fec/_native_src/aff3ct_bridge/
  bridge_ldpc.cpp    -- standalone C++ process, linked directly against
                         AFF3CT's built static lib, calling its
                         Decoder_LDPC::build()/decode_siho() API straight
                         (bypassing AFF3CT's Source/Channel/Modem/BFER loop
                         entirely)
  qc_export.py       -- exports spectracuda's own base matrices into
                         AFF3CT's native .qc format
  build.sh / setup_aff3ct.sh -- build the bridge / clone+build AFF3CT+bridge
  verify_bridge.py   -- bit-exact correctness check against spectracuda's
                         own decode()
```

`bridge_ldpc` is a **persistent process**, not spawned per frame: LLRs in on
stdin (`n_cw` float32s), a status + decoded bits out on stdout (1 int32 +
`k` int32s), looping until stdin closes. `spectracuda/fec/_native_aff3ct.py`
owns one such process per LDPC variant (module-level cache, `atexit`
cleanup) and talks to it over that pipe — this specifically avoids paying
AFF3CT's own process-startup cost on every single codeword, the same
"amortize the per-call crossing cost" principle the Hexagon design above is
built around, just with a Unix pipe instead of FastRPC as the transport.

### Two real bugs, found by actually round-tripping a real codeword

Getting `bridge_ldpc` bit-exact against spectracuda's own decode surfaced
two genuine mistakes, both worth recording because neither was caught by
AFF3CT's own self-consistency testing (AFF3CT decoding *its own*
internally-generated random payloads only proves AFF3CT is internally
consistent — it never proves spectracuda's own matrix export or convention
matches what AFF3CT expects, since it never touches a real spectracuda
codeword at all):

1. **A wrong shift-sign negation** in the `.qc` matrix export — the
   original derivation misread AFF3CT's real `QC::_read()` parser formula
   and applied a sign flip that wasn't needed. Fixed by tracing the actual
   parser source line-by-line instead of re-deriving from memory, then
   cross-checking the exported matrix against spectracuda's own `H` in
   Python directly (bit-for-bit identical, independent of any C++ code at
   all).
2. **AFF3CT's own `transform_H_to_G_identity()`** (used internally to
   derive which codeword positions are the "systematic" message bits)
   picks an arbitrary valid choice — it has no way to know spectracuda's
   own fixed convention (message = always the first `k` columns). The
   decoder was reconstructing the *correct* codeword the whole time; only
   the message-bit extraction was reading from the wrong `k` positions,
   which looked like ~50% random-looking corruption even on a clean,
   noiseless codeword. Fixed by hardcoding spectracuda's actual convention
   directly instead of trusting AFF3CT's own auto-derivation.

Both were only found by testing a real encode→corrupt→decode round trip
through the bridge and diffing against spectracuda's own decoder output —
not by inspecting the export code, not by trusting AFF3CT's own sanity
checks. The general lesson: a bridge between two independently-correct
systems can still be wrong at the seam, and the only way to know is to
actually round-trip real data across it.

### Opt-in, fail loud, never silent fallback

Unlike `libcorrect`'s transparent SSE/NEON promotion, this is **not**
automatic:

```python
LDPCCode(variant, decoder="native")   # default, always available
LDPCCode(variant, decoder="aff3ct")   # opt-in, requires bridge_ldpc built
```

`decoder="aff3ct"` without a built `bridge_ldpc` raises `Aff3ctUnavailable`
with the exact build command, never silently falls back to `"native"` — the
opposite contract from `libcorrect`'s "missing compiler degrades quietly."
The difference is what's being asked for: `libcorrect`'s native path is a
transparent optimization of an already-chosen `backend="numpy"`, nothing
new was explicitly requested; `decoder="aff3ct"` **is** an explicit request,
so silently doing something else instead would hide a real, actionable gap
(a decode running 50-100x slower than the caller thinks it asked for) behind
no signal at all.

`encode()` always uses spectracuda's own systematic encoder regardless of
`decoder=` — the bridge is decode-only.

### Getting it running

Nothing clones or builds AFF3CT automatically — it's deliberately not a pip
dependency or part of a fresh `git clone` (reference/aff3ct/ is gitignored,
same "reference, not shipped" status as the rest of `reference/`). One
command does the whole thing, idempotently:

```bash
spectracuda/fec/_native_src/aff3ct_bridge/setup_aff3ct.sh
```

Clones AFF3CT (`--recursive` — it has 6 submodules) if missing, configures +
builds it (Release, static lib — several minutes, hundreds of translation
units), then builds `bridge_ldpc`. Safe to re-run; skips whatever's already
done. If AFF3CT is already built, `.../aff3ct_bridge/build.sh` alone is
enough. Only the bridge's own source (`bridge_ldpc.cpp`, `qc_export.py`,
`build.sh`, `setup_aff3ct.sh`, `verify_bridge.py`) is tracked in git — the
compiled binary (~26MB, statically links the whole of AFF3CT) and object
file are gitignored, rebuilt locally.

## Portability at a glance

| | x86_64 (dev machine) | AArch64 (Pi 5) | Hexagon DSP (QCS6490, future) |
|---|---|---|---|
| Viterbi | Portable C, or SSE4.1 (~2.5x further) if the CPU supports it | Portable C, or NEON (~1.6x further, 2nd-attempt kernel) | Design only — `hexagon_available()` always `False` today |
| Reed-Solomon | Portable C (~130x vs. pure Python) | Portable C (same) | Not planned — no SIMD RS build exists to port from |
| LDPC | `decoder="native"` (numpy, always) or `decoder="aff3ct"` (opt-in, AFF3CT's own SIMD is portable across x86/ARM via MIPP) | Same as x86_64 | N/A |

The `libcorrect` portable build is the one universal floor everywhere a C
compiler exists — every SIMD variant on top of it is a measured, optional
speedup, never a requirement, and every one of them fails back to that floor
(or, for the Hexagon/AFF3CT cases, fails loud) rather than crashing when
unavailable.

## Where to look next

- `spectracuda/fec/_native.py` — libcorrect dispatch, SSE/NEON detail,
  compile-cache mechanics (extensively commented, this document's primary
  source).
- `spectracuda/fec/_native_hexagon.py` — Hexagon design scaffolding.
- `spectracuda/fec/_native_aff3ct.py`, `_native_src/aff3ct_bridge/` — the
  AFF3CT bridge.
- `tests/test_fec_native_acceleration.py` — libcorrect/SSE/NEON correctness
  + interop tests.
- `tests/test_fec_ldpc_aff3ct.py` — AFF3CT bridge correctness tests
  (skipped unless `bridge_ldpc` is built).
- [`docs/hexagon-fec-offload-plan.md`](hexagon-fec-offload-plan.md) — full
  Hexagon plan.
- [`docs/2026-08-27-neon-viterbi-and-rx-throughput.md`](2026-08-27-neon-viterbi-and-rx-throughput.md)
  — the session notes behind the NEON numbers above.
