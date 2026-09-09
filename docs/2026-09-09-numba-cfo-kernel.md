# 2026-09-09 session: fused Numba CFO estimate/correct kernels

Follow-up to `docs/2026-09-09-numba-sync-kernel.md`'s "If resuming work
on this kernel" section -- `SchmidlCoxCFO.correct()`'s cos/sin pass and
the per-batch-item Python loop in `SchmidlCoxCFO.process()` are now
both accelerated.

## What was built

`spectracuda/cfo/_numba_schmidl_cox.py`: two kernels, wired into
`SchmidlCoxCFO.process()`/`.correct()` with the same transparent
`backend != "cupy"` dispatch as the sync kernel.

1. **`correct()`** -- the actual hot path (runs over the whole ~38K-
   sample received frame on every RX call). **First attempt, measured
   and discarded**: a straightforward fused per-sample loop (one
   `cos`/`sin`/complex-multiply per sample, replacing the numpy path's
   4 separate full-array passes) -- same *shape* of fix as the sync
   kernel. Measured **slower** than the numpy path on this machine
   (1.03ms vs 0.54ms/call at the real frame length): numpy's vectorized
   `cos`/`sin` on a float32 array already hits a SIMD-optimized ufunc
   loop that a naive scalar libm call per element in a numba loop
   doesn't match -- this stage's cost is the transcendental evaluation
   itself, not memory traffic, unlike the sync kernel's problem shape.
   **What shipped instead**: a phase-accumulator (NCO) recurrence.
   `exp(j*k*(i+1)) == exp(j*k*i) * exp(j*k)`, so the whole frame needs
   exactly ONE `cos`/`sin` evaluation (the per-sample rotation step),
   then a running complex multiply (4 real mults + 2 adds) per sample
   -- O(1) transcendental calls instead of O(n). Accumulates in
   complex128 internally as a safety margin against the unit-magnitude
   drift a repeated-multiply recurrence can pick up over ~38K steps.
2. **`process()`'s CFO estimate** -- a Python loop over `n_batch`, each
   iteration summing `conj(first_half) * second_half` over L samples
   (L = fft_size/2, e.g. 128) via `xp.sum`. Not frame-length work, so
   benchmarked rather than assumed worth fusing; it was a small but real
   win (see below) and, per the Amdahl's-law argument in the 2026-08-27
   doc, shrinking any GIL-bound stage here also raises
   `Mac.receive_iq_batch`'s multi-core ceiling, not just single-call
   latency, so it was kept.

**`nogil=True` on both kernels** -- unlike the sync kernel (which does
not yet have this, a known gap, see "If resuming" below), these are
written with the multi-core `receive_iq_batch` benefit in mind from the
start. Safe here because each per-row kernel call only reads/writes its
own batch row's slice of already-allocated numpy arrays.

## Verification

1. **Correctness first, unconditionally** (`tests/test_cfo_numba_acceleration.py`,
   8 tests, 1 skipped without cupy): dispatch actually reaches numba for
   `backend="numpy"` (not just importable, for both `process()` and
   `correct()`); `numba_estimate` cross-checked against the original
   `xp.sum` computation across 30 random (eps, offset, tail) combinations
   at `fft_size=64` and `256`; `numba_correct` cross-checked against the
   original cos/sin-vectorized computation across 20 random batches
   (n_batch 1-3, exercising the per-row dispatch loop) at both fft
   sizes, plus a dedicated check at the real ~37.7K-sample frame length
   for accumulator drift (none observed, `atol=1e-5`); `backend="cupy"`
   never reaches either numba path. Full suite: `pytest tests/ -q` ->
   933 passed, 8 skipped, 4 pre-existing failures (same ones documented
   in the sync-kernel session, confirmed unaffected here too via `git
   stash` A/B on those exact test names).
2. **Only then, benchmark** (same machine, isolated calls, real
   ~37.7K-sample frame, 200 rounds, JIT warmed up first):

   | Call | numpy | numba | speedup |
   |---|---|---|---|
   | `SchmidlCoxCFO.correct()` | 0.543 ms | 0.163 ms | **3.34x** |
   | `SchmidlCoxCFO.process()` (estimate) | 0.0108 ms | 0.0036 ms | ~3x (tiny absolute) |

   End-to-end via `examples/benchmark_x86_stages_v3.py 32000`'s
   "sync detect + CFO" line (back-to-back `git stash` A/B, 3 rounds
   each, same run of the machine to control for load variance):
   **0.795ms -> 0.469ms, ~1.7x** on the combined stage (sync detect +
   CFO estimate + CFO correct together -- sync detect's own numba
   kernel from the previous session is already active in both sides of
   this comparison, so the 1.7x here is attributable to this session's
   CFO change alone).

## Files touched
- `spectracuda/cfo/_numba_schmidl_cox.py` (new)
- `spectracuda/cfo/schmidl_cox.py` (dispatch wiring in both `process()`
  and `correct()`)
- `tests/test_cfo_numba_acceleration.py` (new)

## If resuming work here
- **The sync kernel itself still doesn't have `nogil=True`**
  (`sync/_numba_schmidl_cox.py`) -- the 2026-08-27 doc's "resuming work"
  item 1 asked for it there specifically, and this session added it to
  the new CFO kernels but didn't go back and add it to the sync one.
  Low-risk, same reasoning as here (each row's kernel call only touches
  its own batch row), worth doing before leaning on
  `Mac.receive_iq_batch`'s multi-core path for real gains.
- The NCO-recurrence trick used for `correct()` is a reminder to check
  other "many transcendental evaluations of a linear/near-linear
  argument" spots in the pipeline for the same restructuring -- it's a
  bigger win than the fused-memory-traffic pattern when the transcendental
  call itself, not memory movement, is the bottleneck.
- Per the 2026-08-27 doc's Amdahl reasoning, the next highest-leverage
  move is probably still one of: (a) growing further what's GIL-releasing
  in the RX pipeline so `receive_iq_batch`'s multi-core ceiling keeps
  rising, or (b) the multiprocessing alternative outlined there (not
  attempted in either numba session so far).
