# 2026-09-09 session: fused Numba sync/CFO kernel

Follow-up to `docs/2026-08-27-neon-viterbi-and-rx-throughput.md`'s
section 3 ("NOT DONE: sync/CFO acceleration, diagnosed only") -- that
diagnosis is now implemented.

## What was built

`spectracuda/sync/_numba_schmidl_cox.py`: a fused, single-pass
sliding-window replacement for `SchmidlCoxSync.process()`'s windowed
correlation search. The original numpy path computes P(d)/R(d) for
every candidate offset via ~10 separate full-array passes (conj,
multiply, abs, two `cumsum`s, several slice-subtracts, divide,
`argmax`) over the whole received frame -- each a full memory
round-trip. The numba kernel instead maintains `p`/`r1`/`r2` as running
sums, updated O(1) per candidate offset (drop the sample leaving the
window, add the one entering it), tracking the running best
`(metric, start_index)` inline -- one pass through the data, no
intermediate arrays, no separate `argmax` pass.

This is a genuinely different algorithm from the cumsum path (sliding
window vs prefix-sum differencing), not a JIT'd copy of it, so it got
its own correctness proof rather than assumed equivalence -- see
Verification below. Per-candidate accumulators are kept in
complex128/float64 internally (rx itself stays complex64) as a cheap
safety margin against incremental-sum drift over the ~37K-sample frame
length this runs on in practice.

Wired into `SchmidlCoxSync.process()` as a transparent dispatch: numba
path when `backend != "cupy"` and numba is importable, numpy path
otherwise (same silent-fallback contract as `fec/_numba_crc.py`).
**Deliberately excludes `backend="cupy"`** -- unlike CRC's numba
dispatch (which always coerces to host numpy first regardless of
backend), coercing a cupy array to host here would reintroduce exactly
the hidden device<->host round-trip that regressed the full-pipeline
GPU run once already (see the GPU/LDPC findings elsewhere in this
project's history). `backend="cupy"` keeps using the original
xp-vectorized path, which is the one actually meant to run on-device.

## Verification

1. **Correctness first, unconditionally** (`tests/test_sync_numba_acceleration.py`,
   5 tests): the numba path is actually the active default for
   `backend="numpy"` (not just importable); cross-checked against the
   original cumsum computation across 30 random (offset, tail, SNR)
   combinations at both `fft_size=64` and `fft_size=256` (0 mismatches,
   metric agreement within 1e-4); a dedicated check at this project's
   real ~37.7K-sample frame length (no drift observed, metric diff
   ~8e-7); and a check that `backend="cupy"` never calls into the numba
   path at all (`numba_process` monkeypatched to raise if invoked).
   Full existing suite: `pytest tests/ -q` -> 901 passed, 32 skipped, 4
   pre-existing failures (all from the 2026-09-08 pilot-tracking commit,
   already documented, confirmed unaffected by this change via `git
   stash` A/B).
2. **Only then, benchmark** (same machine, same process, numba forced
   off vs on via monkeypatch, real ~37.7K-sample frame, 60 rounds):

   | Path | `SchmidlCoxSync.process()` alone |
   |---|---|
   | numpy (original cumsum) | 0.7317 ms/call |
   | numba (fused sliding-window) | 0.1352 ms/call |

   **5.41x** on the isolated sync-detect call. End-to-end via
   `examples/benchmark_x86_stages_v3.py 32000`'s "sync detect + CFO"
   line (includes `SchmidlCoxCFO.correct()`'s already-optimized
   cos/sin pass, which this change doesn't touch): ~0.23-0.25ms,
   down from this same machine's earlier-session baseline of 0.5071ms
   -- roughly 2x on the full stage, diluted from the 5.4x isolated
   number by the unaccelerated CFO-correct portion.

## Files touched
- `spectracuda/sync/_numba_schmidl_cox.py` (new)
- `spectracuda/sync/schmidl_cox.py` (dispatch wiring)
- `tests/test_sync_numba_acceleration.py` (new)
- `pyproject.toml` (comment update noting sync now also uses the `fast` extra)

## If resuming work on this kernel
Not tapped out -- `SchmidlCoxCFO.correct()`'s cos/sin pass (already
hand-optimized once, ~1ms/frame) and the per-batch-item Python loop in
`SchmidlCoxCFO.process()` are both unexamined for the same fused-kernel
treatment. Per the 2026-08-27 doc's Amdahl's-law note, this stage sits
on the GIL-bound serialized side of `Mac.receive_iq_batch`'s multi-core
split -- further gains here also raise that path's ceiling, not just
single-thread RX.
