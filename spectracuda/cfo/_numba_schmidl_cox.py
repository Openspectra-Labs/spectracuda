"""Optional Numba-JIT acceleration for SchmidlCoxCFO's estimate/correct
pair -- the CFO half of the sync/CFO fusion work whose sync half is
sync/_numba_schmidl_cox.py (see that module's docstring for the shared
transparent, silent-fallback contract this mirrors: numba is an
ordinary optional pip dependency, "available" just means "importable,"
and a missing/failed import falls back to the existing numpy path with
no error and no API change).

**Two separate targets here, benchmarked separately before deciding
which was worth fusing** (see docs/2026-09-09-numba-cfo-kernel.md):

1. `correct()` -- runs over the WHOLE received frame's samples (~38K for
   this project's standard config) on every RX call: builds a float32
   angle array, then cos(angle)/sin(angle)/complex-assemble/multiply as
   4 separate full-array passes over the frame. This is the one flagged
   in docs/2026-08-27-neon-viterbi-and-rx-throughput.md as "already
   hand-optimized once (cos/sin instead of xp.exp), still unexamined for
   fused-kernel treatment."

   **First attempt (measured, then discarded)**: a single fused loop
   calling cos()/sin() per sample -- same *shape* of fix as the sync
   kernel (collapse several full-array passes into one). Measured
   SLOWER than the numpy path on this machine (1.03ms vs 0.54ms/call,
   ~38K-sample frame) -- numpy's vectorized cos/sin on a float32 array
   apparently already hits a SIMD-optimized ufunc loop that a naive
   scalar per-element libm call in a numba loop doesn't match, so
   removing the memory-traffic overhead didn't matter here: the
   transcendental evaluation itself, not memory traffic, is this
   stage's actual cost, unlike the sync kernel's problem shape.

   **What's actually implemented below**: not a fused *loop* over
   cos/sin at all, but a phase-accumulator (NCO) recurrence -- the
   phase-rotation-by-a-constant-CFO problem doesn't need n_samples
   independent cos/sin evaluations in the first place. `exp(j*k*(i+1))
   == exp(j*k*i) * exp(j*k)`, so the per-sample rotation step
   `exp(j*k)` is computed with exactly ONE cos/sin evaluation, then
   applied as a running complex multiply (4 real mults + 2 adds) per
   sample instead of a transcendental call per sample. This is a
   genuinely different algorithm from both the numpy path and the
   discarded fused-loop attempt (O(1) transcendental evaluations vs
   O(n)), so it gets its own drift-focused correctness proof rather
   than assumed equivalence -- see Verification in the doc. The
   recurrence accumulates in complex128 internally (same safety-margin
   discipline as the sync kernel's p/r1/r2) specifically because
   repeated multiplication by a unit-magnitude step can drift the
   accumulator's magnitude away from 1 over many steps; float32
   wouldn't hold that drift down at ~38K samples the way float64 does.
2. `process()`'s CFO estimate -- a Python loop over n_batch, each
   iteration summing `conj(first_half) * second_half` over L samples
   (L = fft_size/2, e.g. 128) via `xp.sum`. This is NOT frame-length
   work (L samples, not ~38K), so measured first rather than assumed
   worth fusing -- see the module-level benchmark note in the doc. Fused
   here too since it turned out cheap to do and, per the Amdahl's-law
   argument in the 2026-08-27 doc, shrinking any GIL-bound stage in
   `SchmidlCoxCFO.process()`/`.correct()` raises `Mac.receive_iq_batch`'s
   parallel ceiling even where the isolated per-call win is modest.

**`nogil=True` on both kernels below** -- unlike the sync kernel (which
does not yet have this), these are written with the multi-core
`receive_iq_batch` benefit in mind from the start (2026-08-27 doc's
"resuming work" item 1). Safe here because each per-row kernel call only
reads/writes its own batch row's slice of already-allocated numpy
arrays -- no shared mutable state across rows, so releasing the GIL
during a call can't race with itself or other rows called concurrently
from separate threads.
"""
from __future__ import annotations

import math
import threading
from typing import Optional

import numpy as np

_lock = threading.Lock()
_checked = False
_njit = None  # the numba.njit decorator, once imported -- None if unavailable


def numba_available() -> bool:
    global _checked, _njit
    if _checked:
        return _njit is not None
    with _lock:
        if _checked:
            return _njit is not None
        _checked = True
        try:
            import numba

            _njit = numba.njit
        except Exception:
            _njit = None
    return _njit is not None


_correct_kernel: Optional[object] = None  # compiled lazily, only once numba_available() is confirmed True
_estimate_kernel: Optional[object] = None  # ditto


def _get_correct_kernel():
    """Compiles the correct() JIT kernel on first real use, not at import
    time -- keeps `import spectracuda` cheap even when numba IS installed
    but CFO correction never runs through this path (e.g. backend="cupy",
    which intentionally skips this module entirely, see cfo/schmidl_cox.py)."""
    global _correct_kernel
    if _correct_kernel is None:
        import numba

        @numba.njit(cache=True, nogil=True)
        def _correct_row(rx_row: np.ndarray, k: float, out: np.ndarray) -> None:
            # NCO recurrence, not a per-sample cos/sin loop -- see module
            # docstring for why the naive fused-loop version was measured
            # and discarded. `step` is the single per-sample rotation
            # exp(j*k), computed ONCE; `phase` starts at exp(j*k*0) = 1
            # and is advanced by one complex multiply per sample, so
            # phase == exp(j*k*i) at the top of iteration i without ever
            # calling cos/sin again. Both accumulate in complex128 (rx_row
            # itself stays whatever dtype it's given, typically complex64)
            # as a cheap safety margin against the unit-magnitude drift a
            # repeated-multiply recurrence can pick up over ~38K steps.
            n_samples = rx_row.shape[0]
            step = complex(math.cos(k), math.sin(k))
            phase = complex(1.0, 0.0)
            for i in range(n_samples):
                out[i] = rx_row[i] * phase
                phase = phase * step

        _correct_kernel = _correct_row
    return _correct_kernel


def _get_estimate_kernel():
    """Same lazy-compile discipline as _get_correct_kernel() -- see there."""
    global _estimate_kernel
    if _estimate_kernel is None:
        import numba

        @numba.njit(cache=True, nogil=True)
        def _estimate_row(rx_row: np.ndarray, d: int, L: int) -> float:
            p = complex(0.0, 0.0)
            for m in range(L):
                p += np.conj(rx_row[d + m]) * rx_row[d + L + m]
            # angle(p)/pi via math.atan2 rather than np.angle -- numba's
            # nopython mode doesn't cover np.angle, and atan2(im, re) is
            # the exact same computation (angle() IS atan2 under the
            # hood), not an approximation of it.
            return math.atan2(p.imag, p.real) / math.pi

        _estimate_kernel = _estimate_row
    return _estimate_kernel


def numba_correct(rx: np.ndarray, cfo_estimate: np.ndarray, fft_size: int) -> np.ndarray:
    """Same contract as SchmidlCoxCFO.correct()'s core computation: rx
    (n_batch, n_samples) complex, cfo_estimate (n_batch,) float, fft_size
    int -> corrected (n_batch, n_samples) complex, same dtype as rx.
    Caller (cfo/schmidl_cox.py) is responsible for only calling this when
    numba_available() is True and backend != "cupy"."""
    fn = _get_correct_kernel()
    n_batch = rx.shape[0]
    out = np.empty_like(rx)
    for b in range(n_batch):
        k = float(-2.0 * np.pi * cfo_estimate[b] / fft_size)
        fn(rx[b], k, out[b])
    return out


def numba_estimate(rx: np.ndarray, start_index: np.ndarray, L: int) -> np.ndarray:
    """Same contract as SchmidlCoxCFO.process()'s core computation: rx
    (n_batch, n_samples) complex, start_index (n_batch,) int, L = half_len
    -> cfo (n_batch,) float64 estimate (fraction of subcarrier spacing).
    Caller is responsible for the same bounds validation the numpy path
    performs (d + 2*L <= n_samples) before calling this -- the kernel
    itself does not re-check, same division of responsibility as
    sync/_numba_schmidl_cox.py's numba_process."""
    fn = _get_estimate_kernel()
    n_batch = rx.shape[0]
    cfo = np.empty((n_batch,), dtype=np.float64)
    for b in range(n_batch):
        cfo[b] = fn(rx[b], int(start_index[b]), L)
    return cfo
