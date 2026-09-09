"""Optional Numba-JIT acceleration for SchmidlCoxSync.process()'s
windowed correlation search.

Same transparent, silent-fallback pattern as fec/_numba_crc.py: numba is
an ordinary optional pip dependency (see pyproject.toml's "fast" extra),
"available" just means "importable," and a missing/failed import falls
back to the existing numpy path with no error and no API change.

**Why this exists**: the numpy path in schmidl_cox.py computes
P(d)/R(d) for every candidate offset via ~10 separate full-array passes
(conj, multiply, abs, two cumsum's, several slice-subtracts, divide,
argmax) over the whole received frame (~37K samples for this project's
standard fft_size=256 config) -- each one a full memory round-trip, even
though the arithmetic per sample is cheap. Same *shape* of problem the
NEON Viterbi kernel fixed (see docs/2026-08-27-neon-viterbi-and-rx-
throughput.md): not bad math, just more memory traffic than the
arithmetic needs.

**The fix**: a single-pass SLIDING-WINDOW correlation, not a cumsum-
then-difference one. P(d)/R1(d)/R2(d) at the next candidate offset are
each an O(1) update from the previous offset's value (drop the sample
leaving the window, add the one entering it) -- no intermediate
prefix-sum arrays, no separate argmax pass (the running max is tracked
inline as the loop goes). This is genuinely a different algorithm from
the numpy path, not just a JIT'd copy of it -- so it gets its own
correctness verification (tests/test_sync_cfo_schmidl_cox.py's existing
suite, run through this path explicitly) rather than assumed-equivalent.

**Numerical note**: the per-candidate window sums (p, r1, r2) are
accumulated in complex128/float64 internally even though rx itself is
typically complex64 -- a deliberate, cheap (O(1) extra work per step,
not O(n)) safety margin against incremental sliding-sum drift over the
~37K-step frame length this runs on in practice, which the numpy path's
cumsum-and-difference approach doesn't accumulate the same way. Output
dtypes match the numpy path's (`int64` start_index, `float64` metric).
"""
from __future__ import annotations

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


_row_kernel: Optional[object] = None  # compiled lazily, only once numba_available() is confirmed True


def _get_row_kernel():
    """Compiles the JIT kernel on first real use, not at import time --
    keeps `import spectracuda` cheap even when numba IS installed but
    sync never runs through this path (e.g. backend="cupy", which
    intentionally skips this module entirely, see schmidl_cox.py)."""
    global _row_kernel
    if _row_kernel is None:
        import numba

        # nogil=True: safe here for the same reason the CFO kernels use it
        # (see docs/2026-09-09-numba-cfo-kernel.md's "If resuming" note) --
        # this function only reads its own batch row's slice of an
        # already-allocated numpy array and returns plain scalars, no
        # Python-object/shared-state touching inside the loop. Without
        # this, Mac.receive_iq_batch()'s worker threads would still
        # serialize on this stage despite it being numba-compiled (a
        # plain @njit still holds the GIL for the call's duration; only
        # nogil=True releases it, same as the C-code Viterbi/RS path
        # does via ctypes).
        @numba.njit(cache=True, nogil=True)
        def _schmidl_cox_row(rx_row: np.ndarray, L: int):
            n_samples = rx_row.shape[0]
            n_candidates = n_samples - 2 * L + 1

            # d=0 window, computed directly (nothing to slide from yet).
            p = complex(0.0, 0.0)
            r1 = 0.0
            r2 = 0.0
            for m in range(L):
                p += np.conj(rx_row[m]) * rx_row[m + L]
                r1 += rx_row[m].real * rx_row[m].real + rx_row[m].imag * rx_row[m].imag
                r2 += rx_row[m + L].real * rx_row[m + L].real + rx_row[m + L].imag * rx_row[m + L].imag

            r = 0.5 * (r1 + r2)
            best_metric = (p.real * p.real + p.imag * p.imag) / (r * r + 1e-12)
            best_d = 0

            # Slide: d -> d+1 drops sample index d (and d+L), adds d+L
            # (and d+2L). e_mid = |rx[d+L]|^2 is shared between r1's
            # incoming term and r2's outgoing term -- computed once, not
            # twice (same dedup logic as the numpy path's e/b1/b2 sharing,
            # see schmidl_cox.py's own module comment for that).
            for d in range(1, n_candidates):
                out_idx = d - 1
                mid_idx = d - 1 + L
                in_idx = d - 1 + 2 * L

                a_out = np.conj(rx_row[out_idx]) * rx_row[mid_idx]
                a_in = np.conj(rx_row[mid_idx]) * rx_row[in_idx]
                p += a_in - a_out

                e_out = rx_row[out_idx].real * rx_row[out_idx].real + rx_row[out_idx].imag * rx_row[out_idx].imag
                e_mid = rx_row[mid_idx].real * rx_row[mid_idx].real + rx_row[mid_idx].imag * rx_row[mid_idx].imag
                e_in = rx_row[in_idx].real * rx_row[in_idx].real + rx_row[in_idx].imag * rx_row[in_idx].imag
                r1 += e_mid - e_out
                r2 += e_in - e_mid

                r = 0.5 * (r1 + r2)
                metric = (p.real * p.real + p.imag * p.imag) / (r * r + 1e-12)
                if metric > best_metric:
                    best_metric = metric
                    best_d = d

            return best_d, best_metric

        _row_kernel = _schmidl_cox_row
    return _row_kernel


def numba_process(rx: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Same contract as SchmidlCoxSync.process()'s core computation:
    rx (n_batch, n_samples) complex -> (start_index (n_batch,) int64,
    metric (n_batch,) float64). Caller (schmidl_cox.py) is responsible
    for validating shape/length and for only calling this when
    numba_available() is True and backend != "cupy"."""
    fn = _get_row_kernel()
    n_batch = rx.shape[0]
    start_index = np.empty(n_batch, dtype=np.int64)
    metric = np.empty(n_batch, dtype=np.float64)
    for b in range(n_batch):
        d, m = fn(rx[b], L)
        start_index[b] = d
        metric[b] = m
    return start_index, metric
