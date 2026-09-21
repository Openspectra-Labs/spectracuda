"""Optional Numba-JIT acceleration for Modem.demodulate() (hard decision)
fused with the EVM power sums that rx_process() computes right after it.

Same transparent, silent-fallback pattern as fec/_numba_crc.py and
sync/_numba_schmidl_cox.py: numba is an ordinary optional pip dependency
("fast" extra), "available" just means "importable", and a missing or
failed import falls back to the existing numpy path with no error and no
API change.

**Why this exists**: cProfile over 40 real QPSK frames (2026-09-09, see
docs/2026-09-09-rx-streaming-partial-preamble-fix.md section 3) put
~1 ms/frame of the RX "everything else" bucket in three back-to-back
sweeps over the same 27648 payload symbols: `Modem.demodulate` (the
generic M-QAM path -- descale, round/clip per axis, gray, bit-unpack,
concat: ~20 array passes and 22 `astype` copies per frame for what is a
sign test at QPSK), then `Modem.modulate(demod_bits)` purely to get the
nearest constellation points back for EVM, then `compute_evm`. The
nearest point is already known inside the hard decision, so one fused
pass yields the bits AND the per-row EVM sums with no round trip.

**Bit-exactness**: the bits are the FEC's input, so this kernel
reproduces the numpy path's float32 arithmetic step for step rather
than "the same math in float64":
- numpy's `symbols / norm` (complex64 / python float) is NOT a division
  by norm: numpy's complex-division loop computes the reciprocal
  `scl = float32(1)/float32(norm)` once and multiplies each component
  by it. The kernel does exactly that.
- `(level + (m-1)) / 2.0` is done in float32, then `np.round`
  (round-half-to-even == `np.rint`), then clip, then int.
- The ideal (nearest-point) symbol used for EVM is
  `float32(2b - (m-1)) * float32(norm)` per axis, matching
  `modulate()`'s `_pam_level` * norm in float32.
Only the EVM power SUMS accumulate in float64 (the numpy path's
`compute_evm` uses float32 pairwise means, so EVM agrees to ~1e-6
relative, not bit-for-bit -- it's a diagnostic readout, verified to
rtol=1e-5 in tests/test_modem_numba_acceleration.py).

Layout contract (matches mapper.py exactly): per symbol, `half` I-axis
bits MSB-first followed by `half` Q-axis bits MSB-first; BPSK is one
bit per symbol from the real axis only.
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

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


_kernel: Optional[object] = None  # compiled lazily, only once numba_available() is confirmed True


def _get_kernel():
    """Compiles on first real use, not at import time (keeps `import
    spectracuda` cheap even with numba installed but a cupy backend that
    never takes this path)."""
    global _kernel
    if _kernel is None:
        import numba

        # nogil=True: only reads its own input rows and writes its own
        # output rows of already-allocated numpy arrays, no Python objects
        # inside -- so Mac.receive_iq_batch()'s worker threads get real
        # parallelism through this stage, same as the sync/CFO kernels.
        @numba.njit(cache=True, nogil=True)
        def _hard_decision(symbols, half, m, scl, norm32, is_bpsk, out_bits, out_err, out_ref):
            n_rows = symbols.shape[0]
            k = symbols.shape[1]
            mm1 = np.float32(m - 1)
            two = np.float32(2.0)
            for r in range(n_rows):
                err = 0.0
                ref = 0.0
                if is_bpsk:
                    for i in range(k):
                        s = symbols[r, i]
                        x = s.real * scl
                        b = 1 if x >= 0 else 0
                        out_bits[r, i] = b
                        ideal_r = np.float32(2 * b - 1) * norm32
                        dr = s.real - ideal_r
                        di = s.imag
                        err += dr * dr + di * di
                        ref += ideal_r * ideal_r
                else:
                    bps = 2 * half
                    for i in range(k):
                        s = symbols[r, i]
                        # per-axis: descale (reciprocal multiply, see module
                        # docstring), level -> natural-binary index, round
                        # half-to-even, clip
                        vr = (s.real * scl + mm1) / two
                        vi = (s.imag * scl + mm1) / two
                        bi = int(np.rint(vr))
                        bq = int(np.rint(vi))
                        if bi < 0:
                            bi = 0
                        elif bi > m - 1:
                            bi = m - 1
                        if bq < 0:
                            bq = 0
                        elif bq > m - 1:
                            bq = m - 1
                        gi = bi ^ (bi >> 1)
                        gq = bq ^ (bq >> 1)
                        base = i * bps
                        for j in range(half):
                            sh = half - 1 - j
                            out_bits[r, base + j] = (gi >> sh) & 1
                            out_bits[r, base + half + j] = (gq >> sh) & 1
                        ideal_r = np.float32(2 * bi - (m - 1)) * norm32
                        ideal_i = np.float32(2 * bq - (m - 1)) * norm32
                        dr = s.real - ideal_r
                        di = s.imag - ideal_i
                        err += dr * dr + di * di
                        ref += ideal_r * ideal_r + ideal_i * ideal_i
                out_err[r] = err
                out_ref[r] = ref

        _kernel = _hard_decision
    return _kernel


def numba_hard_decision(
    symbols: np.ndarray, scheme: str, bits_per_symbol: int, norm: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """symbols (n_rows, k) complex64 -> (bits (n_rows, k*bits_per_symbol)
    uint8, err_power_sum (n_rows,) float64, ref_power_sum (n_rows,)
    float64). Caller (mapper.py) is responsible for only calling this
    when numba_available() is True, backend != "cupy", and symbols is
    complex64 (the float32 arithmetic above is what makes the bits
    bit-exact with the numpy path; a complex128 input takes the numpy
    path instead)."""
    fn = _get_kernel()
    sym = np.ascontiguousarray(symbols, dtype=np.complex64)
    n_rows, k = sym.shape
    is_bpsk = scheme == "bpsk"
    half = 1 if is_bpsk else bits_per_symbol // 2
    m = 2 ** half
    scl = np.float32(1.0) / np.float32(norm)
    bits = np.empty((n_rows, k * bits_per_symbol), dtype=np.uint8)
    err = np.empty(n_rows, dtype=np.float64)
    ref = np.empty(n_rows, dtype=np.float64)
    fn(sym, half, m, scl, np.float32(norm), is_bpsk, bits, err, ref)
    return bits, err, ref


_soft_kernel: Optional[object] = None


def _get_soft_kernel():
    """Soft-decision (max-log LLR) counterpart to _get_kernel(), compiled
    on first real use for the same reason.

    Per-AXIS, not per-point. For a square Gray-coded QAM the constellation
    is separable -- |y-s|^2 = (yr-sr)^2 + (yi-si)^2 with the two axes
    taking their levels independently -- so the min over points carrying
    label 0 on an I-axis bit is (min over that axis's levels) + (min over
    ALL Q levels), and the Q term is identical for label 1 and cancels in
    the difference. The LLR for an I bit therefore depends only on the I
    axis:

        llr_j = min_{a in S0(j)} dr[a] - min_{a in S1(j)} dr[a]

    That turns 16 distances plus 64 comparisons per 16QAM symbol into 8
    distances plus 16 comparisons, and it is the same decomposition the
    hard kernel above already relies on. Mathematically identical to the
    generic min-over-all-points form in mapper.py, which stays as the
    reference the tests check against.
    """
    global _soft_kernel
    if _soft_kernel is None:
        import numba

        # nogil=True for the same reason as the hard kernel: reads its own
        # rows, writes its own rows, no Python objects inside.
        @numba.njit(cache=True, nogil=True)
        def _soft_llr(symbols, lev, idx0, idx1, weight, use_w,
                      clip, levels, use_q, out):
            n_rows = symbols.shape[0]
            k = symbols.shape[1]
            n_lev = lev.shape[0]
            half = idx0.shape[0]
            m = 2 * half
            raw = np.empty((n_rows, k, m), dtype=np.float32)
            dr = np.empty(n_lev, dtype=np.float32)
            di = np.empty(n_lev, dtype=np.float32)
            min_sum = 0.0

            for r in range(n_rows):
                for i in range(k):
                    s = symbols[r, i]
                    sr = s.real
                    si = s.imag
                    br = np.float32(1e30)
                    bi = np.float32(1e30)
                    for a in range(n_lev):
                        u = sr - lev[a]
                        v = si - lev[a]
                        uu = u * u
                        vv = v * v
                        dr[a] = uu
                        di[a] = vv
                        if uu < br:
                            br = uu
                        if vv < bi:
                            bi = vv
                    min_sum += br + bi
                    for j in range(half):
                        r0 = np.float32(1e30)
                        r1 = np.float32(1e30)
                        q0 = np.float32(1e30)
                        q1 = np.float32(1e30)
                        for t in range(idx0.shape[1]):
                            a0 = idx0[j, t]
                            a1 = idx1[j, t]
                            if dr[a0] < r0:
                                r0 = dr[a0]
                            if dr[a1] < r1:
                                r1 = dr[a1]
                            if di[a0] < q0:
                                q0 = di[a0]
                            if di[a1] < q1:
                                q1 = di[a1]
                        raw[r, i, j] = r0 - r1
                        raw[r, i, half + j] = q0 - q1

            # sigma^2 from the mean distance to the nearest point -- the
            # same self-calibration the numpy path uses, so the two agree.
            sigma2 = min_sum / np.float64(n_rows * k)
            if sigma2 <= 0.0:
                sigma2 = 1e-12
            inv = np.float32(1.0 / (2.0 * sigma2 * clip))

            # Pass 2 -- scale, weight, clip, optionally quantize, write the
            # byte. O(n*m), no distance work, so it is nearly free.
            for r in range(n_rows):
                for i in range(k):
                    w = weight[r, i] if use_w else np.float32(1.0)
                    for b in range(m):
                        v = raw[r, i, b] * inv * w
                        if v > np.float32(1.0):
                            v = np.float32(1.0)
                        elif v < np.float32(-1.0):
                            v = np.float32(-1.0)
                        if use_q:
                            v = np.float32(round(v * levels)) / levels
                        qq = int(np.rint(np.float32(128.0) + np.float32(127.0) * v))
                        if qq < 0:
                            qq = 0
                        elif qq > 255:
                            qq = 255
                        out[r, i * m + b] = qq

        _soft_kernel = _soft_llr
    return _soft_kernel


def numba_soft_llr(symbols, pts, labels, weight, llr_clip, llr_bits):
    """symbols (n_rows, k) complex64 -> (n_rows, k*bits_per_symbol) uint8
    in libcorrect's convention. Separable square-QAM only (the caller
    gates on that); bpsk and any non-separable scheme take mapper.py's
    generic numpy path."""
    fn = _get_soft_kernel()
    sym = np.ascontiguousarray(symbols, dtype=np.complex64)
    n_rows, k = sym.shape
    m = labels.shape[1]
    half = m // 2
    n_lev = 2 ** half
    # Axis levels and their Gray labels, read straight off the point table
    # so they cannot drift from modulate()'s mapping.
    lev = np.unique(np.round(pts.real.astype(np.float64), 6)).astype(np.float32)
    if lev.shape[0] != n_lev:
        raise ValueError("non-separable constellation -- use the numpy path")
    idx0 = np.empty((half, n_lev // 2), dtype=np.int32)
    idx1 = np.empty((half, n_lev // 2), dtype=np.int32)
    for j in range(half):
        b0, b1 = [], []
        for a in range(n_lev):
            g = a ^ (a >> 1)
            (b1 if (g >> (half - 1 - j)) & 1 else b0).append(a)
        idx0[j] = np.asarray(b0, dtype=np.int32)
        idx1[j] = np.asarray(b1, dtype=np.int32)
    use_w = weight is not None
    w = (np.ascontiguousarray(weight, dtype=np.float32) if use_w
         else np.zeros((1, 1), dtype=np.float32))
    use_q = llr_bits is not None
    levels = np.float32(2 ** (int(llr_bits) - 1) - 1) if use_q else np.float32(1.0)
    out = np.empty((n_rows, k * m), dtype=np.uint8)
    fn(sym, lev, idx0, idx1, w, use_w, np.float32(llr_clip), levels, use_q, out)
    return out
