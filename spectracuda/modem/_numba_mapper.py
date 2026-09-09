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
