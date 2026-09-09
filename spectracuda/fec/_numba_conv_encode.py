"""Optional Numba-JIT convolutional ENCODER for ConvolutionalCode (rate
1/2, K=7) -- the TX-side counterpart of the decode kernels.

Same transparent, silent-fallback pattern as fec/_numba_crc.py: numba is
an ordinary optional dependency ("fast" extra), "available" means
"importable", and a missing import falls back to the existing native /
numpy paths with no error and no API change.

**Why this exists**: the native encode path (libcorrect's
correct_convolutional_encode via NativeConvolutional*._encode_one) is a
per-bit loop through bit_reader/bit_writer FUNCTION CALLS, wrapped in a
packbits + frombuffer + unpackbits + concatenate round trip per row.
Measured 2026-09-09: 2.13 ms for a 64032-bit PDU on the Pi-5 (~33
ns/bit) and 1.45 ms on x86 -- the largest TX line item once RX had been
brought under the airtime budget -- for what is a 7-tap XOR shift
register. This kernel does it in one pass over the UNPACKED bit arrays
ConvolutionalCode.encode() already works in: per input bit, one 128-entry
table lookup (7-bit register -> the two output bits) and two byte
stores. No packing, no per-bit calls.

**Bit-exactness**: the register convention is exactly viterbi.py's own
pure path (`reg = (state << 1) | b`, bit 0 = current input, `out1 =
parity(reg & G1)` at even output positions, `out2 = parity(reg & G2)` at
odd, K-1 = 6 zero tail bits), which is itself verified bit-exact against
libcorrect's encoder by tests/test_fec_native_acceleration.py; this
kernel is verified against both directly in tests/test_fec_conv_encode_numba.py.
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


def _parity(x: int) -> int:
    return bin(x).count("1") & 1


def build_output_table(g1: int, g2: int) -> np.ndarray:
    """table[reg] = out1 | (out2 << 1) for every 7-bit register value,
    reg = (state << 1) | input_bit -- the same convention as
    ConvolutionalCode._build_transition_tables()."""
    table = np.empty(128, dtype=np.uint8)
    for reg in range(128):
        table[reg] = _parity(reg & g1) | (_parity(reg & g2) << 1)
    return table


_kernel: Optional[object] = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        import numba

        # nogil=True: touches only its own rows of already-allocated numpy
        # arrays, so TX-side batching across threads gets real parallelism.
        @numba.njit(cache=True, nogil=True)
        def _encode(bits, table, tail_bits, out):
            n_batch = bits.shape[0]
            k = bits.shape[1]
            for r in range(n_batch):
                state = 0
                for t in range(k):
                    reg = ((state << 1) | (bits[r, t] & 1)) & 127
                    o = table[reg]
                    out[r, 2 * t] = o & 1
                    out[r, 2 * t + 1] = o >> 1
                    state = reg & 63
                for t in range(k, k + tail_bits):  # zero tail: flush the register back to state 0
                    reg = (state << 1) & 127
                    o = table[reg]
                    out[r, 2 * t] = o & 1
                    out[r, 2 * t + 1] = o >> 1
                    state = reg & 63

        _kernel = _encode
    return _kernel


def numba_encode(bits: np.ndarray, table: np.ndarray, tail_bits: int) -> np.ndarray:
    """bits (n_batch, k) 0/1 -> (n_batch, 2*(k+tail_bits)) uint8, exactly
    ConvolutionalCode.encode()'s contract. Caller is responsible for only
    calling this when numba_available() is True and the input is a host
    numpy array."""
    fn = _get_kernel()
    bits = np.ascontiguousarray(bits, dtype=np.uint8)
    n_batch, k = bits.shape
    out = np.empty((n_batch, 2 * (k + tail_bits)), dtype=np.uint8)
    fn(bits, table, tail_bits, out)
    return out
