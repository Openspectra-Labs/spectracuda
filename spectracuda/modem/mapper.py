"""Modem: Modem(scheme) -- one class, scheme-name string, mirrors
liquid-dsp's modem_create(scheme) directly.

Implements standard Gray-coded square PSK/QAM constellations (the same
family liquid-dsp itself uses): BPSK, QPSK, 16-QAM, 64-QAM, 256-QAM.
Per-axis (I/Q) Gray-coded PAM with the well-known IEEE 802.11-style
average-power normalization constants (1/sqrt(2), 1/sqrt(10), 1/sqrt(42),
1/sqrt(170)) -- cross-checked against those known standard constants
since reference/liquid-dsp isn't buildable in this environment yet (no
autoconf/automake/libtool/cmake installed). A bit-exact cross-check
against liquid-dsp's own modem module (docs/liquid-dsp-api-inventory.md)
is still owed once build tooling is available -- in particular, the
convention used here (first half of each symbol's bits -> I axis, second
half -> Q axis) is this project's own choice and has NOT been verified to
match liquid-dsp's bit ordering.

Batch-shape contract: modulate() takes (n_batch, n_bits) bits ->
(n_batch, n_bits // bits_per_symbol) complex64. demodulate() (hard
decision, nearest constellation point) is the exact inverse.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ..block import Block
from ._numba_mapper import numba_available, numba_hard_decision

_BITS_PER_SYMBOL = {
    "bpsk": 1,
    "qpsk": 2,
    "qam16": 4,
    "qam64": 6,
    "qam256": 8,
}


class Modem(Block):
    """Gray-coded PSK/QAM modulator-demodulator.

    Parameters
    ----------
    scheme:
        One of "bpsk", "qpsk", "qam16", "qam64", "qam256".
    """

    def __init__(self, scheme: str, *, backend=None) -> None:
        super().__init__(backend=backend)
        if scheme not in _BITS_PER_SYMBOL:
            raise ValueError(
                f"Unknown modem scheme {scheme!r}; expected one of "
                f"{sorted(_BITS_PER_SYMBOL)}"
            )
        self.scheme = scheme
        self.bits_per_symbol = _BITS_PER_SYMBOL[scheme]
        self.batch_shape_doc = (
            f"(n_batch, n_bits) bits in -> "
            f"(n_batch, n_bits // {self.bits_per_symbol}) complex64 out, "
            f"and the exact inverse for demodulate()"
        )
        # Precomputed once here, not on every modulate()/demodulate() call:
        # `half` (bits per I/Q axis) and therefore these weight/shift arrays
        # are fixed for this instance's whole lifetime. Re-deriving them
        # per call (per-symbol-batch, i.e. every OFDM symbol) was a real,
        # measured cost -- np.arange()+shift-table construction running on
        # the hot TX/RX path for no reason, since nothing in it ever
        # changes after __init__.
        half = 1 if scheme == "bpsk" else self.bits_per_symbol // 2
        xp = self.xp
        self._weights_half = 1 << xp.arange(half - 1, -1, -1, dtype="int64")
        self._shifts_half = xp.arange(half - 1, -1, -1, dtype="int64")
        self._norm = self._compute_norm_factor()

    # -- internal helpers ---------------------------------------------------

    def _gray_to_binary(self, g, nbits: int):
        """Parallel-prefix-XOR gray-to-binary, elementwise over an xp array."""
        b = g
        shift = 1
        while shift < nbits:
            b = b ^ (b >> shift)
            shift *= 2
        return b

    def _binary_to_gray(self, b):
        return b ^ (b >> 1)

    def _bits_to_int(self, bits):
        """Pack MSB-first bits along the last axis into integers. `bits`
        is always `half`-wide here (this instance's fixed I/Q axis width),
        so the weight table is the precomputed `self._weights_half`, not
        rebuilt per call."""
        return (bits.astype("int64") * self._weights_half).sum(axis=-1)

    def _int_to_bits(self, ints, nbits: int):
        """Unpack integers into MSB-first bits along a new last axis.
        `nbits` is always this instance's fixed `half`, so the shift
        table is the precomputed `self._shifts_half`, not rebuilt per
        call."""
        return ((ints[..., None] >> self._shifts_half) & 1).astype("uint8")

    def _pam_level(self, binary_idx, nbits: int):
        """Natural-binary index (0..2**nbits-1) -> symmetric odd PAM level
        (-(2**nbits-1), ..., -1, 1, ..., 2**nbits-1). float32, not
        float64 -- this runs for every symbol modulated/demodulated, and
        float64 elementwise math has drastically lower throughput than
        float32 on Jetson-class GPUs (this was a real bug: the array
        used to be built in float64 then immediately downcast, paying
        the double-precision cost for nothing)."""
        return 2 * binary_idx.astype("float32") - (2 ** nbits - 1)

    def _compute_norm_factor(self) -> float:
        """Average-symbol-power normalization (matches the well-known
        IEEE 802.11-style constants: 1, 1/sqrt(2), 1/sqrt(10), 1/sqrt(42),
        1/sqrt(170) for bpsk/qpsk/16/64/256-QAM respectively)."""
        if self.scheme == "bpsk":
            return 1.0
        half = self.bits_per_symbol // 2
        m = 2 ** half
        avg_energy_per_axis = (m * m - 1) / 3.0
        return 1.0 / (2 * avg_energy_per_axis) ** 0.5

    # -- public API -----------------------------------------------------------

    def modulate(self, bits: Any) -> Any:
        xp = self.xp
        bits = xp.asarray(bits)
        if bits.shape[-1] % self.bits_per_symbol != 0:
            raise ValueError(
                f"bit count {bits.shape[-1]} is not a multiple of "
                f"bits_per_symbol={self.bits_per_symbol}"
            )
        n_symbols = bits.shape[-1] // self.bits_per_symbol
        grouped = bits.reshape(bits.shape[0], n_symbols, self.bits_per_symbol)
        norm = self._norm

        if self.scheme == "bpsk":
            b = grouped[..., 0]
            level = 2 * b.astype("float32") - 1  # float32, not float64 -- see _pam_level
            return (level * norm).astype("complex64")

        half = self.bits_per_symbol // 2
        i_gray = self._bits_to_int(grouped[..., :half])
        q_gray = self._bits_to_int(grouped[..., half:])
        i_bin = self._gray_to_binary(i_gray, half)
        q_bin = self._gray_to_binary(q_gray, half)
        i_level = self._pam_level(i_bin, half)
        q_level = self._pam_level(q_bin, half)
        return ((i_level + 1j * q_level) * norm).astype("complex64")

    def _numba_path_applies(self, symbols: Any) -> bool:
        """Transparent Numba-JIT hard decision (see modem/_numba_mapper.py's
        own docstring). Only for backend="numpy" AND complex64 input: the
        kernel's float32 arithmetic is what makes its bits bit-exact with
        the numpy path below, so a complex128 input (rare; nothing in the
        pipeline produces one) keeps the numpy path rather than risking a
        rounding-boundary mismatch. cupy inputs are NOT coerced to host
        here -- same reasoning as sync/schmidl_cox.py: that would
        reintroduce a hidden device<->host round-trip."""
        return (
            self.backend != "cupy"
            and numba_available()
            and isinstance(symbols, np.ndarray)
            and symbols.dtype == np.complex64
            and symbols.ndim == 2
        )

    def demodulate(self, symbols: Any) -> Any:
        """Hard-decision demodulation (nearest constellation point)."""
        symbols = self.xp.asarray(symbols)
        if self._numba_path_applies(symbols):
            bits, _, _ = numba_hard_decision(symbols, self.scheme, self.bits_per_symbol, self._norm)
            return bits
        return self._demodulate_numpy(symbols)

    def _point_table(self):
        """(points, labels) for the whole constellation, built by running
        modulate() over every bit pattern -- so the Gray mapping is never
        written down twice and cannot drift from modulate()'s."""
        if getattr(self, "_pt_cache", None) is None:
            m = self.bits_per_symbol
            labels = np.array([[(i >> (m - 1 - b)) & 1 for b in range(m)]
                               for i in range(1 << m)], dtype="uint8")
            pts = np.asarray(self.modulate(labels.reshape(1, -1)))[0]
            self._pt_cache = (pts.astype("complex64"), labels)
        return self._pt_cache

    def demodulate_soft(self, symbols: Any, weight: Any = None,
                        llr_clip: float = 6.0, llr_bits: Any = None) -> Any:
        """Max-log soft demodulation -> one uint8 per coded bit, in
        libcorrect's convention (0 = certainly 0, 255 = certainly 1,
        128 = no information).

        Why this exists: demodulate() picks the nearest point and returns
        0/1, which DESTROYS how close the symbol was to the decision
        boundary -- and that distance is the only thing a soft decoder can
        use. Feeding hard bits to a soft decoder as 0/255 returns exactly
        the hard answer (asserted in tests/test_modem_soft.py).

        Per bit b:   llr = min_{label_b=0}|y-s|^2 - min_{label_b=1}|y-s|^2
        negative -> bit 0, positive -> bit 1, magnitude = confidence.

        `weight` (optional, broadcast over the last axis) scales that
        confidence per subcarrier. Pass |H[k]|^2 normalized to unit mean:
        after equalization a faded subcarrier's noise is amplified by
        1/|H|^2, so its bits deserve proportionally less trust. This is
        the whole point on a frequency-selective channel -- without it a
        deeply faded bin hands the decoder confident garbage.

        The scale is self-calibrated: the mean squared distance to the
        nearest point estimates the post-equalization noise power, so
        llr/(2*sigma^2) is an LLR in nats and `llr_clip` nats saturates
        the byte range.

        `llr_bits` quantizes the LLR to that many SIGNED bits before it is
        written into the byte -- None (the default) keeps the full 8-bit
        range. This exists for hardware sizing rather than for software:
        an FPGA Viterbi does not want 8-bit branch metrics if 4 carry the
        same coding gain, and the add-compare-select path-metric width
        follows directly from it. `llr_clip` is the other half of that
        question -- bit width alone does not determine performance,
        because the clipping range decides what those levels SPAN.
        """
        xp = self.xp
        y = xp.asarray(symbols)
        if y.ndim == 1:
            y = y[None, :]
        pts, labels = self._point_table()
        pts = xp.asarray(pts)
        d = xp.abs(y[..., None] - pts[None, None, :]) ** 2      # (..., M)
        sigma2 = float(xp.mean(xp.min(d, axis=-1))) or 1e-12
        m = self.bits_per_symbol
        out = xp.empty(y.shape + (m,), dtype="float32")
        for b in range(m):
            zero = xp.asarray(np.flatnonzero(labels[:, b] == 0))
            one = xp.asarray(np.flatnonzero(labels[:, b] == 1))
            out[..., b] = xp.min(d[..., zero], axis=-1) - xp.min(d[..., one], axis=-1)
        llr = out / (2.0 * sigma2)
        if weight is not None:
            w = xp.asarray(weight)
            llr = llr * w[..., None]
        llr = xp.clip(llr / llr_clip, -1.0, 1.0)
        if llr_bits is not None:
            b = int(llr_bits)
            if b < 2:
                raise ValueError(f"llr_bits must be >= 2 (got {b}); 1 bit IS hard decision")
            # Mid-tread signed quantizer: 2*L+1 levels symmetric about 0,
            # L = 2^(b-1)-1. Keeping a level AT zero matters -- that is the
            # "no information" symbol a faded subcarrier needs to be able
            # to produce.
            L = float(2 ** (b - 1) - 1)
            llr = xp.round(llr * L) / L
        soft = xp.clip(xp.round(128.0 + 127.0 * llr), 0, 255).astype("uint8")
        return soft.reshape(y.shape[0], -1)

    def demodulate_stats(self, symbols: Any) -> Tuple[Any, Any, Any]:
        """demodulate() plus the two per-row power sums EVM is built from:
        (bits, sum |symbol - nearest_point|^2, sum |nearest_point|^2), each
        sum over the last axis, so that
        sqrt(err_sum / ref_sum) == compute_evm(symbols, modulate(bits))
        -- what rx_process() computes for its `evm` readout -- WITHOUT the
        demodulate -> modulate round trip: the nearest point is already
        known inside the hard decision. One fused pass on the numba path;
        the numpy fallback below does the literal round trip so both
        paths return the same thing (verified in
        tests/test_modem_numba_acceleration.py)."""
        xp = self.xp
        symbols = xp.asarray(symbols)
        if self._numba_path_applies(symbols):
            return numba_hard_decision(symbols, self.scheme, self.bits_per_symbol, self._norm)
        bits = self._demodulate_numpy(symbols)
        ideal = self.modulate(bits)
        err = xp.sum(xp.abs(symbols - ideal) ** 2, axis=-1)
        ref = xp.sum(xp.abs(ideal) ** 2, axis=-1)
        return bits, err, ref

    def _demodulate_numpy(self, symbols: Any) -> Any:
        """The original xp-vectorized hard decision -- the reference the
        numba kernel is verified bit-exact against, and the path every
        cupy-backend instance still takes."""
        xp = self.xp
        norm = self._norm
        descaled = symbols / norm

        if self.scheme == "bpsk":
            bits = (xp.real(descaled) >= 0).astype("uint8")
            return bits.reshape(symbols.shape[0], -1)

        half = self.bits_per_symbol // 2
        m = 2 ** half

        def _level_to_binary(level):
            b = xp.round((level + (m - 1)) / 2.0)
            return xp.clip(b, 0, m - 1).astype("int64")

        i_bin = _level_to_binary(xp.real(descaled))
        q_bin = _level_to_binary(xp.imag(descaled))
        i_bits = self._int_to_bits(self._binary_to_gray(i_bin), half)
        q_bits = self._int_to_bits(self._binary_to_gray(q_bin), half)
        bits = xp.concatenate([i_bits, q_bits], axis=-1)
        return bits.reshape(symbols.shape[0], -1)

    def process(self, batch: Any, **kwargs: Any) -> Any:
        """Alias for modulate() (bits -> symbols); call demodulate()
        explicitly for the inverse direction."""
        return self.modulate(batch)
