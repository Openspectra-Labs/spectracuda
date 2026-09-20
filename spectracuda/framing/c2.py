"""The C2 (command-and-control) region's fixed PHY profile.

A frame's payload can carry a small critical region ahead of the main
payload -- MAVLink, control, telemetry -- protected by its own CRC and
its own FEC, so that losing the video does not lose the control link:

    main payload fails CRC   =/=>   C2 fails

See docs/2026-09-20-critical-c2-region-plan.md.

**The profile is fixed, not signalled.** There is no `c2_mcs` or
`c2_fec` header field. The receiver does not need to be told which
scheme the C2 region uses, cannot be told the wrong one, and a
corrupted header therefore cannot cause C2 to be decoded with a scheme
the transmitter never used. The only C2 field on the wire is its
length. That is the same reasoning that keeps the DMRS interval to a
2-bit field with four legal values (see `framing/dmrs.py`): fewer
representable states, fewer ways to be wrong.

QPSK + rs_m8 + conv_v27 is not a new, untested combination -- it is the
concatenated pairing already measured on real hardware at roughly 3-4dB
of coding gain (docs/fec-c-lib-acceleration.md), with Viterbi facing
the channel and Reed-Solomon cleaning up its bursty residue.
"""
from __future__ import annotations

from typing import Any, Dict

#: Modulation for the C2 region. Always QPSK, never the header-selected
#: payload modulation -- the whole point is that C2 does not inherit the
#: main payload's (possibly aggressive) MCS.
C2_MODEM = "qpsk"

#: CRC + two-stage FEC for the C2 region, as kwargs for `Packetizer`.
#: fec0/fec1 ordering matches the project convention (fec0 = inner,
#: applied first on encode; fec1 = outer, closest to the channel and
#: decoded first) -- so Viterbi faces the channel and RS mops up after
#: it. See framing/packetizer.py's module docstring for why round the
#: convention goes.
C2_PROFILE: Dict[str, Any] = {
    "crc": "crc32",
    "fec": "rs_m8",
    "fec1": "conv_v27",
    "interleaver": "block",
    "interleaver_kwargs": {"unit_bits": 8},
}

#: Largest C2 payload, in bytes. A policy limit, NOT a wire-format one:
#: `c2_len_bytes` is a uint16 in the header and can express far more, so
#: this is checked separately on both encode and decode. Keeping the
#: field wider than the cap means an over-large decoded value is
#: recognisable as corruption rather than looking like a legal request.
#:
#: At n_data=216 (432 QPSK coded bits per OFDM symbol) the cap costs 15
#: of the 128 payload slots -- 11.7% of the budget, 432us at 10 MSps.
#: A typical 72-byte MAVLink burst costs 5 symbols (3.9%, 144us).
C2_MAX_BYTES = 320


# -- derived sizing ---------------------------------------------------
#
# The C2 region's length in OFDM symbols is DERIVED from c2_len_bytes
# plus the fixed profile, never signalled. The transmitter and the
# receiver must compute it identically or both regions decode as
# garbage, so it is computed in exactly one place -- here -- and both
# sides call it.

_PACKETIZER = None
_BITS_PER_QAM_SYMBOL = None


def _packetizer():
    """One lazily-built Packetizer, reused. Construction is not free
    (native Viterbi trellis + RS GF tables, ~0.15ms measured -- see
    pipeline/ofdm.py's _rx_payload_codec_cache), and `encoded_length()`
    is pure arithmetic over the codec parameters, so a single shared
    instance is safe and avoids paying that cost per call."""
    global _PACKETIZER
    if _PACKETIZER is None:
        from .packetizer import Packetizer

        _PACKETIZER = Packetizer(**C2_PROFILE)
    return _PACKETIZER


def _bits_per_qam_symbol() -> int:
    global _BITS_PER_QAM_SYMBOL
    if _BITS_PER_QAM_SYMBOL is None:
        from ..modem import Modem

        _BITS_PER_QAM_SYMBOL = Modem(C2_MODEM).bits_per_symbol
    return _BITS_PER_QAM_SYMBOL


def check_c2_len(c2_len_bytes: int) -> None:
    if not (0 <= c2_len_bytes <= C2_MAX_BYTES):
        raise ValueError(
            f"c2_len_bytes={c2_len_bytes} outside 0..{C2_MAX_BYTES} "
            f"(C2_MAX_BYTES; 0 means no C2 region)"
        )


def c2_encoded_bits(c2_len_bytes: int) -> int:
    """Wire bits a `c2_len_bytes`-byte C2 payload occupies after the
    fixed profile's CRC and two FEC stages. 0 bytes -> 0 bits (no
    region at all, not an empty-but-present one)."""
    check_c2_len(c2_len_bytes)
    if c2_len_bytes == 0:
        return 0
    return _packetizer().encoded_length(c2_len_bytes * 8)


def bits_per_c2_symbol(n_data: int) -> int:
    """Coded bits one C2 OFDM symbol carries. Always the QPSK rate --
    the C2 region does not inherit the header-selected payload
    modulation, which is the entire point of it."""
    if n_data <= 0:
        raise ValueError(f"n_data must be > 0, got {n_data}")
    return n_data * _bits_per_qam_symbol()


def n_c2_symbols(c2_len_bytes: int, n_data: int) -> int:
    """OFDM symbols the C2 region occupies -- THE boundary both sides
    must agree on.

    Ceiling division, matching how the main payload is sized: a partial
    last symbol is padded rather than truncated, and the receiver
    recovers the real bit count from `c2_len_bytes` alone, so no
    padding length is needed on the wire.
    """
    encoded = c2_encoded_bits(c2_len_bytes)
    if encoded == 0:
        return 0
    per_symbol = bits_per_c2_symbol(n_data)
    return -(-encoded // per_symbol)
