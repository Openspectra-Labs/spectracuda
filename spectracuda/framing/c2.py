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
