"""Framing layer: header codec, CRC+FEC packetizer, and packet stats --
the pieces that turn decoded OFDM symbols into a validated payload,
independent of the OFDM modem engine itself (liquid-dsp's equivalent
separation: `packetizer` knows nothing about `ofdmflexframegen`/
`ofdmflexframesync`; those call INTO it, not the reverse).

`Ofdm` (spectracuda/pipeline/ofdm.py) owns one `HeaderCodec` and one
`Packetizer` instance and delegates to them, rather than doing this
logic inline -- see docs/todo.md #1.1 for the gap this closes ("you
can't reuse 'decode a framed packet' logic outside Ofdm itself").
"""
from .c2 import C2_MAX_BYTES, C2_MODEM, C2_PROFILE
from .dmrs import (
    DMRS_PERIOD_CODES,
    DMRS_PERIOD_INTERVALS,
    dmrs_slot_map,
    max_data_symbols,
    n_dmrs_symbols,
    segment_lengths,
    total_slots,
)
from .header import HeaderCodec
from .packetizer import Packetizer
from .stats import compute_evm, compute_rssi_db

__all__ = [
    "HeaderCodec",
    "C2_MAX_BYTES",
    "C2_MODEM",
    "C2_PROFILE",
    "Packetizer",
    "compute_evm",
    "compute_rssi_db",
    "DMRS_PERIOD_CODES",
    "DMRS_PERIOD_INTERVALS",
    "dmrs_slot_map",
    "max_data_symbols",
    "n_dmrs_symbols",
    "segment_lengths",
    "total_slots",
]
