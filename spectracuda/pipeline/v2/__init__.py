"""`OfdmV2` -- the PHY re-partitioned into time / frequency / bit domain
blocks separated by FIFOs, one thread per block, so the receive load
spreads over several CPU cores instead of stacking on one.

Read `stage_if.py` first: it carries the dataflow contract and the three
rules that make the split safe (strictly forward, per-stage state
ownership, published arrays immutable).

`pipeline/ofdm.py`'s `Ofdm` is NOT touched by any of this. It remains the
shipping path and is this package's bit-exactness oracle.
"""
from __future__ import annotations

from .stage_if import (
    ControlEvent,
    FftBatch,
    FrameAbort,
    HeaderConfig,
    LlrBatch,
    SoftConfig,
    StreamGap,
    SymbolType,
)

__all__ = [
    "ControlEvent",
    "FftBatch",
    "FrameAbort",
    "HeaderConfig",
    "LlrBatch",
    "SoftConfig",
    "StreamGap",
    "SymbolType",
]
