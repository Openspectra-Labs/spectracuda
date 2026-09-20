"""DMRS (demodulation reference symbol) slot arithmetic: where the
periodic channel-refresh symbols land inside a frame's payload region,
and how many data symbols that leaves.

Pure index math on plain ints/numpy -- no OFDM machinery, no backend, no
`xp` (like `framing/header.py`, this is bit/slot bookkeeping that is
useful and testable without constructing an `Ofdm`). `pipeline/ofdm.py`
calls in; nothing here calls back out.

Why DMRS exists at all: `Ofdm` equalizes every payload symbol against a
single channel estimate taken from the training symbol at the *start* of
the frame. Per-symbol pilots then track common phase error continuously,
but they cannot re-measure the frequency-selective shape of H[k]. On a
long frame that initial estimate goes stale. A DMRS is the training
symbol re-transmitted mid-payload so H[k] can be measured again -- see
docs/2026-09-20-dmrs-periodic-channel-refresh-plan.md.

Two conventions this module fixes, both of which are easy to get subtly
wrong and are therefore asserted directly in tests/test_framing_dmrs.py:

1. **The interval counts DATA symbols, not slots.** `interval=32` means
   "a DMRS after every 32 data symbols", so the DMRS-to-DMRS distance is
   33 slots. Counting data symbols is what makes the refresh interval
   *exactly* constant across sample rates: with fft_size=256/cp_len=32,
   16 data symbols at 5 MSps, 32 at 10 MSps and 64 at 20 MSps are all
   921.6us, with zero rounding error. Counting slots would not be.

2. **A trailing DMRS is suppressed.** A DMRS emitted after the last data
   symbol would carry a channel estimate that no data ever uses, so the
   count is `(n_data - 1) // interval`, not `n_data // interval`. With
   interval=32, 32 data symbols get NO DMRS; 33 get one.

`MAX_PAYLOAD_SYMBOLS` is a limit on the TOTAL slot count (data + DMRS),
never on data alone -- see `max_data_symbols()`.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

#: Slot-map marker values (`dmrs_slot_map()`'s dtype is uint8).
DATA_SLOT = 0
DMRS_SLOT = 1

#: Header wire codes for the 2-bit `dmrs_period` field -> interval in
#: data symbols. 0 means "no DMRS in this frame". These four are the
#: only intervals representable on the wire; the functions below accept
#: any non-negative interval so that tests can use small ones (and so a
#: future wire change needs no change here), with the legal-value check
#: living at the `Ofdm`/`HeaderCodec` boundary instead.
DMRS_PERIOD_CODES: Dict[int, int] = {0: 0, 1: 16, 2: 32, 3: 64}
DMRS_PERIOD_INTERVALS: Dict[int, int] = {v: k for k, v in DMRS_PERIOD_CODES.items()}


def _check(n_data_symbols: int, interval: int) -> None:
    if n_data_symbols < 0:
        raise ValueError(f"n_data_symbols must be >= 0, got {n_data_symbols}")
    if interval < 0:
        raise ValueError(f"interval must be >= 0 (0 = off), got {interval}")


def n_dmrs_symbols(n_data_symbols: int, interval: int) -> int:
    """How many DMRS symbols a payload of `n_data_symbols` data symbols
    carries at this interval. `interval=0` disables DMRS entirely.

    The `- 1` is the trailing-DMRS suppression described in the module
    docstring: it is what makes `n_data_symbols == interval` produce
    zero DMRS rather than one stranded at the end.
    """
    _check(n_data_symbols, interval)
    if interval == 0 or n_data_symbols == 0:
        return 0
    return (n_data_symbols - 1) // interval


def total_slots(n_data_symbols: int, interval: int) -> int:
    """Payload-region slot count: data symbols plus their DMRS. This --
    not `n_data_symbols` -- is what `Ofdm.MAX_PAYLOAD_SYMBOLS` bounds."""
    return n_data_symbols + n_dmrs_symbols(n_data_symbols, interval)


def dmrs_slot_indices(n_data_symbols: int, interval: int) -> np.ndarray:
    """Positions of the DMRS symbols within the payload region, as
    indices into the `total_slots()`-long slot sequence.

    The k-th DMRS (k = 1, 2, ...) sits after k*interval data symbols and
    after the k-1 DMRS already emitted, hence `k*interval + (k-1)`.
    """
    n = n_dmrs_symbols(n_data_symbols, interval)
    k = np.arange(1, n + 1, dtype=np.int64)
    return k * interval + (k - 1)


def dmrs_slot_map(n_data_symbols: int, interval: int) -> np.ndarray:
    """The payload region as a flat `total_slots()`-long uint8 array of
    `DATA_SLOT`/`DMRS_SLOT` markers, in transmission order.

    This is the single source of truth both directions use: the
    transmitter writes a payload symbol wherever it reads `DATA_SLOT`
    and the training waveform wherever it reads `DMRS_SLOT`, and the
    receiver extracts against the same map.
    """
    slots = np.full(total_slots(n_data_symbols, interval), DATA_SLOT, dtype=np.uint8)
    slots[dmrs_slot_indices(n_data_symbols, interval)] = DMRS_SLOT
    return slots


def segment_lengths(n_data_symbols: int, interval: int) -> np.ndarray:
    """Data-symbol count per channel-estimate segment, in order.

    A "segment" is the run of data symbols that shares one estimate of
    H[k]: segment 0 uses the frame's training symbol, segment k>0 uses
    the k-th DMRS. There are therefore always `n_dmrs_symbols() + 1`
    segments, and they sum to `n_data_symbols`.

    This is the array the receiver feeds straight to `xp.repeat(...,
    axis=0)` to expand a stack of per-segment channel estimates into one
    estimate per payload symbol -- replacing the single uniform
    `xp.repeat(h_hat_data, n_payload_symbols)` that assumed one estimate
    for the whole frame.
    """
    n = n_dmrs_symbols(n_data_symbols, interval)
    if n == 0:
        return np.array([n_data_symbols], dtype=np.int64)
    return np.array([interval] * n + [n_data_symbols - n * interval], dtype=np.int64)


def max_data_symbols(max_total_slots: int, interval: int) -> int:
    """Largest data-symbol count whose DMRS still fit inside
    `max_total_slots` -- i.e. the largest n with
    `total_slots(n, interval) <= max_total_slots`.

    For MAX_PAYLOAD_SYMBOLS=128 this is 128/127/125/121 at interval
    0/64/32/16, so enabling DMRS costs payload capacity rather than
    buying extra airtime.

    Deliberately a countdown rather than a closed form. The obvious
    `(max_total_slots*interval + 1) // (interval + 1)` is WRONG -- at
    interval=32 it yields 124 where 125 is correct
    (125 + (125-1)//32 = 125 + 3 = 128). This runs at most
    `max_total_slots` iterations, once per construction, and is correct
    by inspection.
    """
    if max_total_slots < 0:
        raise ValueError(f"max_total_slots must be >= 0, got {max_total_slots}")
    if interval < 0:
        raise ValueError(f"interval must be >= 0 (0 = off), got {interval}")
    if interval == 0:
        return max_total_slots
    for n in range(max_total_slots, 0, -1):
        if total_slots(n, interval) <= max_total_slots:
            return n
    return 0
