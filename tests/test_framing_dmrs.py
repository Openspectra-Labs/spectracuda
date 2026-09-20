"""DMRS slot arithmetic. Every convention here is one an implementation
can get subtly wrong while still looking plausible -- an off-by-one in
the trailing-DMRS rule or in the slot positions produces a frame that
transmits and decodes, just against the wrong channel estimate -- so
these assert the numbers directly rather than round-tripping them.

See docs/2026-09-20-dmrs-periodic-channel-refresh-plan.md.
"""
import numpy as np
import pytest

from spectracuda.framing.dmrs import (
    DATA_SLOT,
    DMRS_PERIOD_CODES,
    DMRS_PERIOD_INTERVALS,
    DMRS_SLOT,
    dmrs_slot_indices,
    dmrs_slot_map,
    max_data_symbols,
    n_dmrs_symbols,
    segment_lengths,
    total_slots,
)

MAX_PAYLOAD_SYMBOLS = 128  # mirrors Ofdm.MAX_PAYLOAD_SYMBOLS


# -- the trailing-DMRS rule -------------------------------------------


def test_interval_off_never_inserts():
    for n in (0, 1, 31, 128, 10_000):
        assert n_dmrs_symbols(n, 0) == 0
        assert total_slots(n, 0) == n


def test_exactly_one_interval_of_data_gets_no_dmrs():
    """32 data symbols at interval 32 needs no refresh -- a DMRS after
    the last data symbol would carry an estimate nothing uses. This is
    the `- 1` in (n_data - 1) // interval."""
    assert n_dmrs_symbols(32, 32) == 0
    assert n_dmrs_symbols(33, 32) == 1


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_first_dmrs_appears_one_symbol_after_the_interval(interval):
    assert n_dmrs_symbols(interval, interval) == 0
    assert n_dmrs_symbols(interval + 1, interval) == 1


def test_short_frames_never_pay():
    """A ~1ms TXOP at 10 MSps is 31 payload symbols; at the diagonal
    interval (32) it carries no DMRS at all."""
    assert n_dmrs_symbols(31, 32) == 0
    assert n_dmrs_symbols(31, 16) == 1


# -- the 128 limit is a TOTAL -----------------------------------------


@pytest.mark.parametrize(
    "interval,expected_data,expected_dmrs",
    [(0, 128, 0), (64, 127, 1), (32, 125, 3), (16, 121, 7)],
)
def test_max_data_symbols_fills_exactly_128_slots(interval, expected_data, expected_dmrs):
    """DMRS comes OUT of MAX_PAYLOAD_SYMBOLS, never on top of it: the
    data ceiling drops instead, and the total lands on exactly 128."""
    n = max_data_symbols(MAX_PAYLOAD_SYMBOLS, interval)
    assert n == expected_data
    assert n_dmrs_symbols(n, interval) == expected_dmrs
    assert total_slots(n, interval) == MAX_PAYLOAD_SYMBOLS


@pytest.mark.parametrize("interval", [0, 16, 32, 64])
def test_one_more_data_symbol_would_overflow(interval):
    """max_data_symbols() is the LARGEST such n, not merely a safe one."""
    n = max_data_symbols(MAX_PAYLOAD_SYMBOLS, interval)
    assert total_slots(n + 1, interval) > MAX_PAYLOAD_SYMBOLS


def test_closed_form_shortcut_would_be_wrong():
    """Guards the countdown in max_data_symbols() against being
    'simplified' into the closed form, which is off by one at
    interval=32 (yields 124, correct answer is 125)."""
    interval = 32
    tempting = (MAX_PAYLOAD_SYMBOLS * interval + 1) // (interval + 1)
    assert tempting == 124
    assert max_data_symbols(MAX_PAYLOAD_SYMBOLS, interval) == 125


# -- slot map ---------------------------------------------------------


def test_slot_positions_account_for_preceding_dmrs():
    """The k-th DMRS sits at k*interval + (k-1): each earlier DMRS has
    already consumed a slot, so they are NOT at plain multiples of the
    interval."""
    assert dmrs_slot_indices(125, 32).tolist() == [32, 65, 98]


def test_slot_map_matches_the_worked_example_from_the_plan():
    """53 data symbols at interval 16 -> 3 DMRS, 56 slots."""
    slots = dmrs_slot_map(53, 16)
    assert len(slots) == 56
    assert int((slots == DMRS_SLOT).sum()) == 3
    assert int((slots == DATA_SLOT).sum()) == 53
    rendered = "".join("D" if s == DMRS_SLOT else "d" for s in slots)
    assert rendered == "d" * 16 + "D" + "d" * 16 + "D" + "d" * 16 + "D" + "d" * 5


@pytest.mark.parametrize("interval", [0, 4, 16, 32, 64])
@pytest.mark.parametrize("n_data", [0, 1, 17, 53, 121, 125, 128])
def test_slot_map_is_self_consistent(interval, n_data):
    slots = dmrs_slot_map(n_data, interval)
    assert len(slots) == total_slots(n_data, interval)
    assert int((slots == DATA_SLOT).sum()) == n_data
    assert int((slots == DMRS_SLOT).sum()) == n_dmrs_symbols(n_data, interval)


@pytest.mark.parametrize("interval", [4, 16, 32])
@pytest.mark.parametrize("n_data", [1, 17, 53, 125])
def test_no_dmrs_is_ever_the_final_slot(interval, n_data):
    """A trailing DMRS would refresh H[k] for data that does not exist."""
    slots = dmrs_slot_map(n_data, interval)
    assert slots[-1] == DATA_SLOT


@pytest.mark.parametrize("interval", [4, 16, 32])
@pytest.mark.parametrize("n_data", [17, 53, 125])
def test_data_symbols_between_dmrs_equal_the_interval(interval, n_data):
    """The gaps the interval actually promises: every run of data before
    a DMRS is exactly `interval` long."""
    slots = dmrs_slot_map(n_data, interval)
    (dmrs_at,) = np.where(slots == DMRS_SLOT)
    boundaries = np.concatenate([[-1], dmrs_at])
    runs = np.diff(boundaries) - 1
    assert runs.tolist() == [interval] * len(dmrs_at)


# -- segment lengths (what the RX feeds to xp.repeat) -----------------


def test_segment_lengths_cover_every_data_symbol_once():
    """125 data symbols at interval 32: three full segments then the
    29-symbol remainder, one segment per channel estimate."""
    assert segment_lengths(125, 32).tolist() == [32, 32, 32, 29]


@pytest.mark.parametrize("interval", [0, 4, 16, 32, 64])
@pytest.mark.parametrize("n_data", [0, 1, 17, 53, 121, 125, 128])
def test_one_segment_per_channel_estimate(interval, n_data):
    """There is always exactly one more segment than there are DMRS
    (segment 0 belongs to the training symbol), and the segments account
    for every data symbol exactly once -- the invariant that keeps
    xp.repeat()'s output aligned with the payload symbol sequence."""
    lengths = segment_lengths(n_data, interval)
    assert len(lengths) == n_dmrs_symbols(n_data, interval) + 1
    assert int(lengths.sum()) == n_data


def test_segment_lengths_align_with_the_slot_map():
    """Independent cross-check: walking the slot map must reproduce
    segment_lengths() exactly."""
    for interval in (4, 16, 32):
        for n_data in (1, 17, 53, 125):
            walked, run = [], 0
            for slot in dmrs_slot_map(n_data, interval):
                if slot == DMRS_SLOT:
                    walked.append(run)
                    run = 0
                else:
                    run += 1
            walked.append(run)
            assert walked == segment_lengths(n_data, interval).tolist()


# -- wire codes -------------------------------------------------------


def test_period_codes_are_a_2_bit_field():
    assert sorted(DMRS_PERIOD_CODES) == [0, 1, 2, 3]
    assert DMRS_PERIOD_CODES == {0: 0, 1: 16, 2: 32, 3: 64}
    assert DMRS_PERIOD_INTERVALS[32] == 2


def test_period_code_round_trip():
    for code, interval in DMRS_PERIOD_CODES.items():
        assert DMRS_PERIOD_INTERVALS[interval] == code


# -- input validation -------------------------------------------------


@pytest.mark.parametrize("bad", [-1, -32])
def test_negative_interval_rejected(bad):
    with pytest.raises(ValueError, match="interval"):
        n_dmrs_symbols(10, bad)


def test_negative_data_symbols_rejected():
    with pytest.raises(ValueError, match="n_data_symbols"):
        n_dmrs_symbols(-1, 32)
