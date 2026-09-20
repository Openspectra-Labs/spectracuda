"""DMRS transmit-side insertion (plan step 2): generate_frame() lays
payload symbols and DMRS out against framing/dmrs.py's slot map.

The receiver does not understand DMRS yet -- that is step 3 -- so these
assert the emitted waveform directly rather than round-tripping it.

See docs/2026-09-20-dmrs-periodic-channel-refresh-plan.md.
"""
import numpy as np
import pytest

from spectracuda.framing import dmrs as D
from spectracuda.pipeline import Ofdm

FFT, CP, NDATA, NPILOT = 256, 32, 216, 8
SLOT = FFT + CP


def make(dmrs_interval=0, **kw):
    return Ofdm(
        fft_size=FFT, n_pilot=NPILOT, n_data=NDATA, cp_len=CP,
        modem="qpsk", fec="none", crc="crc32",
        dmrs_interval=dmrs_interval, **kw
    )


def payload_bits(n_bits, seed=0):
    return np.random.default_rng(seed).integers(0, 2, size=(1, n_bits)).astype("uint8")


def overhead_samples(ofdm):
    """preamble (no CP) + training symbol(s) + header symbol(s)."""
    return FFT + SLOT * (ofdm.n_training_symbols + ofdm.num_symbols_header)


def n_data_symbols_for(ofdm, bits):
    encoded = ofdm.packetizer.encode(bits).shape[-1]
    return -(-encoded // ofdm.bits_per_ofdm_symbol)


# -- constructor ------------------------------------------------------


def test_defaults_to_off():
    assert make().dmrs_interval == 0


@pytest.mark.parametrize("interval", [0, 16, 32, 64])
def test_accepts_the_four_wire_values(interval):
    assert make(interval).dmrs_interval == interval


@pytest.mark.parametrize("bad", [1, 8, 15, 33, 128, -16])
def test_rejects_intervals_the_header_cannot_carry(bad):
    """The 2-bit dmrs_period field can only express 0/16/32/64, so the
    constructor is where a non-representable value must be caught --
    framing/dmrs.py's arithmetic is deliberately general."""
    with pytest.raises(ValueError, match="dmrs_interval"):
        make(bad)


@pytest.mark.parametrize(
    "interval,expected", [(0, 128), (64, 127), (32, 125), (16, 121)]
)
def test_max_data_symbols_drops_as_dmrs_density_rises(interval, expected):
    """DMRS comes out of MAX_PAYLOAD_SYMBOLS, it does not extend it."""
    ofdm = make(interval)
    assert ofdm.max_data_symbols == expected
    assert D.total_slots(expected, interval) == Ofdm.MAX_PAYLOAD_SYMBOLS


# -- frame geometry ---------------------------------------------------


def test_interval_off_frame_length_is_unchanged():
    """Regression gate: with DMRS off the frame must be exactly what it
    was before this feature existed."""
    ofdm = make(0)
    bits = payload_bits(40_000)
    frame = ofdm.generate_frame(bits)
    n_data = n_data_symbols_for(ofdm, bits)
    assert frame.shape[-1] == overhead_samples(ofdm) + n_data * SLOT


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_frame_grows_by_exactly_the_dmrs_slots(interval):
    ofdm = make(interval)
    bits = payload_bits(40_000)
    n_data = n_data_symbols_for(ofdm, bits)
    n_dmrs = D.n_dmrs_symbols(n_data, interval)
    assert n_dmrs > 0, "test payload too short to exercise this interval"

    frame = ofdm.generate_frame(bits)
    expected = overhead_samples(ofdm) + (n_data + n_dmrs) * SLOT
    assert frame.shape[-1] == expected
    assert frame.shape[-1] - make(0).generate_frame(bits).shape[-1] == n_dmrs * SLOT


def test_short_frame_carries_no_dmrs_at_the_diagonal_interval():
    """A ~1ms TXOP at 10 MSps is ~31 payload symbols; at interval 32 it
    must be byte-identical to the DMRS-off frame."""
    bits = payload_bits(6_000)
    off = make(0).generate_frame(bits)
    on = make(32).generate_frame(bits)
    assert n_data_symbols_for(make(32), bits) <= 32
    np.testing.assert_array_equal(np.asarray(on), np.asarray(off))


# -- slot contents ----------------------------------------------------


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_every_dmrs_slot_is_the_training_waveform(interval):
    """A DMRS *is* the training symbol re-transmitted -- that is what
    lets the receiver reuse the existing channel estimator unchanged."""
    ofdm = make(interval)
    bits = payload_bits(40_000)
    frame = np.asarray(ofdm.generate_frame(bits))

    training = np.asarray(ofdm.mod.process(ofdm._train_grid_freq[None, :])[0])
    base = overhead_samples(ofdm)
    slot_map = D.dmrs_slot_map(n_data_symbols_for(ofdm, bits), interval)

    for i in np.where(slot_map == D.DMRS_SLOT)[0]:
        np.testing.assert_array_equal(frame[0, base + i * SLOT: base + (i + 1) * SLOT], training)


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_payload_symbols_are_repositioned_not_altered(interval):
    """The strong check: pulling the DATA slots back out of a DMRS frame
    must reproduce the DMRS-off frame's payload region exactly. Proves
    insertion only moves payload symbols -- it does not renumber,
    re-pad or re-modulate them."""
    bits = payload_bits(40_000)
    off_ofdm, on_ofdm = make(0), make(interval)
    off = np.asarray(off_ofdm.generate_frame(bits))
    on = np.asarray(on_ofdm.generate_frame(bits))

    base = overhead_samples(on_ofdm)
    n_data = n_data_symbols_for(on_ofdm, bits)
    slot_map = D.dmrs_slot_map(n_data, interval)
    data_at = np.where(slot_map == D.DATA_SLOT)[0]

    recovered = np.concatenate([on[0, base + i * SLOT: base + (i + 1) * SLOT] for i in data_at])
    np.testing.assert_array_equal(recovered, off[0, base:])


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_preamble_training_and_header_are_untouched(interval):
    bits = payload_bits(40_000)
    off = np.asarray(make(0).generate_frame(bits))
    on = np.asarray(make(interval).generate_frame(bits))
    base = overhead_samples(make(interval))
    np.testing.assert_array_equal(on[0, :base], off[0, :base])


# -- the 128 total-slot guard -----------------------------------------


@pytest.mark.parametrize("interval", [0, 16, 32, 64])
def test_max_data_symbols_worth_of_payload_is_accepted(interval):
    """Exactly at the ceiling must work, not merely below it."""
    ofdm = make(interval)
    bits = payload_bits(ofdm.max_data_symbols * ofdm.bits_per_ofdm_symbol - 32)
    frame = ofdm.generate_frame(bits)
    n_data = n_data_symbols_for(ofdm, bits)
    assert n_data == ofdm.max_data_symbols
    assert D.total_slots(n_data, interval) == Ofdm.MAX_PAYLOAD_SYMBOLS
    assert frame.shape[-1] == overhead_samples(ofdm) + Ofdm.MAX_PAYLOAD_SYMBOLS * SLOT


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_one_symbol_past_the_ceiling_raises(interval):
    ofdm = make(interval)
    bits = payload_bits((ofdm.max_data_symbols + 1) * ofdm.bits_per_ofdm_symbol - 32)
    with pytest.raises(ValueError, match="MAX_PAYLOAD_SYMBOLS"):
        ofdm.generate_frame(bits)


def test_guard_message_names_data_dmrs_and_total():
    """A frame rejected at 125 data + 3 DMRS must not read as though it
    were rejected at 128 data symbols."""
    ofdm = make(32)
    bits = payload_bits((ofdm.max_data_symbols + 1) * ofdm.bits_per_ofdm_symbol - 32)
    with pytest.raises(ValueError) as excinfo:
        ofdm.generate_frame(bits)
    msg = str(excinfo.value)
    assert "126 data" in msg and "3 DMRS" in msg and "129" in msg
    assert "dmrs_interval=32" in msg and "ceiling is 125" in msg


def test_dmrs_cannot_buy_extra_airtime():
    """The whole point of the total-slot rule: a maximum-length frame is
    the same length whatever the interval."""
    lengths = set()
    for interval in (0, 16, 32, 64):
        ofdm = make(interval)
        bits = payload_bits(ofdm.max_data_symbols * ofdm.bits_per_ofdm_symbol - 32)
        lengths.add(int(ofdm.generate_frame(bits).shape[-1]))
    assert len(lengths) == 1
