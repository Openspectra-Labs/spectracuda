"""DMRS periodicity on the wire (plan step 4): the 2-bit `dmrs_period`
field in HeaderCodec byte 5, and the receiver resolving the interval
from the decoded header instead of from its own configuration.

This is what makes the feature usable between two independently
configured radios. Before it, the receiver read self.dmrs_interval, so
both ends had to be set up by hand to match.

Byte 5 was `fec1 & 0x1F` -- its top three bits were unconditionally zero
and the decoder already masked them off, so the field costs no header
space and an older decoder still recovers fec1 correctly.

See docs/2026-09-20-dmrs-periodic-channel-refresh-plan.md.
"""
import numpy as np
import pytest

from spectracuda.framing import HeaderCodec
from spectracuda.framing.dmrs import DMRS_PERIOD_CODES
from spectracuda.pipeline import Ofdm

INTERVALS = [0, 16, 32, 64]


def make(dmrs_interval=0):
    return Ofdm(
        fft_size=256, n_pilot=8, n_data=216, cp_len=32,
        modem="qpsk", fec="none", crc="crc32", dmrs_interval=dmrs_interval,
    )


def payload(n_bits=40_000, seed=0):
    return np.random.default_rng(seed).integers(0, 2, size=(1, n_bits)).astype("uint8")


# -- codec ------------------------------------------------------------


@pytest.mark.parametrize("interval", INTERVALS)
def test_round_trips_every_wire_value(interval):
    codec = HeaderCodec()
    bits = codec.encode_bits(1000, "qpsk", "conv_v27", None, "crc32", "rs_m8", interval)
    assert codec.decode_bits(bits)["dmrs_interval"] == interval


@pytest.mark.parametrize("fec1", ["none", "conv_v27", "rs_m8"])
def test_fec1_survives_alongside_a_nonzero_dmrs_period(fec1):
    """The two share byte 5. A mask error would corrupt fec1 only for
    certain intervals, which is exactly the kind of bug that hides."""
    codec = HeaderCodec()
    for interval in INTERVALS:
        bits = codec.encode_bits(1000, "qpsk", "conv_v27", None, "crc32", fec1, interval)
        decoded = codec.decode_bits(bits)
        assert decoded["fec1"] == fec1
        assert decoded["dmrs_interval"] == interval


@pytest.mark.parametrize("interval", INTERVALS)
def test_byte_5_bit_7_stays_reserved(interval):
    """Bit 7 is left free for a future denser interval. If the field
    ever silently widened, this catches it."""
    codec = HeaderCodec()
    bits = codec.encode_bits(1000, "qpsk", "conv_v27", None, "crc32", "rs_m8", interval)
    unscrambled = np.asarray(bits, dtype="uint8") ^ codec._scramble_mask
    byte5 = np.packbits(unscrambled).tobytes()[5]
    assert byte5 & 0x80 == 0


def test_defaults_to_off_when_not_passed():
    """Every existing caller omits the argument and must keep working."""
    codec = HeaderCodec()
    bits = codec.encode_bits(1000, "qpsk", "conv_v27", None, "crc32", "rs_m8")
    assert codec.decode_bits(bits)["dmrs_interval"] == 0


def test_every_2_bit_code_decodes():
    """Unlike fec/crc, no dmrs_period bit pattern is invalid -- all four
    codes must resolve, so header corruption can never raise here."""
    assert sorted(DMRS_PERIOD_CODES) == [0, 1, 2, 3]
    assert sorted(DMRS_PERIOD_CODES.values()) == INTERVALS


@pytest.mark.parametrize("bad", [1, 8, 33, 128])
def test_encode_rejects_unrepresentable_intervals(bad):
    with pytest.raises(ValueError, match="dmrs_interval"):
        HeaderCodec().encode_bits(1000, "qpsk", "conv_v27", None, "crc32", "none", bad)


# -- the receiver resolves it from the header -------------------------


@pytest.mark.parametrize("interval", INTERVALS)
def test_receiver_never_told_the_transmitters_interval(interval):
    """The headline behavior. The receiver is built with DMRS OFF and
    still decodes a frame carrying any interval, because it reads the
    header -- the same way it already resolves mod_scheme and fec0."""
    tx, rx = make(interval), make(0)
    bits = payload()
    result = rx.rx_process(tx.generate_frame(bits))
    assert result["header"]["dmrs_interval"] == interval
    assert bool(np.asarray(result["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(result["bits"])[0][: bits.shape[1]], bits[0])


@pytest.mark.parametrize("rx_interval", INTERVALS)
def test_receivers_own_setting_is_ignored_on_receive(rx_interval):
    """A receiver configured with a DIFFERENT interval than the sender
    must still decode: self.dmrs_interval is transmit-side only. If the
    receiver used its own value it would slice the payload region at the
    wrong offsets and fail."""
    tx, rx = make(32), make(rx_interval)
    bits = payload()
    result = rx.rx_process(tx.generate_frame(bits))
    assert result["header"]["dmrs_interval"] == 32
    assert bool(np.asarray(result["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(result["bits"])[0][: bits.shape[1]], bits[0])


def test_mismatched_ends_still_track_a_time_varying_channel():
    """End to end, with the impairment DMRS exists for and a receiver
    that was never configured for it."""
    from test_ofdm_dmrs_rx import time_varying_two_ray

    tx, rx = make(32), make(0)
    bits = payload(125 * tx.bits_per_ofdm_symbol - 32)
    assert bool(np.asarray(rx.rx_process(time_varying_two_ray(tx.generate_frame(bits)))["crc_valid"])[0])

    tx_off = make(0)
    r_off = rx.rx_process(time_varying_two_ray(tx_off.generate_frame(bits)))
    assert not bool(np.asarray(r_off["crc_valid"])[0])


# -- reconfigure_tx_scheme --------------------------------------------


@pytest.mark.parametrize("interval", INTERVALS)
def test_reconfigure_changes_what_is_transmitted(interval):
    tx, rx = make(0), make(0)
    tx.reconfigure_tx_scheme(dmrs_interval=interval)
    assert tx.dmrs_interval == interval
    result = rx.rx_process(tx.generate_frame(payload()))
    assert result["header"]["dmrs_interval"] == interval
    assert bool(np.asarray(result["crc_valid"])[0])


def test_reconfigure_moves_the_data_ceiling():
    """Callers with their own segmentation math must re-read
    max_data_symbols after this -- bits_per_ofdm_symbol does not move."""
    ofdm = make(0)
    assert ofdm.max_data_symbols == 128
    before = ofdm.reconfigure_tx_scheme(dmrs_interval=16)
    assert ofdm.max_data_symbols == 121
    assert before == ofdm.bits_per_ofdm_symbol  # unchanged, hence insufficient on its own


def test_reconfigure_leaves_other_settings_alone():
    ofdm = make(0)
    ofdm.reconfigure_tx_scheme(dmrs_interval=32)
    assert (ofdm.modem.scheme, ofdm.fec, ofdm.fec1, ofdm.crc) == ("qpsk", "none", "none", "crc32")


def test_reconfigure_validates():
    with pytest.raises(ValueError, match="dmrs_interval"):
        make(0).reconfigure_tx_scheme(dmrs_interval=8)


def test_reconfigure_without_the_argument_is_a_no_op_for_dmrs():
    ofdm = make(32)
    ofdm.reconfigure_tx_scheme(modem="qam16")
    assert ofdm.dmrs_interval == 32
