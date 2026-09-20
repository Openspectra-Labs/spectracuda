"""C2 region transmit-side emission (critical-region plan step 3).

The receiver does not understand the region yet -- that is step 4 -- so
these pull the C2 symbols back out of the emitted waveform by hand and
decode them directly, rather than round-tripping through rx_process().

See docs/2026-09-20-critical-c2-region-plan.md.
"""
import numpy as np
import pytest

from spectracuda.framing import c2 as C
from spectracuda.framing import dmrs as D
from spectracuda.pipeline import Ofdm

FFT, CP, NDATA, NPILOT = 256, 32, 216, 8
SLOT = FFT + CP


def make(dmrs_interval=0, modem="qpsk"):
    return Ofdm(
        fft_size=FFT, n_pilot=NPILOT, n_data=NDATA, cp_len=CP,
        modem=modem, fec="none", crc="crc32", dmrs_interval=dmrs_interval,
    )


def bits(n, seed=0):
    return np.random.default_rng(seed).integers(0, 2, size=(1, n)).astype("uint8")


def overhead(ofdm):
    return FFT + SLOT * (ofdm.n_training_symbols + ofdm.num_symbols_header)


def n_main_symbols(ofdm, payload):
    return -(-ofdm.packetizer.encode(payload).shape[-1] // ofdm.bits_per_ofdm_symbol)


def extract_c2(ofdm, frame, n_c2, n_total_data, interval, c2_len_bytes):
    """Pull the C2 slots out of an emitted frame and decode them."""
    slot_map = D.dmrs_slot_map(n_total_data, interval)
    data_at = np.where(slot_map == D.DATA_SLOT)[0][:n_c2]
    base = overhead(ofdm)
    slots = np.stack([np.asarray(frame)[0, base + i * SLOT: base + (i + 1) * SLOT] for i in data_at])
    rx_grid = ofdm.demod.process(slots)
    recovered = np.asarray(
        ofdm.c2_modem.demodulate(ofdm.grid.extract_data(ofdm.xp, rx_grid))
    ).reshape(1, -1)
    wire_len = ofdm.c2_packetizer.encoded_length(c2_len_bytes * 8)
    return ofdm.c2_packetizer.decode(recovered[:, :wire_len])


# -- no C2 is unchanged -----------------------------------------------


def test_no_c2_is_the_default():
    ofdm = make()
    payload = bits(20_000)
    frame = ofdm.generate_frame(payload)
    assert frame.shape[-1] == overhead(ofdm) + n_main_symbols(ofdm, payload) * SLOT


def test_c2_bits_none_leaves_the_header_field_zero():
    ofdm = make()
    result_header = ofdm.rx_process(ofdm.generate_frame(bits(20_000)))["header"]
    assert result_header["c2_len_bytes"] == 0


# -- the region is emitted --------------------------------------------


@pytest.mark.parametrize("c2_len", [1, 16, 72, 128, 320])
def test_frame_grows_by_exactly_the_c2_symbols(c2_len):
    ofdm = make()
    payload = bits(20_000)
    without = ofdm.generate_frame(payload).shape[-1]
    with_c2 = ofdm.generate_frame(payload, c2_bits=bits(c2_len * 8, seed=1)).shape[-1]
    assert with_c2 - without == C.n_c2_symbols(c2_len, NDATA) * SLOT


@pytest.mark.parametrize("c2_len", [16, 72, 320])
def test_header_carries_the_c2_length(c2_len):
    ofdm = make()
    frame = ofdm.generate_frame(bits(20_000), c2_bits=bits(c2_len * 8, seed=1))
    assert ofdm.rx_process(frame)["header"]["c2_len_bytes"] == c2_len


@pytest.mark.parametrize("c2_len", [1, 16, 72, 128, 320])
def test_c2_symbols_decode_back_to_the_c2_bits(c2_len):
    ofdm = make()
    payload = bits(20_000)
    c2 = bits(c2_len * 8, seed=1)
    frame = ofdm.generate_frame(payload, c2_bits=c2)
    decoded = extract_c2(ofdm, frame, C.n_c2_symbols(c2_len, NDATA),
                         C.n_c2_symbols(c2_len, NDATA) + n_main_symbols(ofdm, payload),
                         0, c2_len)
    assert bool(np.asarray(decoded["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(decoded["bits"])[0][: c2_len * 8], c2[0])


@pytest.mark.parametrize("main_modem", ["qpsk", "qam16", "qam64", "qam256"])
def test_c2_stays_qpsk_whatever_the_main_payload_uses(main_modem):
    """The whole point: C2 does not inherit the header-selected MCS.
    With a 256QAM main payload the two regions carry 8x different bits
    per symbol, so a region that silently used self.modem would produce
    the wrong symbol count and fail to decode here."""
    ofdm = make(modem=main_modem)
    payload = bits(20_000)
    c2 = bits(72 * 8, seed=1)
    frame = ofdm.generate_frame(payload, c2_bits=c2)
    n_c2 = C.n_c2_symbols(72, NDATA)
    decoded = extract_c2(ofdm, frame, n_c2, n_c2 + n_main_symbols(ofdm, payload), 0, 72)
    assert bool(np.asarray(decoded["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(decoded["bits"])[0][: 72 * 8], c2[0])


# -- interaction with DMRS --------------------------------------------


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_dmrs_counts_across_both_regions(interval):
    """The refresh interval runs over every payload-carrying symbol,
    not just the main payload -- the channel does not care which region
    a symbol belongs to."""
    ofdm = make(dmrs_interval=interval)
    c2_len = 320
    n_c2 = C.n_c2_symbols(c2_len, NDATA)
    # Size the main payload so it sits just BELOW the first refresh on
    # its own, and only crosses it once the C2 symbols are counted too.
    # Otherwise the test passes whether or not C2 is included in the
    # count, and proves nothing.
    target_main = interval - 10
    payload = bits(target_main * ofdm.bits_per_ofdm_symbol - 32)
    n_main = n_main_symbols(ofdm, payload)
    assert n_main == target_main

    expected_dmrs = D.n_dmrs_symbols(n_c2 + n_main, interval)
    assert D.n_dmrs_symbols(n_main, interval) == 0, "main alone must not trigger a refresh"
    assert expected_dmrs == 1, "C2 must push it over the first refresh"

    frame = ofdm.generate_frame(payload, c2_bits=bits(c2_len * 8, seed=1))
    expected = overhead(ofdm) + (n_c2 + n_main + expected_dmrs) * SLOT
    assert frame.shape[-1] == expected


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_c2_decodes_with_dmrs_interleaved(interval):
    ofdm = make(dmrs_interval=interval)
    payload = bits(20_000)
    c2 = bits(72 * 8, seed=1)
    frame = ofdm.generate_frame(payload, c2_bits=c2)
    n_c2 = C.n_c2_symbols(72, NDATA)
    decoded = extract_c2(ofdm, frame, n_c2, n_c2 + n_main_symbols(ofdm, payload), interval, 72)
    assert bool(np.asarray(decoded["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(decoded["bits"])[0][: 72 * 8], c2[0])


# -- C2-only frames ---------------------------------------------------


def test_c2_only_frame():
    """The MAC needs this when control traffic is pending and there is
    no data to send. A zero-length main payload is already legal."""
    ofdm = make()
    c2 = bits(72 * 8, seed=1)
    frame = ofdm.generate_frame(np.zeros((1, 0), "uint8"), c2_bits=c2)
    n_c2 = C.n_c2_symbols(72, NDATA)
    decoded = extract_c2(ofdm, frame, n_c2, n_c2 + 1, 0, 72)  # main pads to 1 symbol
    assert bool(np.asarray(decoded["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(decoded["bits"])[0][: 72 * 8], c2[0])


# -- the three-term budget --------------------------------------------


@pytest.mark.parametrize("interval", [0, 16, 32, 64])
@pytest.mark.parametrize("c2_len", [0, 72, 320])
def test_max_data_symbols_for_fills_exactly_the_budget(interval, c2_len):
    ofdm = make(dmrs_interval=interval)
    n_c2 = C.n_c2_symbols(c2_len, NDATA)
    n_main = ofdm.max_data_symbols_for(c2_len)
    total = n_c2 + n_main
    assert total + D.n_dmrs_symbols(total, interval) == Ofdm.MAX_PAYLOAD_SYMBOLS


def test_c2_costs_payload_capacity_not_airtime():
    """C2 comes OUT of the 128-slot budget, exactly as DMRS does."""
    ofdm = make()
    assert ofdm.max_data_symbols_for(0) == 128
    assert ofdm.max_data_symbols_for(72) == 128 - C.n_c2_symbols(72, NDATA)
    assert ofdm.max_data_symbols_for(320) == 128 - C.n_c2_symbols(320, NDATA)


def test_over_budget_names_all_three_terms():
    ofdm = make(dmrs_interval=32)
    too_big = (ofdm.max_data_symbols_for(320) + 2) * ofdm.bits_per_ofdm_symbol - 32
    with pytest.raises(ValueError) as excinfo:
        ofdm.generate_frame(bits(too_big), c2_bits=bits(320 * 8, seed=1))
    msg = str(excinfo.value)
    assert "15 C2" in msg and "DMRS" in msg and "MAX_PAYLOAD_SYMBOLS=128" in msg


# -- input validation -------------------------------------------------


def test_rejects_c2_longer_than_the_cap():
    ofdm = make()
    with pytest.raises(ValueError, match="c2_len_bytes"):
        ofdm.generate_frame(bits(1000), c2_bits=bits((C.C2_MAX_BYTES + 1) * 8))


def test_rejects_partial_bytes():
    """c2_len_bytes is a byte count on the wire, so a bit count that is
    not a whole number of bytes cannot be represented."""
    ofdm = make()
    with pytest.raises(ValueError, match="whole number of bytes"):
        ofdm.generate_frame(bits(1000), c2_bits=bits(12))


def test_rejects_batch_mismatch():
    ofdm = make()
    payload = np.zeros((3, 1000), "uint8")
    with pytest.raises(ValueError, match="batch"):
        ofdm.generate_frame(payload, c2_bits=np.zeros((2, 64), "uint8"))


def test_accepts_1d_c2_bits():
    ofdm = make()
    frame = ofdm.generate_frame(bits(1000), c2_bits=np.zeros(64 * 8, "uint8"))
    assert ofdm.rx_process(frame)["header"]["c2_len_bytes"] == 64


@pytest.mark.parametrize("n_batch", [1, 3])
def test_multi_batch_carries_a_c2_region_per_item(n_batch):
    ofdm = make()
    payload = np.random.default_rng(2).integers(0, 2, size=(n_batch, 20_000)).astype("uint8")
    c2 = np.random.default_rng(3).integers(0, 2, size=(n_batch, 72 * 8)).astype("uint8")
    frame = ofdm.generate_frame(payload, c2_bits=c2)
    n_c2 = C.n_c2_symbols(72, NDATA)
    expected = overhead(ofdm) + (n_c2 + n_main_symbols(ofdm, payload)) * SLOT
    assert frame.shape == (n_batch, expected)


# -- lazy construction ------------------------------------------------


def test_c2_codec_is_not_built_until_used():
    """Most frames carry no C2 region, and Packetizer construction is
    not free (native Viterbi trellis + RS GF tables)."""
    ofdm = make()
    assert ofdm._c2_packetizer is None and ofdm._c2_modem is None
    ofdm.generate_frame(bits(1000))
    assert ofdm._c2_packetizer is None, "a no-C2 frame must not build the C2 codec"
    ofdm.generate_frame(bits(1000), c2_bits=bits(64 * 8))
    assert ofdm._c2_packetizer is not None and ofdm._c2_modem is not None
