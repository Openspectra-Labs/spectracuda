"""C2 region receive-side decode (critical-region plan step 4).

The step's reason to exist is one property:

    main payload fails   =/=>   C2 fails

`test_c2_survives_a_destroyed_main_payload` and
`test_c2_survives_an_uncorrectable_main_payload` are the tests that
matter; everything else guards the boundary arithmetic that makes them
possible.

The boundary is DERIVED, never signalled: both sides compute
`n_c2_symbols(c2_len_bytes, n_data)` from the header's length field and
the fixed profile. If they ever disagree, both regions decode as
garbage -- which is why the 64QAM/256QAM cases below matter more than
the QPSK ones. With both regions at QPSK a boundary error is masked,
because the symbols either side of it carry the same bits per symbol.

See docs/2026-09-20-critical-c2-region-plan.md.
"""
import numpy as np
import pytest

from spectracuda.framing import c2 as C
from spectracuda.framing import dmrs as D
from spectracuda.pipeline import Ofdm

FFT, CP, NDATA, NPILOT = 256, 32, 216, 8
SLOT = FFT + CP
FEC_KW = dict(fec="rs_m8", fec1="conv_v27",
              interleaver="block", interleaver_kwargs={"unit_bits": 8})


def make(dmrs_interval=0, modem="qpsk", **kw):
    return Ofdm(fft_size=FFT, n_pilot=NPILOT, n_data=NDATA, cp_len=CP,
                modem=modem, crc="crc32", dmrs_interval=dmrs_interval,
                **{"fec": "none", **kw})


def bits(n, seed=0):
    return np.random.default_rng(seed).integers(0, 2, size=(1, n)).astype("uint8")


def overhead(ofdm):
    return FFT + SLOT * (ofdm.n_training_symbols + ofdm.num_symbols_header)


def n_main_symbols(ofdm, payload):
    return -(-ofdm.packetizer.encode(payload).shape[-1] // ofdm.bits_per_ofdm_symbol)


def wreck_main_region(ofdm, frame, payload, c2_len, interval, seed=99):
    """Replace every MAIN-payload slot with noise, leaving the C2 slots,
    the DMRS slots, the preamble, training and header untouched."""
    frame = np.asarray(frame).copy()
    n_c2 = C.n_c2_symbols(c2_len, NDATA)
    slot_map = D.dmrs_slot_map(n_c2 + n_main_symbols(ofdm, payload), interval)
    data_at = np.where(slot_map == D.DATA_SLOT)[0]
    base = overhead(ofdm)
    rng = np.random.default_rng(seed)
    for i in data_at[n_c2:]:
        frame[0, base + i * SLOT: base + (i + 1) * SLOT] = 0.5 * (
            rng.standard_normal(SLOT) + 1j * rng.standard_normal(SLOT)
        )
    return frame


# -- THE point of the feature -----------------------------------------


@pytest.mark.parametrize("interval", [0, 32])
@pytest.mark.parametrize("main_modem", ["qam64", "qam256"])
def test_c2_survives_a_destroyed_main_payload(interval, main_modem):
    """Every main-payload symbol replaced with noise. C2 must still
    deliver, bit-exact, with a valid CRC."""
    ofdm = make(interval, main_modem)
    payload, c2 = bits(20_000), bits(72 * 8, seed=1)
    frame = wreck_main_region(ofdm, ofdm.generate_frame(payload, c2_bits=c2),
                              payload, 72, interval)
    result = ofdm.rx_process(frame)
    assert not bool(np.asarray(result["crc_valid"])[0])
    assert bool(np.asarray(result["c2_crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(result["c2_bits"])[0], c2[0])


def test_c2_survives_an_uncorrectable_main_payload():
    """With FEC on the main payload, a destroyed region raises instead
    of failing CRC (Packetizer's documented behaviour). C2 is decoded
    BEFORE that, so it is still recoverable from the stash -- which is
    what rx_streaming hands up."""
    ofdm = make(32, "qam64", **FEC_KW)
    payload, c2 = bits(20_000), bits(72 * 8, seed=1)
    frame = wreck_main_region(ofdm, ofdm.generate_frame(payload, c2_bits=c2),
                              payload, 72, 32)
    with pytest.raises(ValueError):
        ofdm.rx_process(frame)
    stashed = ofdm._last_c2_result
    assert bool(np.asarray(stashed["c2_crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(stashed["c2_bits"])[0], c2[0])


def test_streaming_hands_up_c2_when_the_main_payload_gives_up():
    """The path the MAC actually uses. Returning None here would throw
    away control data that arrived intact."""
    ofdm = make(0, "qam64", **FEC_KW)
    payload, c2 = bits(20_000), bits(72 * 8, seed=1)
    frame = wreck_main_region(ofdm, ofdm.generate_frame(payload, c2_bits=c2),
                              payload, 72, 0)
    stream = np.concatenate(
        [np.zeros((1, 500), "complex64"), frame, np.zeros((1, 2000), "complex64")], axis=1)
    ofdm.reset_stream()
    out = [r for i in range(0, stream.shape[1], 2000)
           if (r := ofdm.rx_streaming(stream[:, i:i + 2000])) is not None]
    assert len(out) == 1
    assert out[0]["bits"] is None                       # main payload lost
    assert bool(np.asarray(out[0]["c2_crc_valid"])[0])  # C2 delivered
    np.testing.assert_array_equal(np.asarray(out[0]["c2_bits"])[0], c2[0])


# -- clean round trip -------------------------------------------------


def test_no_c2_reports_none_not_missing_keys():
    """The result dict's key set is fully enumerated, so the C2 fields
    are present and None rather than absent."""
    ofdm = make()
    result = ofdm.rx_process(ofdm.generate_frame(bits(20_000)))
    assert result["c2_bits"] is None
    assert result["c2_crc_valid"] is None
    assert result["c2_evm"] is None


def test_no_frame_found_still_has_the_c2_keys():
    ofdm = make()
    result = ofdm.rx_process(np.zeros((1, 40_000), "complex64"))
    for key in ("c2_bits", "c2_crc_valid", "c2_evm"):
        assert key in result and result[key] is None


@pytest.mark.parametrize("c2_len", [1, 16, 72, 128, 320])
@pytest.mark.parametrize("main_modem", ["qpsk", "qam64"])
def test_c2_round_trips(c2_len, main_modem):
    ofdm = make(0, main_modem)
    c2 = bits(c2_len * 8, seed=1)
    result = ofdm.rx_process(ofdm.generate_frame(bits(20_000), c2_bits=c2))
    assert bool(np.asarray(result["c2_crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(result["c2_bits"])[0], c2[0])


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_c2_round_trips_with_dmrs(interval):
    """DMRS slots are interleaved across both regions, so the boundary
    is not at a fixed slot offset."""
    ofdm = make(interval, "qam64")
    payload, c2 = bits(20_000), bits(72 * 8, seed=1)
    result = ofdm.rx_process(ofdm.generate_frame(payload, c2_bits=c2))
    assert bool(np.asarray(result["c2_crc_valid"])[0])
    assert bool(np.asarray(result["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(result["c2_bits"])[0], c2[0])
    np.testing.assert_array_equal(
        np.asarray(result["bits"])[0][: payload.shape[1]], payload[0])


def test_both_regions_round_trip_with_fec_and_dmrs():
    """Everything on at once."""
    ofdm = make(32, "qam16", **FEC_KW)
    payload, c2 = bits(10_000), bits(320 * 8, seed=1)
    result = ofdm.rx_process(ofdm.generate_frame(payload, c2_bits=c2))
    assert bool(np.asarray(result["crc_valid"])[0])
    assert bool(np.asarray(result["c2_crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(result["c2_bits"])[0], c2[0])
    np.testing.assert_array_equal(
        np.asarray(result["bits"])[0][: payload.shape[1]], payload[0])


def test_c2_only_frame_round_trips():
    ofdm = make()
    c2 = bits(72 * 8, seed=1)
    result = ofdm.rx_process(
        ofdm.generate_frame(np.zeros((1, 0), "uint8"), c2_bits=c2))
    assert bool(np.asarray(result["c2_crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(result["c2_bits"])[0], c2[0])


@pytest.mark.parametrize("n_batch", [2, 3])
def test_multi_batch_splits_per_item_not_per_row(n_batch):
    """Row order is batch-major/symbol-minor, so the region split is
    along axis 1 of the (n_batch, n_data_total, ...) view. A flat row
    slice would take item 0's whole frame as 'the C2 region' and still
    pass at n_batch=1."""
    ofdm = make(0, "qam64")
    payload = np.random.default_rng(2).integers(0, 2, size=(n_batch, 20_000)).astype("uint8")
    c2 = np.random.default_rng(3).integers(0, 2, size=(n_batch, 72 * 8)).astype("uint8")
    assert not np.array_equal(c2[0], c2[1])
    result = ofdm.rx_process(ofdm.generate_frame(payload, c2_bits=c2))
    assert np.asarray(result["c2_crc_valid"]).all()
    for i in range(n_batch):
        np.testing.assert_array_equal(np.asarray(result["c2_bits"])[i], c2[i])


# -- the receiver is not told the profile -----------------------------


def test_receiver_resolves_everything_from_the_header():
    """A receiver built with no C2 and no DMRS decodes a frame carrying
    both -- the C2 profile is fixed, and its length comes off the wire."""
    tx = make(32, "qam64")
    rx = make(0, "qpsk")
    payload, c2 = bits(20_000), bits(72 * 8, seed=1)
    result = rx.rx_process(tx.generate_frame(payload, c2_bits=c2))
    assert result["header"]["c2_len_bytes"] == 72
    assert bool(np.asarray(result["c2_crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(result["c2_bits"])[0], c2[0])


# -- EVM is reported per region ---------------------------------------


def test_evm_is_reported_separately_for_each_region():
    """`evm` keeps meaning 'EVM at the advertised MCS' -- mixing the
    QPSK C2 symbols into it would bias any adaptive-MCS controller
    reading it."""
    ofdm = make(0, "qam64")
    result = ofdm.rx_process(
        ofdm.generate_frame(bits(20_000), c2_bits=bits(72 * 8, seed=1)))
    assert result["c2_evm"] is not None
    assert float(np.asarray(result["evm"])[0]) >= 0
    assert float(np.asarray(result["c2_evm"])[0]) >= 0
