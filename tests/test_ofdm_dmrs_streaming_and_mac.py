"""DMRS in the streaming receiver (plan step 6) and in MAC segment
sizing (plan step 5).

Both are places that computed a frame's size from `n_payload_symbols`
alone. That was correct until DMRS started interleaving extra slots
among the data symbols, and both fail in ways a short-frame test would
never reach:

- `rx_streaming` declares a frame complete at
  `pos + n_payload_symbols * slot_len`, so it would hand the payload
  decoder a truncated buffer and evict the tail of every DMRS-bearing
  frame. It is the only frame-length computation outside
  `_decode_payload_from_header`.
- `compute_max_segment_bits` sizes against all 128 slots being data, so
  the MAC would oversize a segment and `generate_frame()` would raise --
  at maximum segment size, on real traffic.

See docs/2026-09-20-dmrs-periodic-channel-refresh-plan.md.
"""
import numpy as np
import pytest

from spectracuda.framing import dmrs as D
from spectracuda.mac import Mac
from spectracuda.pipeline import Ofdm

INTERVALS = [0, 16, 32, 64]
OFDM_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    modem="qpsk", fec="none", crc="crc32",
)


def make(dmrs_interval=0):
    return Ofdm(dmrs_interval=dmrs_interval, **OFDM_KWARGS)


def payload(n_bits=40_000, seed=0):
    return np.random.default_rng(seed).integers(0, 2, size=(1, n_bits)).astype("uint8")


def drain(rx, stream, chunk=2000):
    """Feed a stream through rx_streaming() in chunks, collecting every
    completed frame."""
    rx.reset_stream()
    frames = []
    for i in range(0, stream.shape[1], chunk):
        result = rx.rx_streaming(stream[:, i: i + chunk])
        if result is not None and result.get("bits") is not None:
            frames.append(result)
    return frames


def wrap(frame, lead=500, tail=500):
    frame = np.asarray(frame)
    return np.concatenate(
        [np.zeros((1, lead), "complex64"), frame, np.zeros((1, tail), "complex64")], axis=1
    )


# -- step 6: streaming ------------------------------------------------


@pytest.mark.parametrize("interval", INTERVALS)
def test_streaming_decodes_a_dmrs_frame(interval):
    """Without the total-slot fix, the receiver calls the frame complete
    before its last slots have arrived and the decode fails."""
    tx, rx = make(interval), make(0)
    bits = payload()
    frames = drain(rx, wrap(tx.generate_frame(bits)))
    assert len(frames) == 1
    assert bool(np.asarray(frames[0]["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(frames[0]["bits"])[0][: bits.shape[1]], bits[0])
    assert frames[0]["header"]["dmrs_interval"] == interval


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_streaming_evicts_exactly_one_frame_from_the_buffer(interval):
    """frame_end also drives buffer eviction, so an undercount leaves
    the tail of frame 1 in the buffer and misaligns frame 2. Two
    back-to-back frames with DIFFERENT payloads catch that; a
    single-frame test never would."""
    tx, rx = make(interval), make(0)
    bits_a, bits_b = payload(40_000, seed=1), payload(40_000, seed=2)
    assert not np.array_equal(bits_a, bits_b)

    stream = np.concatenate([
        np.zeros((1, 500), "complex64"),
        np.asarray(tx.generate_frame(bits_a)),
        np.asarray(tx.generate_frame(bits_b)),
        np.zeros((1, 500), "complex64"),
    ], axis=1)

    frames = drain(rx, stream)
    assert len(frames) == 2
    for frame, expected in zip(frames, (bits_a, bits_b)):
        assert bool(np.asarray(frame["crc_valid"])[0])
        np.testing.assert_array_equal(
            np.asarray(frame["bits"])[0][: expected.shape[1]], expected[0]
        )


@pytest.mark.parametrize("chunk", [500, 2000, 100_000])
def test_streaming_is_insensitive_to_chunk_size(chunk):
    """A DMRS frame must reassemble whether its slots arrive a few at a
    time or all at once."""
    tx, rx = make(32), make(0)
    bits = payload()
    frames = drain(rx, wrap(tx.generate_frame(bits)), chunk=chunk)
    assert len(frames) == 1
    assert bool(np.asarray(frames[0]["crc_valid"])[0])


def test_streaming_tracks_a_time_varying_channel():
    """The feature's actual purpose, through the streaming path."""
    from test_ofdm_dmrs_rx import time_varying_two_ray

    bits = payload(125 * make(32).bits_per_ofdm_symbol - 32)
    rx = make(0)

    on = drain(rx, wrap(time_varying_two_ray(make(32).generate_frame(bits))))
    assert len(on) == 1 and bool(np.asarray(on[0]["crc_valid"])[0])

    off = drain(rx, wrap(time_varying_two_ray(make(0).generate_frame(bits))))
    assert not (off and bool(np.asarray(off[0]["crc_valid"])[0]))


# -- step 5: MAC capacity ---------------------------------------------


@pytest.mark.parametrize("interval,expected_data,expected_dmrs", [
    (0, 128, 0), (64, 127, 1), (32, 125, 3), (16, 121, 7),
])
def test_max_segment_fills_exactly_the_slot_budget(interval, expected_data, expected_dmrs):
    """The MAC's largest segment must land on exactly 128 TOTAL slots --
    not 128 data symbols plus DMRS on top."""
    mac = Mac("um", ofdm_kwargs=dict(OFDM_KWARGS, dmrs_interval=interval))
    ofdm = mac.ofdm
    encoded = ofdm.packetizer.encoded_length(mac.max_segment_bits)
    n_data = -(-encoded // ofdm.bits_per_ofdm_symbol)
    assert n_data == expected_data
    assert D.n_dmrs_symbols(n_data, interval) == expected_dmrs
    assert D.total_slots(n_data, interval) == Ofdm.MAX_PAYLOAD_SYMBOLS


@pytest.mark.parametrize("interval", INTERVALS)
def test_a_max_size_segment_actually_transmits(interval):
    """The failure this guards against is the MAC handing
    generate_frame() a payload only it thinks fits."""
    mac = Mac("um", ofdm_kwargs=dict(OFDM_KWARGS, dmrs_interval=interval))
    mac.ofdm.generate_frame(np.zeros((1, mac.max_segment_bits), "uint8"))  # must not raise


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_dmrs_shrinks_the_segment_budget(interval):
    """DMRS costs payload capacity -- if the budget were unchanged, the
    MAC would still be sizing against the raw constant."""
    off = Mac("um", ofdm_kwargs=dict(OFDM_KWARGS, dmrs_interval=0))
    on = Mac("um", ofdm_kwargs=dict(OFDM_KWARGS, dmrs_interval=interval))
    assert on.max_segment_bits < off.max_segment_bits


def test_denser_dmrs_costs_more_capacity():
    budgets = {
        iv: Mac("um", ofdm_kwargs=dict(OFDM_KWARGS, dmrs_interval=iv)).max_segment_bits
        for iv in INTERVALS
    }
    assert budgets[0] > budgets[64] > budgets[32] > budgets[16]
