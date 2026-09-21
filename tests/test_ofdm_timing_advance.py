"""`timing_advance`: deliberately place every OFDM FFT window a couple of
samples early, inside the cyclic prefix.

The asymmetry this exists for: a window that starts inside the CP is a
true cyclic shift of the correct one, so it costs only a phase ramp
across subcarriers -- and the channel estimate, measured through that
same window, absorbs the ramp exactly. A window that starts LATE is not a
cyclic shift: it runs past the symbol's last sample into the next symbol,
and that leakage depends on the neighbouring symbol's data, so no
per-subcarrier H[k] can represent it. Early is free; late is not.

That matters because Schmidl-Cox's timing metric here has a 2-sample flat
top (the preamble carries no CP, so there is no broad plateau), and a
multipath echo biases the contest toward the LATE candidate -- and does
so MORE reliably as SNR rises, because a cleaner metric resolves a peak
the echo has genuinely shifted.

See docs/2026-09-21-dmrs-differential-doppler-characterization.md and
examples/dmrs_doppler_study.py --part sync.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel

FFT, CP, NDATA, NPILOT = 256, 64, 216, 8
FS = 10e6
TAIL = 2048


def make(timing_advance=None, cp_len=CP, **kw):
    return Ofdm(
        fft_size=FFT, n_pilot=NPILOT, n_data=NDATA, cp_len=cp_len,
        modem="qam16", fec="none", crc="crc32",
        sync="schmidl_cox", cfo="schmidl_cox", n_training_symbols=2,
        timing_advance=timing_advance, **kw
    )


def two_ray(tx, a, snr_db, seed):
    """Static two-ray: LOS plus a one-sample echo, no Doppler -- this is
    purely about where the FFT window lands. Built from the shared
    `sim.Channel` rather than hand-rolled, so the trailing-capture
    discipline (see that class's docstring) is not re-derived here."""
    return Channel(
        snr_db=snr_db,
        multipath_taps=np.array([1.0, a], dtype="complex64"),
        tail_samples=TAIL, seed=seed, backend="numpy",
    ).process(tx)


def bits_for(ofdm, n_symbols=40, seed=1):
    n_bits = min(n_symbols, ofdm.max_data_symbols) * ofdm.bits_per_ofdm_symbol - 32
    return np.random.default_rng(seed).integers(0, 2, size=(1, n_bits)).astype("uint8")


def run(ofdm, a, snr_db, seed):
    """(crc_valid, mean EVM). Deliberately does NOT assert CRC -- the
    whole point is that the un-advanced window loses frames an advanced
    one keeps, so the failing case has to stay measurable."""
    ofdm.debug_payload_symbols = True
    bits = bits_for(ofdm)
    r = ofdm.rx_process(two_ray(ofdm.generate_frame(bits), a, snr_db, seed))
    return bool(np.asarray(r["crc_valid"])[0]), float(np.asarray(r["evm"])[0])


# -- construction ------------------------------------------------------


def test_default_is_two_samples():
    assert make().timing_advance == 2


def test_default_is_clamped_by_a_small_cp():
    """cp_len=0 is legal (see hls/gen/emit.py's note). Without a CP there
    is no early direction to move in, so 'auto' must resolve to 0 rather
    than pushing the window outside the symbol."""
    assert make(cp_len=0).timing_advance == 0
    assert make(cp_len=1).timing_advance == 1


def test_explicit_zero_restores_the_pre_fix_behavior():
    assert make(0).timing_advance == 0


@pytest.mark.parametrize("bad", [-1, CP + 1, 999])
def test_advance_outside_the_cp_is_rejected(bad):
    """The advance shares the CP with the channel's delay spread, so it
    cannot exceed cp_len -- a window pushed further back leaves the
    symbol entirely."""
    with pytest.raises(ValueError, match="timing_advance"):
        make(bad)


# -- the reason it exists ----------------------------------------------


def test_advance_does_no_harm_without_an_echo():
    """Single-path channel: sync already lands correctly, so the advance
    must change nothing measurable. Early placement is free -- that is
    the whole premise."""
    off_crc, off = run(make(0), a=0.0, snr_db=25.0, seed=0)
    on_crc, on = run(make(2), a=0.0, snr_db=25.0, seed=0)
    assert off_crc and on_crc
    assert on == pytest.approx(off, abs=5e-3)


@pytest.mark.parametrize("a,snr_db", [(0.2, 25.0), (0.3, 30.0), (0.3, 40.0), (0.4, 40.0)])
def test_advance_recovers_frames_an_echo_would_otherwise_lose(a, snr_db):
    """The gate, and it is not merely a margin argument: with an echo
    strong enough to bias sync late, uncoded 16QAM LOSES the frame
    outright, and the advanced window recovers it. EVM roughly 0.17 ->
    0.01-0.07 across these cases.

    Note the un-advanced EVM barely improves with SNR (0.181 at 25 dB,
    0.172 at 40 dB): the error is inter-symbol leakage, not noise, so
    more SNR does not dilute it -- it makes the mis-placement more
    certain by sharpening the metric the echo has already shifted."""
    off_crc, off = run(make(0), a=a, snr_db=snr_db, seed=0)
    on_crc, on = run(make(2), a=a, snr_db=snr_db, seed=0)
    assert not off_crc, "expected the un-advanced window to lose this frame"
    assert on_crc, "advanced window must recover it"
    assert on < 0.5 * off


def test_advanced_window_still_round_trips_bit_exactly():
    """Guards the obvious way to get this wrong: shifting `pos` without
    shifting it for EVERY window. Training, header and payload all walk
    forward from one `pos`, so they move together or the header decodes
    against a channel estimate taken somewhere else."""
    ofdm = make(2)
    bits = bits_for(ofdm)
    r = ofdm.rx_process(two_ray(ofdm.generate_frame(bits), 0.3, 30.0, 0))
    assert r["frame_found"]
    assert bool(np.asarray(r["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(r["bits"])[0][: bits.shape[1]], bits[0])


def test_advance_does_not_move_the_reported_start_index():
    """The advance is applied at `pos` (the OFDM window origin), NOT to
    start_index -- the CFO estimator keeps correlating at the preamble
    position sync actually detected. A caller reading start_index must
    see the detection, not the internal window placement."""
    bits = bits_for(make(0))
    rx = two_ray(make(0).generate_frame(bits), 0.3, 30.0, 0)
    a0 = int(np.asarray(make(0).rx_process(rx)["start_index"]).ravel()[0])
    a2 = int(np.asarray(make(2).rx_process(rx)["start_index"]).ravel()[0])
    assert a0 == a2
