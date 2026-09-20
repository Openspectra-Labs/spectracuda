"""DMRS receive-side channel refresh (plan step 3): each payload symbol
is equalized against the estimate from the most recent reference symbol
before it, instead of the single training-symbol estimate reused for the
whole frame.

The three gates from the plan, in order:

1. `dmrs_interval=0` is unchanged from before the feature existed.
2. On a STATIC channel, DMRS on decodes identically to DMRS off -- a
   refreshed estimate of an unchanging channel must agree with the
   original.
3. On a channel whose H[k] genuinely CHANGES during the frame, EVM late
   in the frame improves measurably. This is the gate that says whether
   the feature works at all; the other two only say it does no harm.

Gate 3 needs a channel the existing sim cannot produce. `sim/channel.py`
offers static multipath, AWGN and a CONSTANT CFO -- but a constant CFO
is a pure phase rotation, which the existing per-symbol CPE correction
already removes, so it would show nothing. `time_varying_two_ray()`
below rotates the second tap instead, making H[k] both
frequency-selective and time-varying, which no single phase correction
can undo.

See docs/2026-09-20-dmrs-periodic-channel-refresh-plan.md.
"""
import numpy as np
import pytest

from spectracuda.pipeline import Ofdm

FFT, CP, NDATA, NPILOT = 256, 32, 216, 8
FS = 10e6


def make(dmrs_interval=0, **kw):
    return Ofdm(
        fft_size=FFT, n_pilot=NPILOT, n_data=NDATA, cp_len=CP,
        modem="qpsk", fec="none", crc="crc32",
        dmrs_interval=dmrs_interval, **kw
    )


def bits_for(ofdm, n_symbols, seed=1, n_batch=1):
    n_symbols = min(n_symbols, ofdm.max_data_symbols)
    n_bits = n_symbols * ofdm.bits_per_ofdm_symbol - 32  # room for crc32
    return np.random.default_rng(seed).integers(0, 2, size=(n_batch, n_bits)).astype("uint8")


#: Noise is drawn at this fixed length and then sliced -- see
#: time_varying_two_ray() for why that matters.
_NOISE_LEN = 80_000
#: Trailing capture samples appended to every simulated frame -- see below.
_TAIL = 1024


def time_varying_two_ray(tx, a=0.5, fd=100.0, fs=FS, snr_db=30.0, seed=0):
    """rx[n] = tx[n] + a*exp(j*2*pi*fd*n/fs)*tx[n-1].

    H[k,t] = 1 + a*e^{j2*pi*fd*t}*e^{-j2*pi*k/N}: the notch moves across
    the band as the frame runs, so the channel estimate taken at the
    start of the frame goes progressively stale. Exactly the impairment
    DMRS exists to fix, and exactly the one a phase-only correction
    cannot.

    Two details that are methodology, not physics, and that silently
    corrupted an earlier round of measurements with this helper (see
    docs/2026-09-20-dmrs-static-channel-cost.md):

    1. **Trailing samples.** Multipath can move the detected frame start
       a sample late. A frame that ends flush with the array then needs
       one sample past the end and the bounds check raises -- which
       looks like a failed packet but is a simulation boundary, not a
       decode failure. A real capture always has samples after the
       frame, so the tail is added here to match.
    2. **Length-independent noise.** Drawing `standard_normal(n)` twice
       makes the IMAGINARY part depend on `n`: the second draw starts
       wherever the first ended. DMRS changes the frame length, so the
       same seed would otherwise hand each interval different preamble
       noise -- different sync behaviour, and an apparent
       interval-dependent PER that is pure artifact. Drawing at a fixed
       length and slicing keeps every interval on identical noise.
    """
    tx = np.asarray(tx)
    tx = np.concatenate([tx, np.zeros((tx.shape[0], _TAIL), tx.dtype)], axis=1)
    n = np.arange(tx.shape[-1])
    delayed = np.concatenate([np.zeros((tx.shape[0], 1), tx.dtype), tx[:, :-1]], axis=1)
    rx = tx + a * np.exp(1j * 2 * np.pi * fd * n / fs)[None, :] * delayed

    rng = np.random.default_rng(seed)
    assert rx.shape[-1] <= _NOISE_LEN, "raise _NOISE_LEN for this frame size"
    noise = (rng.standard_normal(_NOISE_LEN) + 1j * rng.standard_normal(_NOISE_LEN))[: rx.shape[-1]]
    noise_power = np.mean(np.abs(rx) ** 2) / (10 ** (snr_db / 10))
    rx = rx + np.sqrt(noise_power / 2) * noise[None, :]
    return rx.astype("complex64")


def evm_per_symbol(result, n_batch=1):
    ev = np.asarray(result["symbol_diagnostics"]["evm_per_symbol"]).ravel()
    return ev.reshape(n_batch, -1)


# -- gate 1: DMRS off is unchanged ------------------------------------


def test_interval_off_round_trips_cleanly():
    ofdm = make(0)
    bits = bits_for(ofdm, 93)
    result = ofdm.rx_process(ofdm.generate_frame(bits))
    assert result["frame_found"]
    assert bool(np.asarray(result["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(result["bits"])[0][: bits.shape[1]], bits[0])


# -- gate 2: static channel, DMRS changes nothing ---------------------


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_static_channel_decodes_identically_with_dmrs_on(interval):
    """A refreshed estimate of an UNCHANGING channel must agree with the
    original, so enabling DMRS must not perturb the decode at all."""
    ofdm = make(interval)
    bits = bits_for(ofdm, 93)
    result = ofdm.rx_process(ofdm.generate_frame(bits))
    assert result["frame_found"]
    assert bool(np.asarray(result["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(result["bits"])[0][: bits.shape[1]], bits[0])


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_static_channel_evm_matches_dmrs_off(interval):
    off = make(0)
    on = make(interval)
    bits = bits_for(on, 93)
    evm_off = float(np.asarray(off.rx_process(off.generate_frame(bits))["evm"])[0])
    evm_on = float(np.asarray(on.rx_process(on.generate_frame(bits))["evm"])[0])
    assert evm_on == pytest.approx(evm_off, abs=1e-3)


@pytest.mark.parametrize("interval", [16, 32])
def test_static_multipath_round_trips(interval):
    """A STATIC but frequency-selective channel: DMRS must neither help
    nor hurt, because the refreshed estimate measures the same H[k].

    Trailing zeros because multipath shifts the sync peak by a sample or
    so, and a frame that ends flush with the buffer then runs one index
    past the end -- pre-existing behavior, reproducible at interval=0,
    and not what this test is about."""
    from spectracuda.sim import Channel

    taps = np.array([1.0, 0.3], dtype="complex64")
    evms = {}
    for iv in (0, interval):
        ofdm = make(iv)
        bits = bits_for(ofdm, 93)
        channel = Channel(snr_db=30.0, multipath_taps=taps, seed=3, backend="numpy")
        rx = np.concatenate(
            [np.asarray(channel.process(ofdm.generate_frame(bits))),
             np.zeros((1, 1024), "complex64")], axis=1
        )
        result = ofdm.rx_process(rx)
        assert bool(np.asarray(result["crc_valid"])[0])
        np.testing.assert_array_equal(np.asarray(result["bits"])[0][: bits.shape[1]], bits[0])
        evms[iv] = float(np.asarray(result["evm"])[0])
    assert evms[interval] == pytest.approx(evms[0], abs=5e-3)


# -- gate 3: the feature actually works -------------------------------


def test_time_varying_channel_evm_no_longer_grows_across_the_frame():
    """The core claim. With one estimate for the whole frame, EVM climbs
    monotonically as that estimate goes stale; with DMRS it stays flat."""
    off, on = make(0), make(32)
    for o in (off, on):
        o.debug_payload_symbols = True

    bits = bits_for(on, 125)
    ev_off = evm_per_symbol(off.rx_process(time_varying_two_ray(off.generate_frame(bits))))[0]
    ev_on = evm_per_symbol(on.rx_process(time_varying_two_ray(on.generate_frame(bits))))[0]

    # DMRS off: the last segment is dramatically worse than the first.
    assert ev_off[-32:].mean() > 3 * ev_off[:32].mean()
    # DMRS on: the last segment is no worse than the first, to within noise.
    assert ev_on[-32:].mean() < 1.5 * ev_on[:32].mean()
    # And late-frame EVM is far better than it was without DMRS.
    assert ev_on[-32:].mean() < 0.5 * ev_off[-32:].mean()


def test_time_varying_channel_first_segment_is_unaffected():
    """Segment 0 uses the training-symbol estimate whether DMRS is on or
    off, so it must be essentially identical in both -- if it moved, the
    refresh would be perturbing data it has no business touching."""
    off, on = make(0), make(32)
    for o in (off, on):
        o.debug_payload_symbols = True
    bits = bits_for(on, 125)
    ev_off = evm_per_symbol(off.rx_process(time_varying_two_ray(off.generate_frame(bits))))[0]
    ev_on = evm_per_symbol(on.rx_process(time_varying_two_ray(on.generate_frame(bits))))[0]
    assert ev_on[:32].mean() == pytest.approx(ev_off[:32].mean(), rel=0.1)


def test_time_varying_channel_recovers_a_frame_that_otherwise_fails_crc():
    """End to end: the same frame over the same channel fails without
    DMRS and passes with it."""
    off, on = make(0), make(32)
    bits = bits_for(on, 125)
    r_off = off.rx_process(time_varying_two_ray(off.generate_frame(bits)))
    r_on = on.rx_process(time_varying_two_ray(on.generate_frame(bits)))
    assert not bool(np.asarray(r_off["crc_valid"])[0])
    assert bool(np.asarray(r_on["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(r_on["bits"])[0][: bits.shape[1]], bits[0])


def test_denser_dmrs_tracks_better():
    """Monotonicity: more refreshes should not track worse."""
    evms = {}
    for interval in (0, 64, 32, 16):
        o = make(interval)
        bits = bits_for(o, 121)  # fits every interval's ceiling
        evms[interval] = float(np.asarray(
            o.rx_process(time_varying_two_ray(o.generate_frame(bits)))["evm"]
        )[0])
    assert evms[64] < evms[0]
    assert evms[32] < evms[64]
    assert evms[16] < evms[32]


# -- the per-segment plumbing itself ----------------------------------


def test_segments_really_use_different_channel_estimates():
    """Guards against the failure mode this step is most exposed to: the
    plumbing looking right while every symbol still gets one estimate.
    If all segments shared an estimate, a time-varying channel would
    produce the SAME growing-EVM profile as DMRS off."""
    on = make(32)
    on.debug_payload_symbols = True
    bits = bits_for(on, 125)
    ev = evm_per_symbol(on.rx_process(time_varying_two_ray(on.generate_frame(bits))))[0]
    seg_means = [ev[0:32].mean(), ev[32:64].mean(), ev[64:96].mean(), ev[96:].mean()]
    # Flat, not monotonically climbing: max/min well under the ~4x the
    # single-estimate case shows over the same frame.
    assert max(seg_means) / min(seg_means) < 1.5


@pytest.mark.parametrize("interval", [16, 32, 64])
def test_multi_batch_keeps_each_item_aligned_with_its_own_estimates(interval):
    """The expansion is batch-major/symbol-minor. A transposed reshape
    would still pass every n_batch=1 test above while silently pairing
    item 0's symbols with item 1's channel -- so this uses DIFFERENT
    payloads per item and requires each to decode to its own bits."""
    ofdm = make(interval)
    bits = bits_for(ofdm, 93, n_batch=3, seed=5)
    assert not np.array_equal(bits[0], bits[1])
    result = ofdm.rx_process(ofdm.generate_frame(bits))
    crc = np.asarray(result["crc_valid"])
    assert crc.shape == (3,) and crc.all()
    for i in range(3):
        np.testing.assert_array_equal(np.asarray(result["bits"])[i][: bits.shape[1]], bits[i])


# -- RX-side total-slot guard -----------------------------------------


def test_override_guard_counts_dmrs_toward_the_limit():
    """126 data symbols is under MAX_PAYLOAD_SYMBOLS on its own, but at
    interval 32 it needs 3 DMRS for 129 slots -- over budget. Rejecting
    it is the whole point of the total-slot rule."""
    ofdm = make(32)
    bits = bits_for(ofdm, 93)
    frame = ofdm.generate_frame(bits)
    with pytest.raises(ValueError) as excinfo:
        ofdm.rx_process(frame, n_payload_symbols=126)
    msg = str(excinfo.value)
    assert "126" in msg and "3 DMRS" in msg and "129" in msg
    assert "MAX_PAYLOAD_SYMBOLS=128" in msg and "ceiling is 125" in msg


def test_override_at_the_ceiling_is_accepted():
    """125 + 3 = 128 is exactly at the limit and must NOT be rejected."""
    ofdm = make(32)
    bits = bits_for(ofdm, 93)
    frame = np.concatenate(
        [np.asarray(ofdm.generate_frame(bits)), np.zeros((1, 128 * 288), "complex64")], axis=1
    )
    ofdm.rx_process(frame, n_payload_symbols=125)  # must not raise
