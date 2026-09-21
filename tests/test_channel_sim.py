import numpy as np
import pytest

from spectracuda.sim import Channel


def test_no_impairments_is_identity():
    channel = Channel(backend="numpy")
    x = (np.random.default_rng(0).standard_normal((2, 50)) + 1j * 0).astype("complex64")
    y = channel.process(x)
    np.testing.assert_allclose(y, x, atol=1e-6)


def test_awgn_adds_noise_at_requested_snr():
    channel = Channel(snr_db=20.0, seed=0, backend="numpy")
    x = np.ones((1, 10000), dtype="complex64")
    y = channel.process(x)
    noise = y - x
    measured_snr_db = 10 * np.log10(np.mean(np.abs(x) ** 2) / np.mean(np.abs(noise) ** 2))
    assert measured_snr_db == pytest.approx(20.0, abs=1.0)


def test_multipath_output_shape_matches_input():
    taps = Channel.random_multipath_taps(3, seed=1)
    channel = Channel(multipath_taps=taps, backend="numpy")
    x = np.ones((2, 20), dtype="complex64")
    y = channel.process(x)
    assert y.shape == x.shape


def test_random_multipath_taps_unit_energy():
    taps = Channel.random_multipath_taps(5, seed=2)
    assert taps.shape == (5,)
    np.testing.assert_allclose(np.sum(np.abs(taps) ** 2), 1.0, atol=1e-5)


def test_cfo_rotates_phase_as_expected():
    fft_size = 64
    eps = 0.25
    channel = Channel(cfo=eps, cfo_fft_size=fft_size, backend="numpy")
    x = np.ones((1, fft_size), dtype="complex64")
    y = channel.process(x)
    n = np.arange(fft_size)
    expected = np.exp(1j * 2 * np.pi * eps * n / fft_size)
    np.testing.assert_allclose(y[0], expected, atol=1e-6)


def test_cfo_without_fft_size_raises():
    with pytest.raises(ValueError):
        Channel(cfo=0.1, backend="numpy")


def test_1d_input_is_promoted_to_batch_of_one():
    channel = Channel(snr_db=30.0, seed=0, backend="numpy")
    x = np.ones(16, dtype="complex64")
    y = channel.process(x)
    assert y.shape == (1, 16)


def test_impairments_compose_awgn_multipath_cfo():
    taps = Channel.random_multipath_taps(3, seed=3)
    channel = Channel(snr_db=30.0, multipath_taps=taps, cfo=0.1, cfo_fft_size=32, seed=3, backend="numpy")
    x = np.ones((2, 32), dtype="complex64")
    y = channel.process(x)
    assert y.shape == x.shape
    assert not np.allclose(y, x)  # something actually happened


# -- time-varying multipath (tap_doppler_hz) ---------------------------


def _ref(x, taps, dopplers, fs):
    """Independent reference: y[n] = sum_k taps[k]*e^{j2pi f_k n/fs}*x[n-k],
    written out directly rather than reusing the implementation."""
    x = np.asarray(x)
    n = np.arange(x.shape[-1])
    y = np.zeros_like(x, dtype="complex128")
    for k, (h, f) in enumerate(zip(taps, dopplers)):
        shifted = np.concatenate([np.zeros((x.shape[0], k), x.dtype), x[:, :x.shape[-1]-k]], axis=-1) if k else x
        y += h * np.exp(1j*2*np.pi*f*n/fs)[None, :] * shifted
    return y


def test_zero_tap_doppler_matches_the_static_multipath_path():
    """The two branches must agree exactly when nothing is moving --
    otherwise every existing static-multipath result silently shifts the
    moment a caller adds tap_doppler_hz=[0, 0]."""
    taps = np.array([1.0, 0.2], dtype="complex64")
    x = (np.random.default_rng(0).standard_normal((2, 2000))
         + 1j * np.random.default_rng(1).standard_normal((2, 2000))).astype("complex64")
    static = Channel(multipath_taps=taps, backend="numpy").process(x)
    moving = Channel(multipath_taps=taps, tap_doppler_hz=[0.0, 0.0],
                     sample_rate_hz=10e6, backend="numpy").process(x)
    np.testing.assert_allclose(np.asarray(moving), np.asarray(static), atol=0, rtol=0)


def test_tap_doppler_matches_the_closed_form():
    taps = np.array([1.0, 0.2], dtype="complex64")
    dop = [1600.0, 1900.0]
    fs = 20e6
    x = (np.random.default_rng(2).standard_normal((1, 3000))
         + 1j * np.random.default_rng(3).standard_normal((1, 3000))).astype("complex64")
    got = Channel(multipath_taps=taps, tap_doppler_hz=dop, sample_rate_hz=fs,
                  backend="numpy").process(x)
    np.testing.assert_allclose(np.asarray(got), _ref(x, taps, dop, fs), atol=1e-5)


def test_equal_tap_doppler_leaves_the_channel_shape_static():
    """The point of the parameter. Equal shifts on every tap are a COMMON
    Doppler -- one rotating phase on the whole signal, which CFO/CPE
    remove -- so dividing it out must recover the static channel. Only a
    DIFFERENCE between taps makes H[k] itself time-varying."""
    taps = np.array([1.0, 0.2], dtype="complex64")
    fs, f = 20e6, 1600.0
    x = (np.random.default_rng(4).standard_normal((1, 3000))
         + 1j * np.random.default_rng(5).standard_normal((1, 3000))).astype("complex64")
    common = np.asarray(Channel(multipath_taps=taps, tap_doppler_hz=[f, f],
                                sample_rate_hz=fs, backend="numpy").process(x))
    static = np.asarray(Channel(multipath_taps=taps, backend="numpy").process(x))
    derotated = common * np.exp(-1j*2*np.pi*f*np.arange(x.shape[-1])/fs)[None, :]
    np.testing.assert_allclose(derotated, static, atol=1e-5)


def test_differential_tap_doppler_does_make_it_time_varying():
    """Complement of the test above: with unequal tap shifts, no single
    de-rotation can recover a static channel."""
    taps = np.array([1.0, 0.2], dtype="complex64")
    fs = 20e6
    x = (np.random.default_rng(6).standard_normal((1, 3000))
         + 1j * np.random.default_rng(7).standard_normal((1, 3000))).astype("complex64")
    diff = np.asarray(Channel(multipath_taps=taps, tap_doppler_hz=[1600.0, 1900.0],
                              sample_rate_hz=fs, backend="numpy").process(x))
    static = np.asarray(Channel(multipath_taps=taps, backend="numpy").process(x))
    for f in (1600.0, 1750.0, 1900.0):
        derotated = diff * np.exp(-1j*2*np.pi*f*np.arange(x.shape[-1])/fs)[None, :]
        assert not np.allclose(derotated, static, atol=1e-3)


def test_tap_doppler_requires_taps_and_a_sample_rate():
    with pytest.raises(ValueError, match="needs multipath_taps"):
        Channel(tap_doppler_hz=[0.0], backend="numpy")
    with pytest.raises(ValueError, match="sample_rate_hz is required"):
        Channel(multipath_taps=np.array([1.0], "complex64"),
                tap_doppler_hz=[0.0], backend="numpy")


def test_tap_doppler_length_must_match_taps():
    with pytest.raises(ValueError, match="one .*shift per tap"):
        Channel(multipath_taps=np.array([1.0, 0.2], "complex64"),
                tap_doppler_hz=[0.0], sample_rate_hz=10e6, backend="numpy")


# -- methodology knobs -------------------------------------------------


def test_tail_samples_extends_the_output_and_carries_the_multipath_tail():
    taps = np.array([1.0, 0.5], dtype="complex64")
    x = np.ones((1, 100), dtype="complex64")
    y = np.asarray(Channel(multipath_taps=taps, tail_samples=32,
                           backend="numpy").process(x))
    assert y.shape == (1, 132)
    # the echo of the last real sample lands in the tail, not nowhere
    assert abs(y[0, 100]) == pytest.approx(0.5, abs=1e-5)


def test_noise_draw_len_keeps_the_same_seed_sample_identical_across_lengths():
    """The artifact this exists for: without it, changing the frame length
    changes the noise REALIZATION, so a DMRS interval or payload size
    change looks like a channel effect. Compared on the signal-free case
    so the SNR-scaling term (which legitimately tracks input power) does
    not enter."""
    x = np.zeros((1, 4000), dtype="complex64")
    x[:] = 1.0
    kw = dict(snr_db=20.0, seed=7, backend="numpy")
    long_ = np.asarray(Channel(noise_draw_len=8000, **kw).process(x))
    short = np.asarray(Channel(noise_draw_len=8000, **kw).process(x[:, :2500]))
    np.testing.assert_allclose(long_[:, :2500], short, atol=1e-6)

    loose_long = np.asarray(Channel(**kw).process(x))
    loose_short = np.asarray(Channel(**kw).process(x[:, :2500]))
    assert not np.allclose(loose_long[:, :2500], loose_short, atol=1e-3)


def test_noise_draw_len_shorter_than_the_input_raises():
    with pytest.raises(ValueError, match="shorter than"):
        Channel(snr_db=20.0, noise_draw_len=10, backend="numpy").process(
            np.ones((1, 100), dtype="complex64")
        )


# -- sparse path specification (paths_to_taps) -------------------------


def test_paths_to_taps_places_paths_at_their_sample_delays():
    taps, dop, delays = Channel.paths_to_taps([
        {"amplitude": 1.0, "delay_ns": 0},
        {"amplitude": 0.6, "delay_ns": 500, "phase_rad": 1.2, "doppler_hz": 300.0},
    ], 20e6)
    assert taps.shape == (11,)                       # 500 ns = 10 samples
    assert np.flatnonzero(taps).tolist() == [0, 10]
    assert taps[0] == pytest.approx(1.0)
    assert taps[10] == pytest.approx(0.6 * np.exp(1j * 1.2), abs=1e-6)
    assert dop[10] == 300.0
    assert delays == {0: 0.0, 10: 500.0}


def test_paths_to_taps_reports_the_delay_it_actually_realized():
    """Delays quantize to whole samples. The helper returns what it built
    so a caller reports quantization instead of implying 50 ns resolution
    can express any physical delay."""
    _, _, delays = Channel.paths_to_taps([
        {"amplitude": 1.0, "delay_ns": 0},
        {"amplitude": 0.5, "delay_ns": 130},         # 2.6 samples -> 3
    ], 20e6)
    assert delays[3] == pytest.approx(150.0)         # not 130


def test_paths_to_taps_sums_paths_that_land_on_one_sample():
    """Two unresolvable paths add coherently -- which is what the channel
    does -- rather than one overwriting the other."""
    taps, _, _ = Channel.paths_to_taps([
        {"amplitude": 1.0, "delay_ns": 0},
        {"amplitude": 0.3, "delay_ns": 100},
        {"amplitude": 0.4, "delay_ns": 110},         # also rounds to 2 samples
    ], 20e6)
    assert taps[2] == pytest.approx(0.7, abs=1e-6)


def test_paths_to_taps_refuses_a_collision_with_different_doppler():
    """One tap cannot carry two Doppler shifts; silently dropping one
    would quietly change the channel being tested."""
    with pytest.raises(ValueError, match="two.*Doppler"):
        Channel.paths_to_taps([
            {"amplitude": 1.0, "delay_ns": 0},
            {"amplitude": 0.3, "delay_ns": 100, "doppler_hz": 100.0},
            {"amplitude": 0.4, "delay_ns": 110, "doppler_hz": 300.0},
        ], 20e6)


def test_paths_to_taps_round_trips_through_process():
    """The helper's output must actually drive process() to the same thing
    a hand-built dense channel would."""
    x = (np.random.default_rng(11).standard_normal((1, 2000))
         + 1j * np.random.default_rng(12).standard_normal((1, 2000))).astype("complex64")
    taps, dop, _ = Channel.paths_to_taps([
        {"amplitude": 1.0, "delay_ns": 0, "doppler_hz": 1600.0},
        {"amplitude": 0.8, "delay_ns": 200, "phase_rad": 0.7, "doppler_hz": 1900.0},
    ], 20e6)
    via_helper = Channel(multipath_taps=taps, tap_doppler_hz=dop,
                         sample_rate_hz=20e6, backend="numpy").process(x)
    dense = np.zeros(5, dtype="complex64")
    dense[0], dense[4] = 1.0, 0.8 * np.exp(1j * 0.7)
    dense_dop = np.array([1600.0, 0, 0, 0, 1900.0])
    via_dense = Channel(multipath_taps=dense, tap_doppler_hz=dense_dop,
                        sample_rate_hz=20e6, backend="numpy").process(x)
    np.testing.assert_allclose(np.asarray(via_helper), np.asarray(via_dense), atol=1e-6)


def test_zero_taps_are_skipped_without_changing_the_result():
    """The sparse-skip optimization must be invisible: a channel padded
    with zero taps has to equal the same channel without them."""
    x = (np.random.default_rng(13).standard_normal((1, 1500))
         + 1j * np.random.default_rng(14).standard_normal((1, 1500))).astype("complex64")
    tight = np.array([1.0, 0.5], dtype="complex64")
    padded = np.array([1.0, 0.5, 0.0, 0.0, 0.0], dtype="complex64")
    a = Channel(multipath_taps=tight, tap_doppler_hz=[0.0, 200.0],
                sample_rate_hz=20e6, backend="numpy").process(x)
    b = Channel(multipath_taps=padded, tap_doppler_hz=[0.0, 200.0, 0, 0, 0],
                sample_rate_hz=20e6, backend="numpy").process(x)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=0, rtol=0)
