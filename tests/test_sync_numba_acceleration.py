"""Correctness gate for sync/_numba_schmidl_cox.py's fused sliding-window
kernel -- same two-gate discipline as the NEON Viterbi kernel and the
CRC numba path (tests/test_fec_native_acceleration.py): correctness
first, unconditionally, before trusting any benchmark number.

Skipped (not failed) when numba isn't installed -- see
sync/_numba_schmidl_cox.py's own docstring for the silent-fallback
contract this mirrors.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectracuda.backend import cupy_available
from spectracuda.sync import _numba_schmidl_cox as nb_mod
from spectracuda.sync import schmidl_cox as sc_mod
from spectracuda.sync.schmidl_cox import SchmidlCoxSync

_NUMBA_OK = nb_mod.numba_available()


def _numpy_reference(rx: np.ndarray, L: int):
    """The original cumsum-based computation, called directly so both
    paths can be compared on identical input within one test process."""
    n_samples = rx.shape[-1]
    a = np.conj(rx[:, :-L]) * rx[:, L:]
    n_batch = rx.shape[0]

    def _cumsum_with_leading_zero(x):
        zero = np.zeros((n_batch, 1), dtype=x.dtype)
        return np.concatenate([zero, np.cumsum(x, axis=-1)], axis=-1)

    a_cum = _cumsum_with_leading_zero(a)
    e = np.abs(rx) ** 2
    e_cum = _cumsum_with_leading_zero(e)
    b1_cum = e_cum[:, : n_samples - L + 1]
    b2_cum = e_cum[:, L:] - e_cum[:, L : L + 1]

    n_candidates = n_samples - 2 * L + 1
    p = a_cum[:, L : L + n_candidates] - a_cum[:, :n_candidates]
    r1 = b1_cum[:, L : L + n_candidates] - b1_cum[:, :n_candidates]
    r2 = b2_cum[:, L : L + n_candidates] - b2_cum[:, :n_candidates]
    r = 0.5 * (r1 + r2)
    metric = np.abs(p) ** 2 / (r ** 2 + 1e-12)
    start_index = np.argmax(metric, axis=-1)
    batch_idx = np.arange(n_batch)
    return start_index, metric[batch_idx, start_index]


def _make_noisy_frame(sync, rng, true_offset, tail, snr_db):
    preamble = sync.generate_preamble(seed=int(rng.integers(0, 1000)))
    rx_clean = np.concatenate(
        [np.zeros(true_offset, dtype="complex64"), preamble, np.zeros(tail, dtype="complex64")]
    ).astype("complex64")
    sig_power = float(np.mean(np.abs(preamble) ** 2))
    noise_std = np.sqrt((sig_power / 10 ** (snr_db / 10)) / 2)
    noise = (rng.standard_normal(rx_clean.shape[-1]) + 1j * rng.standard_normal(rx_clean.shape[-1])) * noise_std
    return (rx_clean + noise).astype("complex64")[None, :]


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- sync acceleration inactive on this machine")
def test_numba_sync_path_is_active_by_default():
    """Proves the dispatch actually picks numba for backend='numpy', not
    just that the module imports cleanly -- same style as
    test_crc_numba_path_is_reachable."""
    sync = SchmidlCoxSync(fft_size=64, backend="numpy")
    preamble = sync.generate_preamble(seed=1)
    rx = np.concatenate([preamble, np.zeros(20, dtype="complex64")])[None, :].astype("complex64")

    called = {"numba": False}
    real_numba_process = sc_mod.numba_process

    def _spy(rx_arr, L):
        called["numba"] = True
        return real_numba_process(rx_arr, L)

    sc_mod.numba_process = _spy
    try:
        sync.process(rx)
    finally:
        sc_mod.numba_process = real_numba_process
    assert called["numba"]


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- sync acceleration inactive on this machine")
@pytest.mark.parametrize("fft_size", [64, 256])
def test_numba_matches_numpy_reference_on_random_frames(fft_size):
    """Cross-check the fused sliding-window kernel against the original
    cumsum-based computation across many random (offset, tail, SNR)
    combinations -- this is a genuinely different algorithm (incremental
    sliding-window sums vs prefix-sum differencing), not a JIT'd copy, so
    it needs its own equivalence proof rather than assumed correctness
    from the existing small-signal test suite alone."""
    sync = SchmidlCoxSync(fft_size=fft_size, backend="numpy")
    L = sync.half_len
    rng_master = np.random.default_rng(1234)

    for _ in range(30):
        true_offset = int(rng_master.integers(0, 300))
        tail = int(rng_master.integers(0, 300))
        snr_db = float(rng_master.uniform(-5, 30))
        rx = _make_noisy_frame(sync, rng_master, true_offset, tail, snr_db)

        np_start, np_metric = _numpy_reference(rx, L)
        nb_start, nb_metric = nb_mod.numba_process(rx, L)

        assert int(np_start[0]) == int(nb_start[0])
        assert float(nb_metric[0]) == pytest.approx(float(np_metric[0]), abs=1e-4)


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- sync acceleration inactive on this machine")
def test_numba_matches_numpy_on_full_scale_frame():
    """Same cross-check at this project's real ~37.7K-sample frame
    length (fft_size=256, standard benchmark config) -- guards against
    the sliding-window accumulator drifting over many more incremental
    updates than the small toy signals above exercise."""
    sync = SchmidlCoxSync(fft_size=256, backend="numpy")
    L = sync.half_len
    rng = np.random.default_rng(99)
    tail = 37696 - 256 - 300
    rx = _make_noisy_frame(sync, rng, true_offset=300, tail=tail, snr_db=10.0)

    np_start, np_metric = _numpy_reference(rx, L)
    nb_start, nb_metric = nb_mod.numba_process(rx, L)

    assert int(np_start[0]) == int(nb_start[0]) == 300
    assert float(nb_metric[0]) == pytest.approx(float(np_metric[0]), abs=1e-4)


@pytest.mark.skipif(not cupy_available(), reason="cupy/CUDA not available on this machine")
def test_cupy_backend_never_takes_the_numba_path():
    """The numba path is intentionally NOT taken for backend='cupy' --
    see schmidl_cox.py's own comment for why (would reintroduce a hidden
    device<->host round-trip). Verified by making numba_process raise if
    it's ever called while backend='cupy'."""
    sync = SchmidlCoxSync(fft_size=64, backend="cupy")
    preamble = sync.generate_preamble(seed=1)
    rx = sync.xp.concatenate([preamble, sync.xp.zeros(20, dtype="complex64")])[None, :]

    def _boom(rx_arr, L):
        raise AssertionError("numba_process must not be called for backend='cupy'")

    real_numba_process = sc_mod.numba_process
    sc_mod.numba_process = _boom
    try:
        result = sync.process(rx)
    finally:
        sc_mod.numba_process = real_numba_process
    assert int(result["start_index"][0]) == 0
