"""Correctness gate for cfo/_numba_schmidl_cox.py's fused estimate/correct
kernels -- same two-gate discipline as the sync kernel
(tests/test_sync_numba_acceleration.py) and the CRC numba path
(tests/test_fec_native_acceleration.py): correctness first,
unconditionally, before trusting any benchmark number.

Skipped (not failed) when numba isn't installed -- see
cfo/_numba_schmidl_cox.py's own docstring for the silent-fallback
contract this mirrors.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectracuda.backend import cupy_available
from spectracuda.cfo import _numba_schmidl_cox as nb_mod
from spectracuda.cfo import schmidl_cox as cfo_mod
from spectracuda.cfo.schmidl_cox import SchmidlCoxCFO
from spectracuda.sync.schmidl_cox import SchmidlCoxSync

_NUMBA_OK = nb_mod.numba_available()


def _numpy_estimate(rx: np.ndarray, start_index: np.ndarray, L: int) -> np.ndarray:
    """The original per-batch-item xp.sum computation, called directly so
    both paths can be compared on identical input within one test process."""
    n_batch = rx.shape[0]
    cfo = np.empty((n_batch,), dtype=np.float64)
    for b in range(n_batch):
        d = int(start_index[b])
        first_half = rx[b, d : d + L]
        second_half = rx[b, d + L : d + 2 * L]
        p = np.sum(np.conj(first_half) * second_half)
        cfo[b] = float(np.angle(p) / np.pi)
    return cfo


def _numpy_correct(rx: np.ndarray, cfo_estimate: np.ndarray, fft_size: int) -> np.ndarray:
    """The original cos()/sin() vectorized computation, called directly."""
    cfo_estimate = np.asarray(cfo_estimate).astype("float32")
    n = np.arange(rx.shape[-1], dtype="float32")
    angle = (-2 * np.pi * cfo_estimate[:, None] * n[None, :] / fft_size).astype("float32")
    phase = (np.cos(angle) + 1j * np.sin(angle)).astype(rx.dtype if np.iscomplexobj(rx) else "complex64")
    return rx * phase


def _make_cfo_frame(sync, rng, fft_size, eps_true, true_offset, tail):
    preamble = sync.generate_preamble(seed=int(rng.integers(0, 1000)))
    n = np.arange(fft_size)
    preamble_cfo = (preamble * np.exp(1j * 2 * np.pi * eps_true * n / fft_size)).astype("complex64")
    rx = np.concatenate(
        [np.zeros(true_offset, dtype="complex64"), preamble_cfo, np.zeros(tail, dtype="complex64")]
    ).astype("complex64")
    return rx[None, :]


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- CFO acceleration inactive on this machine")
def test_numba_estimate_path_is_active_by_default():
    """Proves the dispatch actually picks numba for backend='numpy', not
    just that the module imports cleanly -- same style as
    test_numba_sync_path_is_active_by_default."""
    cfo_block = SchmidlCoxCFO(64, backend="numpy")
    rx = np.zeros((1, 128), dtype="complex64")

    called = {"numba": False}
    real = cfo_mod.numba_estimate

    def _spy(rx_arr, start_index, L):
        called["numba"] = True
        return real(rx_arr, start_index, L)

    cfo_mod.numba_estimate = _spy
    try:
        cfo_block.process(rx, start_index=np.array([0]))
    finally:
        cfo_mod.numba_estimate = real
    assert called["numba"]


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- CFO acceleration inactive on this machine")
def test_numba_correct_path_is_active_by_default():
    cfo_block = SchmidlCoxCFO(64, backend="numpy")
    rx = np.zeros((1, 128), dtype="complex64")

    called = {"numba": False}
    real = cfo_mod.numba_correct

    def _spy(rx_arr, cfo_estimate, fft_size):
        called["numba"] = True
        return real(rx_arr, cfo_estimate, fft_size)

    cfo_mod.numba_correct = _spy
    try:
        cfo_block.correct(rx, np.array([0.1]))
    finally:
        cfo_mod.numba_correct = real
    assert called["numba"]


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- CFO acceleration inactive on this machine")
@pytest.mark.parametrize("fft_size", [64, 256])
def test_numba_estimate_matches_numpy_reference(fft_size):
    """Cross-check the fused numba estimate kernel against the original
    xp.sum-based computation across many random (eps, offset, tail)
    combinations."""
    sync = SchmidlCoxSync(fft_size, backend="numpy")
    L = sync.half_len
    rng = np.random.default_rng(42)

    for _ in range(30):
        eps_true = float(rng.uniform(-0.9, 0.9))
        true_offset = int(rng.integers(0, 300))
        tail = int(rng.integers(0, 300))
        rx = _make_cfo_frame(sync, rng, fft_size, eps_true, true_offset, tail)
        start_index = np.array([true_offset])

        np_est = _numpy_estimate(rx, start_index, L)
        nb_est = nb_mod.numba_estimate(rx, start_index, L)

        assert float(nb_est[0]) == pytest.approx(float(np_est[0]), abs=1e-6)
        assert float(nb_est[0]) == pytest.approx(eps_true, abs=1e-3)


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- CFO acceleration inactive on this machine")
@pytest.mark.parametrize("fft_size", [64, 256])
def test_numba_correct_matches_numpy_reference(fft_size):
    """Cross-check the fused numba correct kernel against the original
    cos()/sin()-vectorized computation across many random (eps, n_batch)
    combinations, including batches with more than one item (per-row
    kernel dispatch, not just the n_batch=1 common case)."""
    rng = np.random.default_rng(7)
    n_samples = 2000

    for _ in range(20):
        n_batch = int(rng.integers(1, 4))
        rx = (
            rng.standard_normal((n_batch, n_samples)) + 1j * rng.standard_normal((n_batch, n_samples))
        ).astype("complex64")
        cfo_estimate = rng.uniform(-0.9, 0.9, size=n_batch)

        np_out = _numpy_correct(rx, cfo_estimate, fft_size)
        nb_out = nb_mod.numba_correct(rx, cfo_estimate.astype("float64"), fft_size)

        np.testing.assert_allclose(nb_out, np_out, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- CFO acceleration inactive on this machine")
def test_numba_correct_matches_numpy_on_full_scale_frame():
    """Same cross-check at this project's real ~37.7K-sample frame length
    (fft_size=256) -- guards against any per-sample accumulation drift
    (there shouldn't be any, since each sample's phase is computed
    directly from k*i rather than incrementally, but this is the same
    real-scale gate the sync kernel's test suite applies)."""
    fft_size = 256
    n_samples = 37696
    rng = np.random.default_rng(99)
    rx = (rng.standard_normal((1, n_samples)) + 1j * rng.standard_normal((1, n_samples))).astype("complex64")
    cfo_estimate = np.array([0.37])

    np_out = _numpy_correct(rx, cfo_estimate, fft_size)
    nb_out = nb_mod.numba_correct(rx, cfo_estimate.astype("float64"), fft_size)

    np.testing.assert_allclose(nb_out, np_out, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not cupy_available(), reason="cupy/CUDA not available on this machine")
def test_cupy_backend_never_takes_the_numba_cfo_path():
    """The numba path is intentionally NOT taken for backend='cupy' --
    see schmidl_cox.py's own comment for why (would reintroduce a hidden
    device<->host round-trip). Verified by making both numba_estimate and
    numba_correct raise if ever called while backend='cupy'."""
    cfo_block = SchmidlCoxCFO(64, backend="cupy")
    xp = cfo_block.xp
    rx = xp.zeros((1, 128), dtype="complex64")

    def _boom_estimate(rx_arr, start_index, L):
        raise AssertionError("numba_estimate must not be called for backend='cupy'")

    def _boom_correct(rx_arr, cfo_estimate, fft_size):
        raise AssertionError("numba_correct must not be called for backend='cupy'")

    real_estimate, real_correct = cfo_mod.numba_estimate, cfo_mod.numba_correct
    cfo_mod.numba_estimate = _boom_estimate
    cfo_mod.numba_correct = _boom_correct
    try:
        est = cfo_block.process(rx, start_index=xp.array([0]))
        cfo_block.correct(rx, est)
    finally:
        cfo_mod.numba_estimate = real_estimate
        cfo_mod.numba_correct = real_correct
