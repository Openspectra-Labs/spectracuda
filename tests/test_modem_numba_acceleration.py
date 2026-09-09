"""Correctness gate for modem/_numba_mapper.py's fused hard-decision +
EVM kernel -- same two-gate discipline as every accelerated kernel
before it (NEON Viterbi, CRC, sync, CFO): bit-exact bits first,
unconditionally, before any benchmark number is trusted. Skipped (not
failed) when numba isn't installed.

The bits are the FEC's input, so the bar is bit-for-bit equality with
the numpy path across all five schemes, at high AND low SNR (low SNR is
what actually exercises the round/clip decision boundaries), plus
symbols placed deliberately ON those boundaries.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectracuda.backend import cupy_available
from spectracuda.modem import _numba_mapper as nb_mod
from spectracuda.modem import mapper as mapper_mod
from spectracuda.modem import Modem

_NUMBA_OK = nb_mod.numba_available()
SCHEMES = ["bpsk", "qpsk", "qam16", "qam64", "qam256"]


def _noisy_symbols(modem: Modem, rng, n_rows: int, n_symbols: int, snr_db: float) -> np.ndarray:
    bits = rng.integers(0, 2, size=(n_rows, n_symbols * modem.bits_per_symbol)).astype("uint8")
    clean = modem.modulate(bits)
    noise_std = np.sqrt(10 ** (-snr_db / 10) / 2)
    noise = (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)) * noise_std
    return (clean + noise).astype("complex64")


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- modem acceleration inactive on this machine")
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("snr_db", [30.0, 10.0, 0.0, -5.0])
def test_bits_are_bit_exact_with_numpy_path(scheme, snr_db):
    modem = Modem(scheme, backend="numpy")
    rng = np.random.default_rng(hash((scheme, snr_db)) % (2**32))
    symbols = _noisy_symbols(modem, rng, n_rows=7, n_symbols=997, snr_db=snr_db)

    numpy_bits = modem._demodulate_numpy(symbols)
    numba_bits, _, _ = nb_mod.numba_hard_decision(symbols, scheme, modem.bits_per_symbol, modem._norm)

    assert numba_bits.dtype == np.uint8
    assert numba_bits.shape == numpy_bits.shape
    assert np.array_equal(numba_bits, numpy_bits)


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- modem acceleration inactive on this machine")
@pytest.mark.parametrize("scheme", ["qam16", "qam64", "qam256"])
def test_bits_are_bit_exact_on_decision_boundaries(scheme):
    """Symbols sitting exactly on / a float32-ulp either side of the
    per-axis decision thresholds -- where round-half-to-even and clip
    decide the bit, and where a float64-vs-float32 reimplementation
    would first diverge. Also far outside the constellation (clip)."""
    modem = Modem(scheme, backend="numpy")
    half = modem.bits_per_symbol // 2
    m = 2 ** half
    # decision thresholds between adjacent PAM levels are the even
    # integers -(m-2)..(m-2) in descaled units; multiply back by norm
    thresholds = np.arange(-(m - 2), m - 1, 2, dtype=np.float32) * np.float32(modem._norm)
    around = np.concatenate([
        thresholds,
        np.nextafter(thresholds, np.float32(np.inf)),
        np.nextafter(thresholds, np.float32(-np.inf)),
        np.array([-1e3, 1e3, -0.0, 0.0], dtype=np.float32),
    ]).astype(np.float32)
    re, im = np.meshgrid(around, around)
    symbols = (re + 1j * im).astype("complex64").reshape(1, -1)

    numpy_bits = modem._demodulate_numpy(symbols)
    numba_bits, _, _ = nb_mod.numba_hard_decision(symbols, scheme, modem.bits_per_symbol, modem._norm)
    assert np.array_equal(numba_bits, numpy_bits)


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- modem acceleration inactive on this machine")
@pytest.mark.parametrize("scheme", SCHEMES)
def test_evm_sums_match_modulate_round_trip(scheme):
    """sqrt(err/ref) from the fused kernel must equal the numpy
    definition rx_process() used before: compute_evm(symbols,
    modulate(demodulate(symbols)))."""
    from spectracuda.framing import compute_evm

    modem = Modem(scheme, backend="numpy")
    rng = np.random.default_rng(7)
    symbols = _noisy_symbols(modem, rng, n_rows=5, n_symbols=1234, snr_db=12.0)

    bits, err, ref = modem.demodulate_stats(symbols)
    assert np.array_equal(bits, modem._demodulate_numpy(symbols))
    reference_evm = compute_evm(np, symbols, modem.modulate(bits))
    np.testing.assert_allclose(np.sqrt(err / ref), reference_evm, rtol=1e-5)


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- modem acceleration inactive on this machine")
def test_numba_path_is_active_by_default_for_numpy_backend():
    """Proves the dispatch actually reaches the kernel for a plain
    complex64 numpy input -- not just that the module imports."""
    modem = Modem("qam16", backend="numpy")
    symbols = _noisy_symbols(modem, np.random.default_rng(1), 2, 64, 20.0)
    called = {"n": 0}
    real = mapper_mod.numba_hard_decision

    def _spy(*args, **kwargs):
        called["n"] += 1
        return real(*args, **kwargs)

    mapper_mod.numba_hard_decision = _spy
    try:
        modem.demodulate(symbols)
        modem.demodulate_stats(symbols)
    finally:
        mapper_mod.numba_hard_decision = real
    assert called["n"] == 2


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- modem acceleration inactive on this machine")
def test_complex128_input_keeps_the_numpy_path():
    """The kernel's bit-exactness rests on float32 arithmetic; a
    complex128 input must not be silently narrowed into it."""
    modem = Modem("qam64", backend="numpy")
    symbols = _noisy_symbols(modem, np.random.default_rng(2), 1, 32, 20.0).astype("complex128")
    real = mapper_mod.numba_hard_decision

    def _boom(*args, **kwargs):
        raise AssertionError("numba kernel must not run on complex128 input")

    mapper_mod.numba_hard_decision = _boom
    try:
        bits = modem.demodulate(symbols)
    finally:
        mapper_mod.numba_hard_decision = real
    assert np.array_equal(bits, modem._demodulate_numpy(symbols))


@pytest.mark.skipif(not cupy_available(), reason="cupy/CUDA not available on this machine")
def test_cupy_backend_never_takes_the_numba_path():
    modem = Modem("qpsk", backend="cupy")
    xp = modem.xp
    symbols = xp.asarray(_noisy_symbols(Modem("qpsk", backend="numpy"), np.random.default_rng(3), 1, 32, 20.0))
    real = mapper_mod.numba_hard_decision

    def _boom(*args, **kwargs):
        raise AssertionError("numba kernel must not run for backend='cupy'")

    mapper_mod.numba_hard_decision = _boom
    try:
        modem.demodulate(symbols)
        modem.demodulate_stats(symbols)
    finally:
        mapper_mod.numba_hard_decision = real


@pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- modem acceleration inactive on this machine")
def test_rx_process_bits_identical_and_evm_equal_with_kernel_on_vs_off():
    """End-to-end through the real pipeline: rx_process() on a real,
    bound, noisy frame must return identical payload bits and the same
    EVM (to float32 tolerance) whether the fused kernel is active or the
    numpy round-trip fallback is forced."""
    from spectracuda.mac import Mac

    phy = dict(
        fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qam16",
        fec="rs_m8", fec1="conv_v27", crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse", backend="numpy",
    )
    tx = Mac(mode="um", ofdm_kwargs=phy)
    rx = Mac(mode="um", ofdm_kwargs=phy)
    assert tx.handle_bind_response_iq(rx.handle_bind_request_iq(tx.build_bind_request()))
    rng = np.random.default_rng(11)
    frame = np.asarray(tx.send_iq(rng.integers(0, 2, size=16000).astype("uint8"))[0])
    rms = float(np.sqrt(np.mean(np.abs(frame) ** 2)))
    noisy = (frame + (rng.standard_normal(frame.shape) + 1j * rng.standard_normal(frame.shape)) * rms / 20).astype("complex64")

    on = rx.ofdm.rx_process(noisy)
    real = mapper_mod.numba_available
    mapper_mod.numba_available = lambda: False
    try:
        off = rx.ofdm.rx_process(noisy)
    finally:
        mapper_mod.numba_available = real

    assert on["frame_found"] and off["frame_found"]
    assert np.array_equal(np.asarray(on["bits"]), np.asarray(off["bits"]))
    assert bool(np.all(on["crc_valid"])) and bool(np.all(off["crc_valid"]))
    np.testing.assert_allclose(np.asarray(on["evm"]), np.asarray(off["evm"]), rtol=1e-4)
    assert np.asarray(on["evm"]).dtype == np.float32
