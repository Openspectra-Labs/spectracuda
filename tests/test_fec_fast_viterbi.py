"""Bit-exactness gate for the "fast" Viterbi decoder
(fec/_native_src/libcorrect/src/convolutional/fast/) against libcorrect's
portable decoder -- the contract in include/correct-fast.h.

The reference is the PORTABLE build (NativeConvolutional), the thing
every other accelerated path in this project is itself measured against.
Corrupted input is the important part: clean codewords only exercise the
"everything agrees" path, while heavy corruption (well past the code's
correction capability) is what drives the tie-breaks in both phases and
the truncated-traceback divergences the fast decoder has to reproduce
exactly.

Skipped (not failed) when either native build isn't available on this
machine (no C compiler), same as tests/test_fec_native_acceleration.py.
"""
from __future__ import annotations

import ctypes
import platform

import numpy as np
import pytest

from spectracuda.fec import _native

_REF_OK = _native.native_available()
_FAST_OK = _native.fast_available()
pytestmark = pytest.mark.skipif(
    not (_REF_OK and _FAST_OK), reason="portable and/or fast native FEC build unavailable on this machine"
)


@pytest.fixture(scope="module")
def codecs():
    return _native.NativeConvolutional(), _native.NativeConvolutionalFast()


def _corrupt(bits: np.ndarray, rate: float, rng) -> np.ndarray:
    if rate == 0.0:
        return bits
    flips = rng.random(bits.shape) < rate
    return (bits ^ flips.astype("uint8")).astype("uint8")


# message lengths chosen to cover every (T mod 8) residue of the
# withheld-bits quirk, tiny and large messages, and the two PDU sizes the
# real pipeline decodes (24040 / 32032 bits, see benchmark_x86_stages_v3)
LENGTHS = [1, 2, 3, 5, 6, 7, 8, 13, 39, 100, 101, 194, 255, 256, 1000, 4001, 24040, 32032]
ERROR_RATES = [0.0, 0.01, 0.03, 0.06, 0.12, 0.25, 0.5]


@pytest.mark.parametrize("error_rate", ERROR_RATES)
def test_decode_is_bit_exact_with_portable_across_lengths(codecs, error_rate):
    ref, fast = codecs
    rng = np.random.default_rng(int(error_rate * 1000) + 7)
    for k in LENGTHS:
        for trial in range(2 if k > 4000 else 5):
            msg = rng.integers(0, 2, size=(1, k)).astype("uint8")
            encoded = ref.encode(msg)
            received = _corrupt(encoded, error_rate, rng)
            out_ref = ref.decode(received)
            out_fast = fast.decode(received)
            assert out_ref.shape == out_fast.shape == msg.shape
            assert np.array_equal(out_ref, out_fast), (
                f"k={k} error_rate={error_rate} trial={trial}: fast decoder diverged from portable "
                f"at bit {int(np.argmax(out_ref != out_fast))}"
            )
            if error_rate == 0.0:
                assert np.array_equal(out_ref, msg)  # sanity: both actually decode a clean codeword


def test_encode_is_identical_to_portable(codecs):
    ref, fast = codecs
    rng = np.random.default_rng(3)
    for k in (1, 7, 64, 999, 24040):
        msg = rng.integers(0, 2, size=(3, k)).astype("uint8")
        assert np.array_equal(ref.encode(msg), fast.encode(msg))


def test_raw_c_decode_is_byte_exact_including_withheld_bits(codecs):
    """Below the Python padding workaround: the raw
    correct_convolutional_decode() vs _fast_decode() calls must return
    the same byte count AND the same bytes for every input length,
    including the trailing-bits quirk both are expected to share."""
    ref, fast = codecs
    rng = np.random.default_rng(11)
    poly = (ctypes.c_uint16 * 2)(_native._G1, _native._G2)
    for sets in list(range(12, 80)) + [301, 1000, 4096, 12023]:
        for error_rate in (0.0, 0.05, 0.3):
            num_encoded_bits = 2 * sets
            bits = rng.integers(0, 2, size=num_encoded_bits).astype("uint8")
            encoded_bytes = np.packbits(bits)
            out_ref = (ctypes.c_uint8 * (sets // 8 + 8))()
            out_fast = (ctypes.c_uint8 * (sets // 8 + 8))()
            n_ref = _native._lib.correct_convolutional_decode(
                ref._conv, encoded_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), num_encoded_bits, out_ref
            )
            n_fast = _native._fast_lib.correct_convolutional_fast_decode(
                fast._conv, encoded_bytes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), num_encoded_bits, out_fast
            )
            assert n_ref == n_fast, f"sets={sets}: byte count {n_ref} vs {n_fast}"
            assert bytes(out_ref[:n_ref]) == bytes(out_fast[:n_fast]), f"sets={sets} error_rate={error_rate}: bytes differ"
    del poly


def test_create_rejects_unsupported_codes():
    poly = (ctypes.c_uint16 * 2)(_native._G1, _native._G2)
    assert not _native._fast_lib.correct_convolutional_fast_create(2, 9, poly)
    assert not _native._fast_lib.correct_convolutional_fast_create(3, 7, poly)
    handle = _native._fast_lib.correct_convolutional_fast_create(2, 7, poly)
    assert handle
    _native._fast_lib.correct_convolutional_fast_destroy(handle)


def test_batch_shape_contract(codecs):
    """Same (n_batch, n) in / out shape contract as the other native classes."""
    _, fast = codecs
    rng = np.random.default_rng(5)
    msg = rng.integers(0, 2, size=(4, 333)).astype("uint8")
    enc = fast.encode(msg)
    assert enc.shape == (4, 2 * (333 + _native._TAIL_BITS))
    assert np.array_equal(fast.decode(enc), msg)


# -- dispatch (fec/viterbi.py) ------------------------------------------------

@pytest.mark.skipif(platform.machine() not in ("x86_64", "AMD64"), reason="fast is only the measured default on x86_64 so far")
def test_dispatch_prefers_fast_on_x86_64(monkeypatch):
    monkeypatch.delenv("SPECTRACUDA_VITERBI_BACKEND", raising=False)
    from spectracuda.fec.viterbi import ConvolutionalCode

    assert isinstance(ConvolutionalCode(backend="numpy")._native, _native.NativeConvolutionalFast)


@pytest.mark.skipif(platform.machine() not in ("aarch64", "arm64"), reason="aarch64-only expectation")
def test_dispatch_keeps_neon_default_on_aarch64_until_measured(monkeypatch):
    """The fast kernel has not been A/B'd against NEON on real ARM hardware
    yet, so it must NOT be the default there -- opt-in only. Flip this
    test (and the dispatch) together, with the measured number."""
    monkeypatch.delenv("SPECTRACUDA_VITERBI_BACKEND", raising=False)
    from spectracuda.fec.viterbi import ConvolutionalCode

    if _native.neon_available():
        assert isinstance(ConvolutionalCode(backend="numpy")._native, _native.NativeConvolutionalNEON)


def test_dispatch_env_override(monkeypatch):
    from spectracuda.fec.viterbi import ConvolutionalCode

    monkeypatch.setenv("SPECTRACUDA_VITERBI_BACKEND", "fast")
    assert isinstance(ConvolutionalCode(backend="numpy")._native, _native.NativeConvolutionalFast)
    monkeypatch.setenv("SPECTRACUDA_VITERBI_BACKEND", "portable")
    assert type(ConvolutionalCode(backend="numpy")._native) is _native.NativeConvolutional
    monkeypatch.setenv("SPECTRACUDA_VITERBI_BACKEND", "python")
    assert ConvolutionalCode(backend="numpy")._native is None
    monkeypatch.setenv("SPECTRACUDA_VITERBI_BACKEND", "bogus")
    with pytest.raises(ValueError):
        ConvolutionalCode(backend="numpy")


def test_dispatch_env_override_refuses_to_silently_fall_back(monkeypatch):
    """An override for a backend that isn't available here must raise, not
    quietly pick something else -- otherwise an A/B measurement lies."""
    from spectracuda.fec.viterbi import ConvolutionalCode

    monkeypatch.setenv("SPECTRACUDA_VITERBI_BACKEND", "hexagon")  # never available on any dev machine yet
    with pytest.raises(RuntimeError):
        ConvolutionalCode(backend="numpy")


def test_decode_through_dispatch_matches_pure_python_with_errors(codecs, monkeypatch):
    """End-to-end through ConvolutionalCode: the dispatched fast kernel
    corrects injected errors exactly like the pure-array Viterbi does."""
    from spectracuda.fec.viterbi import ConvolutionalCode

    monkeypatch.setenv("SPECTRACUDA_VITERBI_BACKEND", "fast")
    fast_code = ConvolutionalCode(backend="numpy")
    monkeypatch.setenv("SPECTRACUDA_VITERBI_BACKEND", "python")
    pure_code = ConvolutionalCode(backend="numpy")
    rng = np.random.default_rng(9)
    for k in (39, 194, 4001):
        msg = rng.integers(0, 2, size=(2, k)).astype("uint8")
        enc = fast_code.encode(msg)
        rx = enc.copy()
        for row in range(rx.shape[0]):  # a few scattered errors, well within the code's capability
            for pos in rng.choice(rx.shape[1], size=6, replace=False):
                rx[row, pos] ^= 1
        assert np.array_equal(fast_code.decode(rx), msg)
        assert np.array_equal(pure_code.decode(rx), msg)
