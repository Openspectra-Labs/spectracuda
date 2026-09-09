"""Bit-exactness gate for the Numba convolutional encoder
(fec/_numba_conv_encode.py) against BOTH existing encoders: libcorrect's
(via the native wrapper) and ConvolutionalCode's pure-array path. The
encoded bits are what goes on the wire, so equality is exact, at every
length residue, 1 bit to a full 64k-bit PDU."""
from __future__ import annotations

import numpy as np
import pytest

from spectracuda.fec import _native
from spectracuda.fec import _numba_conv_encode as nb_mod
from spectracuda.fec import viterbi as viterbi_mod
from spectracuda.fec.viterbi import ConvolutionalCode

_NUMBA_OK = nb_mod.numba_available()
pytestmark = pytest.mark.skipif(not _NUMBA_OK, reason="numba not installed -- encoder acceleration inactive on this machine")

LENGTHS = [1, 2, 5, 6, 7, 8, 13, 39, 100, 194, 255, 256, 1000, 4001, 24040, 32032, 64032]


def _pure_encode(bits):
    c = ConvolutionalCode(backend="numpy")
    c._use_numba_encode = False
    c._native = None
    return np.asarray(c.encode(bits))


def test_matches_pure_array_encoder_at_every_length():
    table = nb_mod.build_output_table(viterbi_mod._G1, viterbi_mod._G2)
    rng = np.random.default_rng(1)
    for k in LENGTHS:
        n_batch = 3 if k < 5000 else 1
        bits = rng.integers(0, 2, size=(n_batch, k)).astype("uint8")
        out = nb_mod.numba_encode(bits, table, viterbi_mod._K - 1)
        assert out.shape == (n_batch, 2 * (k + 6)) and out.dtype == np.uint8
        assert np.array_equal(out, _pure_encode(bits)), f"k={k}"


@pytest.mark.skipif(not _native.native_available(), reason="native build unavailable")
def test_matches_libcorrect_encoder_at_every_length():
    native = _native.NativeConvolutional()
    table = nb_mod.build_output_table(viterbi_mod._G1, viterbi_mod._G2)
    rng = np.random.default_rng(2)
    for k in LENGTHS:
        bits = rng.integers(0, 2, size=(2, k)).astype("uint8")
        assert np.array_equal(nb_mod.numba_encode(bits, table, 6), native.encode(bits)), f"k={k}"


def test_encode_is_the_default_path_and_decodes_back(monkeypatch):
    monkeypatch.delenv("SPECTRACUDA_VITERBI_BACKEND", raising=False)
    c = ConvolutionalCode(backend="numpy")
    assert c._use_numba_encode
    called = {"n": 0}
    real = viterbi_mod.numba_encode

    def _spy(*a, **kw):
        called["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(viterbi_mod, "numba_encode", _spy)
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=(2, 777)).astype("uint8")
    enc = c.encode(bits)
    assert called["n"] == 1
    assert np.array_equal(c.decode(enc), bits)


def test_python_override_disables_numba_encode(monkeypatch):
    monkeypatch.setenv("SPECTRACUDA_VITERBI_BACKEND", "python")
    c = ConvolutionalCode(backend="numpy")
    assert not c._use_numba_encode and c._native is None
    bits = np.random.default_rng(4).integers(0, 2, size=(1, 50)).astype("uint8")
    assert np.array_equal(c.encode(bits), _pure_encode(bits))


def test_non_binary_or_non_contiguous_input_is_handled():
    """Values are masked to their LSB (matching the table-index behavior of
    the pure path's astype/int indexing on 0/1 input) and strided views
    are copied to contiguous memory before the kernel runs."""
    table = nb_mod.build_output_table(viterbi_mod._G1, viterbi_mod._G2)
    rng = np.random.default_rng(5)
    big = rng.integers(0, 2, size=(4, 400)).astype("uint8")
    view = big[::2, 17:317]
    assert not view.flags["C_CONTIGUOUS"]
    assert np.array_equal(nb_mod.numba_encode(view, table, 6), _pure_encode(np.ascontiguousarray(view)))
