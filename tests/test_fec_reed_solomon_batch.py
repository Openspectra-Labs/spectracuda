"""The batched Reed-Solomon path (one C call per batch, src/reed-solomon/
batch.c) must be byte-identical to the per-block path it replaced --
clean, with correctable errors, with shortened blocks, and it must
raise on an uncorrectable block exactly as before. Skipped when the
native build is unavailable, like tests/test_fec_native_acceleration.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectracuda.fec import _native

pytestmark = pytest.mark.skipif(not _native.native_available(), reason="native FEC build unavailable on this machine")


@pytest.fixture(scope="module")
def rs():
    return _native.NativeReedSolomon()


def _per_block_encode(rs, msg):
    return np.stack([rs._encode_one(msg[b]) for b in range(msg.shape[0])])


def _per_block_decode(rs, cw):
    return np.stack([rs._decode_one(cw[b]) for b in range(cw.shape[0])])


@pytest.mark.parametrize("real_k", [223, 200, 100, 17, 1])
@pytest.mark.parametrize("n_blocks", [1, 2, 7, 36])
def test_batched_encode_and_decode_match_per_block(rs, real_k, n_blocks):
    rng = np.random.default_rng(real_k * 100 + n_blocks)
    msg = rng.integers(0, 256, size=(n_blocks, real_k)).astype("uint8")

    enc_batch = rs.encode(msg)
    enc_ref = _per_block_encode(rs, msg)
    assert enc_batch.shape == (n_blocks, real_k + _native._RS_NROOTS)
    assert np.array_equal(enc_batch, enc_ref)

    # up to 16 symbol errors per block (t = nroots/2) must be corrected identically
    rx = enc_batch.copy()
    for b in range(n_blocks):
        n_err = int(rng.integers(0, 17))
        for pos in rng.choice(rx.shape[1], size=n_err, replace=False):
            rx[b, pos] ^= int(rng.integers(1, 256))
    dec_batch = rs.decode(rx)
    dec_ref = _per_block_decode(rs, rx)
    assert np.array_equal(dec_batch, dec_ref)
    assert np.array_equal(dec_batch, msg)


def test_uncorrectable_block_raises_like_per_block(rs):
    rng = np.random.default_rng(5)
    msg = rng.integers(0, 256, size=(4, 223)).astype("uint8")
    rx = rs.encode(msg)
    for pos in rng.choice(rx.shape[1], size=40, replace=False):  # 40 > t=16 errors in block 2
        rx[2, pos] ^= 0x5A
    with pytest.raises(ValueError):
        _per_block_decode(rs, rx)
    with pytest.raises(ValueError):
        rs.decode(rx)


def test_non_contiguous_input_is_handled(rs):
    """Callers may hand in slices/views; the batched path must copy them
    into the contiguous layout the C loop expects, not read garbage."""
    rng = np.random.default_rng(8)
    big = rng.integers(0, 256, size=(6, 300)).astype("uint8")
    msg = big[::2, 40:263]  # strided rows, offset columns -> (3, 223), non-contiguous
    assert not msg.flags["C_CONTIGUOUS"]
    enc = rs.encode(msg)
    assert np.array_equal(enc, _per_block_encode(rs, np.ascontiguousarray(msg)))
    assert np.array_equal(rs.decode(enc[::1]), msg)
