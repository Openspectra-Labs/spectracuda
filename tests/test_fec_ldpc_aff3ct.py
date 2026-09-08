"""LDPCCode(decoder="aff3ct") -- skipped entirely unless
spectracuda/fec/_native_src/aff3ct_bridge/bridge_ldpc is built locally
(requires reference/aff3ct/ built first, see that directory's own
"reference, not shipped" status and _native_aff3ct.py's own docstring),
same skip philosophy as test_fec_native_acceleration.py's C-compiler/
NEON checks. Not part of any CI expectation -- dev-machine-only coverage
for this optional backend."""

from __future__ import annotations

import numpy as np
import pytest

from spectracuda.fec import _native_aff3ct
from spectracuda.fec.ldpc import LDPCCode

_AFF3CT_OK = __import__("os").path.isfile(_native_aff3ct._BRIDGE_BIN)


@pytest.mark.skipif(not _AFF3CT_OK, reason="bridge_ldpc not built -- see _native_aff3ct.py's docstring")
class TestLDPCAff3ct:
    def test_rejects_unknown_decoder_value(self) -> None:
        with pytest.raises(ValueError, match="decoder"):
            LDPCCode("ldpc_1944_r12", decoder="not-a-real-decoder")

    def test_full_length_codeword_matches_native_decode(self) -> None:
        code_native = LDPCCode("ldpc_1944_r12", decoder="native")
        code_aff3ct = LDPCCode("ldpc_1944_r12", decoder="aff3ct")
        rng = np.random.default_rng(101)
        info_bits = rng.integers(0, 2, size=(4, code_native.k)).astype(np.uint8)
        codeword = np.asarray(code_native.encode(info_bits))
        flip = rng.random(codeword.shape) < 0.02
        noisy = codeword ^ flip.astype(np.uint8)

        native_out = np.asarray(code_native.decode(noisy, p=0.02))
        aff3ct_out = np.asarray(code_aff3ct.decode(noisy, p=0.02))

        assert np.array_equal(native_out, info_bits)
        assert np.array_equal(aff3ct_out, info_bits)
        assert np.array_equal(aff3ct_out, native_out)

    def test_shortened_codeword(self) -> None:
        code = LDPCCode("ldpc_1944_r12", decoder="aff3ct")
        rng = np.random.default_rng(202)
        real_k = 300
        info_bits = rng.integers(0, 2, size=(2, real_k)).astype(np.uint8)
        codeword = np.asarray(code.encode(info_bits))
        assert codeword.shape == (2, real_k + (code.n - code.k))
        flip = rng.random(codeword.shape) < 0.01
        noisy = codeword ^ flip.astype(np.uint8)

        decoded = np.asarray(code.decode(noisy, p=0.01))
        assert decoded.shape == (2, real_k)
        assert np.array_equal(decoded, info_bits)

    def test_non_convergence_raises_value_error(self) -> None:
        code = LDPCCode("ldpc_1944_r12", decoder="aff3ct")
        rng = np.random.default_rng(303)
        info_bits = rng.integers(0, 2, size=(1, code.k)).astype(np.uint8)
        codeword = np.asarray(code.encode(info_bits))
        flip = rng.random(codeword.shape) < 0.35  # far past this code's correction capability
        noisy = codeword ^ flip.astype(np.uint8)

        with pytest.raises(ValueError, match="converge"):
            code.decode(noisy, p=0.35)

    def test_encode_is_the_same_native_encoder_regardless_of_decoder_choice(self) -> None:
        code_native = LDPCCode("ldpc_1944_r12", decoder="native")
        code_aff3ct = LDPCCode("ldpc_1944_r12", decoder="aff3ct")
        rng = np.random.default_rng(404)
        info_bits = rng.integers(0, 2, size=(1, code_native.k)).astype(np.uint8)
        assert np.array_equal(
            np.asarray(code_native.encode(info_bits)),
            np.asarray(code_aff3ct.encode(info_bits)),
        )


def test_aff3ct_unavailable_error_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Always runs (not gated on _AFF3CT_OK) -- forces the "not built"
    path regardless of this machine's actual state, confirming it fails
    loud with a clear message rather than silently falling back."""
    monkeypatch.setattr(_native_aff3ct, "_BRIDGE_BIN", "/nonexistent/bridge_ldpc")
    _native_aff3ct._processes.clear()  # don't reuse a real cached process from another test
    code = LDPCCode("ldpc_1944_r12", decoder="aff3ct")
    info_bits = np.zeros((1, code.k), dtype=np.uint8)
    codeword = np.asarray(code.encode(info_bits))
    with pytest.raises(_native_aff3ct.Aff3ctUnavailable, match="build"):
        code.decode(codeword, p=0.02)
