"""Soft-decision Viterbi for the inner code (fec1), opt-in via
`Ofdm(soft_decision=True)`.

Why it exists: hard-decision demodulation picks the nearest constellation
point and returns 0/1, discarding how close the symbol was to the
decision boundary. On a frequency-selective channel a deeply faded
subcarrier therefore hands the decoder WRONG bits marked maximally
confident, indistinguishable from good ones -- and measurement shows
Viterbi then amplifies rather than corrects (raw BER 0.052 in, 0.071
out). See docs/2026-09-21-multipath-severity-characterization.md.

The central invariant, asserted below: feeding hard bits to the soft
decoder as 0/255 returns EXACTLY the hard answer. All of the gain comes
from graded confidence, which is why the demapper -- not the decoder --
was the missing piece. libcorrect has shipped
correct_convolutional_decode_soft all along.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectracuda.fec._native import native_available
from spectracuda.fec.fec import FEC
from spectracuda.fec.viterbi import ConvolutionalCode
from spectracuda.modem import Modem
from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel

pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="soft-decision Viterbi is native-only (no pure-Python soft decoder)",
)

FS, N, CP = 20e6, 256, 64
TAIL = 4096


def make(soft, dmrs_interval=32, modem="qam16", interleaver2="block"):
    o = Ofdm(fft_size=N, n_pilot=8, n_data=216, cp_len=CP, modem=modem,
             fec="rs_m8", fec1="conv_v27",
             interleaver="block", interleaver_kwargs={"unit_bits": 8},
             crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
             n_training_symbols=2, dmrs_interval=dmrs_interval,
             soft_decision=soft, interleaver2=interleaver2)
    o.MAX_PAYLOAD_SYMBOLS = 256
    return o


def two_path(o, bits, a, delay_ns, snr_db, seed, phase=1.1):
    taps, dop, _ = Channel.paths_to_taps([
        {"amplitude": 1.0, "delay_ns": 0},
        {"amplitude": a, "delay_ns": delay_ns, "phase_rad": phase},
    ], FS)
    return Channel(snr_db=snr_db, multipath_taps=taps, tap_doppler_hz=dop,
                   sample_rate_hz=FS, tail_samples=TAIL,
                   noise_draw_len=300_000, seed=seed,
                   backend="numpy").process(o.generate_frame(bits))


def payload(n_bytes=5575, seed=1):
    return np.random.default_rng(seed).integers(
        0, 2, size=(1, n_bytes * 8)).astype("uint8")


# -- the invariant that explains the whole design ----------------------


def test_certain_soft_values_reproduce_the_hard_decoder_exactly():
    """0/255 carries no more information than 0/1, so the soft decoder
    must return the hard decoder's answer bit-for-bit. This is why
    binding the soft entry point alone buys nothing: the demapper has to
    stop throwing the confidence away first."""
    c = ConvolutionalCode(backend="numpy")
    msg = np.random.default_rng(0).integers(0, 2, size=(2, 300)).astype("uint8")
    enc = np.asarray(c.encode(msg))
    hard = np.asarray(c.decode(enc))
    soft = np.asarray(c.decode_soft((enc * 255).astype("uint8")))
    np.testing.assert_array_equal(soft, hard)
    np.testing.assert_array_equal(soft, msg)


# -- the soft demapper -------------------------------------------------


@pytest.mark.parametrize("scheme", ["qpsk", "qam16"])
def test_soft_demod_agrees_with_hard_when_noiseless(scheme):
    m = Modem(scheme, backend="numpy")
    bits = np.random.default_rng(0).integers(
        0, 2, size=(1, m.bits_per_symbol * 400)).astype("uint8")
    soft = np.asarray(m.demodulate_soft(m.modulate(bits)))
    np.testing.assert_array_equal((soft > 127).astype("uint8"),
                                  np.asarray(m.demodulate(m.modulate(bits))))


def test_soft_demod_confidence_falls_with_noise():
    """The whole point: the byte has to MOVE toward 128 as the symbol
    approaches a decision boundary, otherwise there is nothing to weigh."""
    m = Modem("qam16", backend="numpy")
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(1, 4 * 2000)).astype("uint8")
    sym = np.asarray(m.modulate(bits))
    clean = np.abs(np.asarray(m.demodulate_soft(sym)).astype(int) - 128).mean()
    noisy_sym = sym + (rng.standard_normal(sym.shape)
                       + 1j * rng.standard_normal(sym.shape)) * 0.25
    noisy = np.abs(np.asarray(m.demodulate_soft(noisy_sym)).astype(int) - 128).mean()
    assert clean > 120
    assert noisy < 0.8 * clean


def test_soft_demod_weight_lowers_confidence_on_faded_subcarriers():
    """|H[k]|^2 weighting is what makes a deep null self-identify. Without
    it the faded bins arrive as confident as the healthy ones."""
    m = Modem("qam16", backend="numpy")
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, size=(1, 4 * 216)).astype("uint8")
    sym = np.asarray(m.modulate(bits)).reshape(1, 216)
    sym = sym + (rng.standard_normal(sym.shape)
                 + 1j * rng.standard_normal(sym.shape)) * 0.15
    w = np.ones((1, 216))
    w[0, :20] = 0.05                       # a deeply faded group
    soft = np.asarray(m.demodulate_soft(sym, weight=w)).reshape(216, 4)
    faded = np.abs(soft[:20].astype(int) - 128).mean()
    healthy = np.abs(soft[20:].astype(int) - 128).mean()
    assert faded < 0.5 * healthy


# -- wiring ------------------------------------------------------------


def test_soft_decision_defaults_on():
    """ON by default as of the multipath work. `soft_decision_active`
    reports what actually happened -- it falls back to hard decision when
    no native soft decoder is present, because a default cannot raise on a
    machine without the compiled library."""
    assert Ofdm(fft_size=64, n_pilot=4, n_data=40, cp_len=16,
                modem="qpsk").soft_decision is True
    assert make(False).soft_decision is False
    o = Ofdm(fft_size=64, n_pilot=4, n_data=40, cp_len=16, modem="qpsk")
    assert o.soft_decision_active is native_available() or o.soft_decision_active is True


def test_rs_has_no_soft_decoder():
    """Only the inner convolutional code can consume soft values -- it is
    the one sitting against the demapper. RS is reached after Viterbi has
    already emitted hard bits."""
    with pytest.raises(NotImplementedError, match="no soft-decision"):
        FEC("rs_m8", backend="numpy").decode_soft(np.zeros((1, 16), "uint8"))


@pytest.mark.parametrize("soft", [False, True])
def test_clean_channel_round_trips_either_way(soft):
    o = make(soft)
    bits = payload()
    r = o.rx_process(two_path(o, bits, a=0.2, delay_ns=100, snr_db=25.0, seed=0))
    assert bool(np.asarray(r["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(r["bits"])[0][: bits.shape[1]], bits[0])


# -- the gate ----------------------------------------------------------


@pytest.mark.parametrize("a,delay_ns", [(0.6, 200), (0.8, 1000)])
def test_soft_recovers_a_frame_hard_decision_loses(a, delay_ns):
    """Strong static frequency-selective fade, no Doppler. Hard decision
    loses these outright (phase 2 of the multipath study measured 0/300);
    soft decision decodes them bit-exactly."""
    bits = payload()
    # interleaver2 pinned off on BOTH sides: it is on by default now and
    # recovers these frames by itself, which would mask what this test is
    # about -- soft decision's own contribution.
    hard = make(False, interleaver2="none")
    try:
        r = hard.rx_process(two_path(hard, bits, a, delay_ns, 15.0, seed=9000))
        hard_ok = bool(np.asarray(r["crc_valid"])[0])
    except ValueError:                      # uncorrectable RS codeword
        hard_ok = False
    assert not hard_ok, "expected hard decision to lose this frame"

    soft = make(True, interleaver2="none")
    r = soft.rx_process(two_path(soft, bits, a, delay_ns, 15.0, seed=9000))
    assert bool(np.asarray(r["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(r["bits"])[0][: bits.shape[1]], bits[0])


# -- the numba soft demapper -------------------------------------------


def test_numba_soft_demod_matches_the_numpy_reference():
    """The kernel decomposes per AXIS (separable square Gray QAM), while
    mapper.py's reference takes the min over ALL constellation points.
    Those are mathematically identical, so they must agree to within
    byte rounding -- this is the gate that says the fast path is not
    quietly a different demapper."""
    from spectracuda.modem._numba_mapper import numba_available
    if not numba_available():
        pytest.skip("numba not installed")
    for scheme in ("qpsk", "qam16"):
        m = Modem(scheme, backend="numpy")
        rng = np.random.default_rng(0)
        rows, carr = 40, 216
        sym = np.asarray(m.modulate(
            rng.integers(0, 2, size=(1, m.bits_per_symbol * rows * carr)).astype("uint8")
        )).reshape(rows, carr)
        sym = (sym + (rng.standard_normal((rows, carr))
                      + 1j * rng.standard_normal((rows, carr))) * 0.12).astype("complex64")
        w = np.abs(rng.standard_normal((rows, carr))) ** 2
        w = (w / w.mean()).astype("float32")

        real = Modem._numba_path_applies
        Modem._numba_path_applies = lambda self, s: False
        try:
            ref = np.asarray(m.demodulate_soft(sym, weight=w))
        finally:
            Modem._numba_path_applies = real
        got = np.asarray(m.demodulate_soft(sym, weight=w))
        assert np.abs(got.astype(int) - ref.astype(int)).max() <= 1
        assert (got != ref).mean() < 1e-3


def test_bpsk_soft_falls_back_instead_of_raising():
    """bpsk is not a separable two-axis QAM, so the per-axis kernel cannot
    express it -- it must take the numpy path rather than blowing up."""
    m = Modem("bpsk", backend="numpy")
    bits = np.random.default_rng(0).integers(0, 2, size=(1, 400)).astype("uint8")
    soft = np.asarray(m.demodulate_soft(m.modulate(bits)))
    assert soft.shape == (1, 400)
    np.testing.assert_array_equal((soft > 127).astype("uint8"),
                                  np.asarray(m.demodulate(m.modulate(bits))))


@pytest.mark.parametrize("llr_bits", [2, 3, 4, 5, 6])
def test_quantizer_produces_the_expected_level_count(llr_bits):
    """2L+1 levels with L = 2^(b-1)-1, and a level AT zero -- that is the
    'no information' symbol a faded subcarrier has to be able to emit."""
    m = Modem("qam16", backend="numpy")
    rng = np.random.default_rng(0)
    sym = np.asarray(m.modulate(rng.integers(0, 2, size=(1, 4 * 3000)).astype("uint8")))
    sym = sym + (rng.standard_normal(sym.shape) + 1j * rng.standard_normal(sym.shape)) * 0.35
    soft = np.asarray(m.demodulate_soft(sym, llr_bits=llr_bits))
    assert len(np.unique(soft)) <= 2 * (2 ** (llr_bits - 1) - 1) + 1
    assert 128 in np.unique(soft)


def test_llr_bits_below_two_is_rejected():
    m = Modem("qam16", backend="numpy")
    sym = np.asarray(m.modulate(np.zeros((1, 16), "uint8")))
    with pytest.raises(ValueError, match="1 bit IS hard decision"):
        m.demodulate_soft(sym, llr_bits=1)


def test_soft_decode_prefers_the_sse_backend_when_available():
    """SSE soft is measured 7.3x faster than the portable soft loop on a
    realistic graded frame, with bit-identical output. If the selection
    silently fell back to portable, soft decision would cost ~5x more RX
    time than it needs to and nothing would fail."""
    from spectracuda.fec._native import sse_available
    c = ConvolutionalCode(backend="numpy")
    msg = np.random.default_rng(0).integers(0, 2, size=(1, 300)).astype("uint8")
    enc = np.asarray(c.encode(msg))
    c.decode_soft((enc * 255).astype("uint8"))
    expected = "NativeConvolutionalSSE" if sse_available() else "NativeConvolutional"
    assert type(c._native_soft).__name__ == expected


def test_sse_and_portable_soft_decoders_agree_on_graded_input():
    """The two must not diverge -- 0/255 rails would not exercise the
    branch metrics that differ between the scalar and vector loops."""
    from spectracuda.fec._native import (NativeConvolutional,
                                         NativeConvolutionalSSE,
                                         sse_available)
    if not sse_available():
        pytest.skip("no SSE build on this machine")
    p, q = NativeConvolutional(), NativeConvolutionalSSE()
    msg = np.random.default_rng(0).integers(0, 2, size=(1, 4000)).astype("uint8")
    enc = np.asarray(p.encode(msg))
    rng = np.random.default_rng(1)
    soft = np.clip(128 + (enc.astype(int) * 2 - 1) * rng.integers(5, 127, enc.shape),
                   0, 255).astype("uint8")
    np.testing.assert_array_equal(np.asarray(p.decode_soft(soft)),
                                  np.asarray(q.decode_soft(soft)))
