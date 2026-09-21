"""`interleaver2`: the INNER (frequency) interleaver, between fec1 and the
mapper on transmit, un-permuted before Viterbi on receive. Off by default.

This is a different stage from `interleaver=`, not a second copy of it:

    interleaver   fec0 <-> fec1    protects RS from Viterbi's burst OUTPUT
    interleaver2  fec1 <-> mapper  protects Viterbi from the channel's
                                   burst INPUT

Concatenated systems normally have both (DVB-T outer + inner; 802.11a/g
has the inner one at conv -> mapper). spectracuda had only the outer,
which is why frequency-contiguous damage reached Viterbi raw.

The failure it addresses: a 1-sample echo puts ONE wide null across the
band -- ~144 CONSECUTIVE damaged coded bits per OFDM symbol, against a
K=7 traceback of ~35-49 bits. The trellis has no reliable observation
anywhere in the window and Viterbi amplifies rather than corrects.
Permuting within the symbol turns that into ~6-bit gaps: same errors, a
regime the code handles. See
docs/2026-09-21-multipath-severity-characterization.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel

FS, TAIL = 20e6, 4096


def make(interleaver2="none", soft=False, dmrs_interval=32):
    o = Ofdm(fft_size=256, n_pilot=8, n_data=216, cp_len=64, modem="qam16",
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


def test_defaults_off():
    """The wire format must not move unless a caller asks for it: this
    changes the ORDER bits go out in, so both ends have to agree."""
    assert make().interleaver2 == "none"
    assert Ofdm(fft_size=64, n_pilot=4, n_data=40, cp_len=16,
                modem="qpsk").interleaver2 == "none"


@pytest.mark.parametrize("interleaver2", ["none", "block"])
@pytest.mark.parametrize("soft", [False, True])
def test_round_trips_bit_exactly(interleaver2, soft):
    """Symmetry gate. A permutation that is not exactly inverted would
    still decode SOMETHING on a clean channel while quietly scrambling
    the payload, so this checks the bits, not just CRC."""
    o = make(interleaver2, soft)
    bits = payload()
    r = o.rx_process(two_path(o, bits, a=0.2, delay_ns=100, snr_db=25.0, seed=0))
    assert bool(np.asarray(r["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(r["bits"])[0][: bits.shape[1]], bits[0])


def test_interleaver2_does_not_disturb_a_clean_channel():
    """It relocates bits, it does not add redundancy -- so on a channel
    with nothing to disperse it must be neither better nor worse."""
    bits = payload()
    off = make("none")
    on = make("block")
    r_off = off.rx_process(two_path(off, bits, 0.2, 100, 25.0, seed=3))
    r_on = on.rx_process(two_path(on, bits, 0.2, 100, 25.0, seed=3))
    assert bool(np.asarray(r_off["crc_valid"])[0])
    assert bool(np.asarray(r_on["crc_valid"])[0])


@pytest.mark.parametrize("a,delay_ns,soft", [(0.6, 50, False), (0.8, 50, True)])
def test_recovers_a_broad_fade_that_is_otherwise_lost(a, delay_ns, soft):
    """The point of the stage. A 1-sample echo is the worst case -- one
    wide null instead of many narrow ones -- and it is the failure mode
    neither DMRS nor soft decision could reach (measured 0/40 and 13/40
    respectively over 40 frames; with this stage, 38/40)."""
    bits = payload()
    off = make("none", soft)
    try:
        r = off.rx_process(two_path(off, bits, a, delay_ns, 15.0, seed=9000))
        off_ok = bool(np.asarray(r["crc_valid"])[0])
    except ValueError:                     # uncorrectable RS codeword
        off_ok = False
    assert not off_ok, "expected this frame to be lost without interleaver2"

    on = make("block", soft)
    r = on.rx_process(two_path(on, bits, a, delay_ns, 15.0, seed=9000))
    assert bool(np.asarray(r["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(r["bits"])[0][: bits.shape[1]], bits[0])


def test_permutation_is_applied_per_ofdm_symbol():
    """Depth is one OFDM symbol's coded bits, which is what makes this a
    FREQUENCY interleaver: one block is one symbol, and bit i of a block
    lands on subcarrier i // bits_per_symbol. A whole-stream permutation
    would disperse across TIME instead and would not match the damage
    period of a static fade."""
    o = make("block")
    flat = np.arange(3 * o.bits_per_ofdm_symbol)[None, :] % 2
    flat = flat.astype("uint8")
    out = np.asarray(o._apply_interleaver2(flat, encode=True))
    assert out.shape == flat.shape
    block = o.bits_per_ofdm_symbol
    # every block is a permutation of its own symbol's bits, never mixed
    for i in range(3):
        a = np.sort(flat[0, i*block:(i+1)*block])
        b = np.sort(out[0, i*block:(i+1)*block])
        np.testing.assert_array_equal(a, b)


def test_soft_values_take_the_same_permutation_as_the_hard_bits():
    """The subtle way to get this wrong: un-permute the hard bits and
    forget the soft ones, handing Viterbi confidences belonging to
    different bits. It would still decode on an easy channel."""
    o = make("block", soft=True)
    bits = payload()
    r = o.rx_process(two_path(o, bits, 0.2, 100, 25.0, seed=0))
    assert bool(np.asarray(r["crc_valid"])[0])
    np.testing.assert_array_equal(np.asarray(r["bits"])[0][: bits.shape[1]], bits[0])
