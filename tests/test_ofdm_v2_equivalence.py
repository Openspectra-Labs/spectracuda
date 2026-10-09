"""`OfdmV2` against `Ofdm` -- the gate the whole V2 effort hangs on.

V2 re-partitions the receive chain into time / frequency / bit domain
blocks so the load can be spread over CPU cores (see
`spectracuda/pipeline/v2/stage_if.py`). It is a re-partition, NOT a modem
redesign: every transform is the same one `Ofdm` runs, on the same
samples. So the acceptance criterion is bit-exactness, and `Ofdm` is the
oracle -- it stays untouched as the shipping path precisely so it can
play that role.

WHY THIS IS CHECKED PER STAGE AND NOT ONLY END TO END. A whole-chain
byte comparison is not sensitive enough to catch a stage that is subtly
wrong: the FPGA work already has a live example, where the RTL's FFT
window sits 2-5 samples from Python's below ~15 dB, the decoded bytes
still match, and only the 4-bit soft values differ
(`fpga/docs/2026-10-05-hdl-rx-tx-status.md`). FEC hides small errors,
which is its job and which makes it a poor oracle for a refactor. Each
stage is therefore compared at its own interface.

A NOTE ON THE SOFT PARAMETERS. There are four of them and `Ofdm`'s
defaults differ from the FPGA profile on every one (metric, bits, clip,
scale). Any arm of this suite must pin all four on both sides or it fails
for reasons unrelated to the partition -- see `SoftConfig`'s docstring.
"""
from __future__ import annotations

import numpy as np
import pytest

from spectracuda.pipeline import Ofdm
from spectracuda.pipeline.v2.env import PhyEnv
from spectracuda.pipeline.v2.rx_time_domain import RxTimeDomain
from spectracuda.pipeline.v2.stage_if import SymbolType
from spectracuda.sim import Channel

FS = 20e6


def make_ofdm(modem="qam16", dmrs_interval=0, **extra):
    o = Ofdm(fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem=modem,
             fec="rs_m8", fec1="conv_v27",
             interleaver="block", interleaver_kwargs={"unit_bits": 8},
             crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
             n_training_symbols=2, dmrs_interval=dmrs_interval,
             backend="numpy", **extra)
    o.MAX_PAYLOAD_SYMBOLS = 256
    return o


def payload(n_bytes=500, seed=0):
    return np.random.default_rng(seed).integers(
        0, 2, size=(1, n_bytes * 8)).astype("uint8")


def v1_fft_symbols(o, rx_iq, n_sym):
    """`Ofdm`'s own time-domain path, replicated step for step.

    Deliberately written out rather than factored: this is the reference
    side of the comparison, so it has to be readable as "what V1 does"
    without following a helper into V2's code.
    """
    xp = o.xp
    rx = o._quantize(xp.asarray(rx_iq))
    start_index = o.sync.process(rx)["start_index"]
    cfo = o.cfo.process(rx, start_index=start_index)
    rx_corrected = o.cfo.correct(rx, cfo)
    pos = start_index + o.fft_size - o.timing_advance
    out = []
    for _ in range(n_sym):
        slot = o._extract_slot(rx_corrected, pos, o.slot_len)
        pos = pos + o.slot_len
        out.append(o.demod.process(slot))
    return xp.concatenate(out, axis=0)


# ---------------------------------------------------------------- TD ----

@pytest.mark.parametrize("modem", ["qpsk", "qam16", "qam64"])
def test_td_fft_bins_are_bit_exact(modem):
    """I1 is where the partition first has to hold.

    V2 runs the FFT in TD for EVERY symbol, DMRS included, where `Ofdm`
    FFTs DMRS slots later inside `_estimate_channel_from_dmrs`. The claim
    is that this moves scheduling and not arithmetic -- the same
    `demod.process` on the same samples -- and that claim is what this
    test holds to account. `array_equal`, not `allclose`: "close" would
    let a real reordering bug through.
    """
    o = make_ofdm(modem)
    rx = o.generate_frame(payload())
    n_sym = o.n_training_symbols + o.num_symbols_header + 6

    td = RxTimeDomain(PhyEnv.from_ofdm(o))
    d = td.detect(rx)
    assert d["frame_found"]
    got = td.symbols(d["frame_id"], 0, n_sym).bins

    np.testing.assert_array_equal(np.asarray(got), np.asarray(v1_fft_symbols(o, rx, n_sym)))


def test_td_is_bit_exact_through_a_real_channel():
    """Clean-frame equality can hide a quantization or dtype slip, since
    a noiseless frame leaves the ADC path and the CFO estimator in their
    easiest state. A two-path channel with noise exercises both."""
    o = make_ofdm("qam16")
    taps, dop, _ = Channel.paths_to_taps(
        [{"amplitude": 1.0, "delay_ns": 0},
         {"amplitude": 0.5, "delay_ns": 100, "phase_rad": 1.1}], FS)
    rx = Channel(snr_db=18.0, multipath_taps=taps, tap_doppler_hz=dop,
                 sample_rate_hz=FS, tail_samples=4096, seed=7,
                 backend="numpy").process(o.generate_frame(payload()))
    n_sym = o.n_training_symbols + o.num_symbols_header + 6

    td = RxTimeDomain(PhyEnv.from_ofdm(o))
    d = td.detect(rx)
    assert d["frame_found"]
    np.testing.assert_array_equal(
        np.asarray(td.symbols(d["frame_id"], 0, n_sym).bins),
        np.asarray(v1_fft_symbols(o, rx, n_sym)))


def test_td_reports_the_same_always_present_observables():
    """`rx_process`'s result contract is a fully enumerated key set
    (`ofdm.py:1215`), and `start_index` / `sync_metric` / `rssi_db` are
    present even when no frame is found. TD owns all three, so it has to
    produce them on both paths."""
    o = make_ofdm("qam16")
    td = RxTimeDomain(PhyEnv.from_ofdm(o))

    rx = o.generate_frame(payload())
    got, want = td.detect(rx), o.rx_process(rx)
    assert got["frame_found"] is True
    np.testing.assert_array_equal(np.asarray(got["start_index"]), np.asarray(want["start_index"]))
    np.testing.assert_array_equal(np.asarray(got["sync_metric"]), np.asarray(want["sync_metric"]))
    np.testing.assert_allclose(np.asarray(got["rssi_db"]), np.asarray(want["rssi_db"]))

    noise = (np.random.default_rng(3).normal(size=(1, 20000)) +
             1j * np.random.default_rng(4).normal(size=(1, 20000))).astype("complex64")
    got, want = td.detect(noise), o.rx_process(noise)
    assert got["frame_found"] is False and want["frame_found"] is False
    assert got["frame_id"] is None
    np.testing.assert_allclose(np.asarray(got["rssi_db"]), np.asarray(want["rssi_db"]))


def test_td_bounds_check_raises_valueerror_not_indexerror():
    """A truncated frame must stay a HANDLED failure. Fancy indexing
    raises IndexError, which `mac/session.py`'s
    `except (ValueError, NotImplementedError)` does not catch -- so
    getting this wrong converts a recoverable loss into a crash. `Ofdm`
    guards the same way for the same reason."""
    o = make_ofdm("qam16")
    rx = o.generate_frame(payload())
    td = RxTimeDomain(PhyEnv.from_ofdm(o))
    d = td.detect(rx)
    with pytest.raises(ValueError, match="truncated frame"):
        td.symbols(d["frame_id"], 0, 4096)


def test_candidate_evaluation_cannot_disturb_the_active_frame():
    """The invariant `rx_time_domain.py` exists to protect.

    A correlation peak inside an active frame's payload may be false, so
    offering one must be pure recording: the active frame's timing,
    samples and emitted symbols all have to come out unchanged, and its
    symbols still bit-exact. Promotion is the only path that may abandon
    a frame, and it must say so with an explicit abort.
    """
    o = make_ofdm("qam16")
    rx = o.generate_frame(payload())
    n_sym = o.n_training_symbols + o.num_symbols_header + 4

    td = RxTimeDomain(PhyEnv.from_ofdm(o))
    d = td.detect(rx)
    fid = d["frame_id"]
    before = np.asarray(td.frame(fid).pos0).copy()

    td.offer_candidate(start_index=9999, metric=0.99, sample_offset=9999)
    td.offer_candidate(start_index=12345, metric=0.42, sample_offset=12345)

    np.testing.assert_array_equal(np.asarray(td.frame(fid).pos0), before)
    np.testing.assert_array_equal(
        np.asarray(td.symbols(fid, 0, n_sym).bins),
        np.asarray(v1_fft_symbols(o, rx, n_sym)))
    assert td.stats["candidates_seen"] == 2
    assert td.stats["frames_aborted"] == 0

    # Promotion is explicit, keeps the strongest candidate, and publishes
    # an abort so FD/BIT can invalidate that frame_id.
    cand = td.promote_candidate(active_frame_id=fid, reason="test")
    assert cand is not None and cand.start_index == 9999
    events = td.drain_control()
    assert len(events) == 1 and events[0].frame_id == fid
    assert td.stats["frames_aborted"] == 1
    with pytest.raises(KeyError):
        td.symbols(fid, 0, 1)
