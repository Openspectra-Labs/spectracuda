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
from spectracuda.pipeline.v2.rx_bit_domain import RxBitDomain
from spectracuda.pipeline.v2.rx_freq_domain import RxFreqDomain
from spectracuda.pipeline.v2.rx_time_domain import RxTimeDomain
from spectracuda.pipeline.v2.stage_if import SoftConfig, SymbolType
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


# ------------------------------------------------------- full chain ----

def soft_config_from(o):
    """Mirror an `Ofdm`'s soft settings into a `SoftConfig`.

    All four, always. Pinning three of them and letting the fourth drift
    is the way this suite would lie: `Ofdm`'s defaults differ from the
    FPGA profile on every one, so a half-pinned arm fails for reasons
    that have nothing to do with the partition.
    """
    return SoftConfig(enabled=o.soft_decision_active, metric=o.soft_llr_metric,
                      bits=o.soft_llr_bits, clip=o.soft_llr_clip, scale=o.soft_llr_scale)


def run_v2(o, rx, chunk=1):
    """Drive TD -> FD -> BIT by hand, single-threaded.

    This is the step-1 composition: the orchestration that the flowgraph
    will later do with queues, written out so the partition can be proved
    bit-exact BEFORE threads are introduced. Nothing here is concurrent,
    which is the point -- a failure now is a partition bug, not a race.
    """
    env = PhyEnv.from_ofdm(o)
    td, bd = RxTimeDomain(env), RxBitDomain(env)
    fd = RxFreqDomain(env, soft_config_from(o))
    fd.configure_expected_fec(o.fec, o.fec1)

    d = td.detect(rx)
    if not d["frame_found"]:
        return None
    fid = d["frame_id"]
    fd.training(td.symbols(fid, 0, env.n_training_symbols, SymbolType.TRAIN))
    cfg = fd.decode_header(td.symbols(
        fid, env.n_training_symbols, env.num_symbols_header, SymbolType.HEADER))

    base = env.n_training_symbols + env.num_symbols_header
    emitted, out = 0, None
    while emitted < cfg.n_total_slots:
        n = min(chunk, cfg.n_total_slots - emitted)
        for llr in fd.body(td.symbols(fid, base + emitted, n)):
            r = bd.push(llr)
            if r is not None:
                out = r
        emitted += n
    if out is not None:
        out["evm"] = fd.evm(fid)
    return out


def two_path(o, tx, snr_db=20.0, seed=11):
    taps, dop, _ = Channel.paths_to_taps(
        [{"amplitude": 1.0, "delay_ns": 0},
         {"amplitude": 0.5, "delay_ns": 100, "phase_rad": 1.1}], FS)
    return Channel(snr_db=snr_db, multipath_taps=taps, tap_doppler_hz=dop,
                   sample_rate_hz=FS, tail_samples=4096, noise_draw_len=300_000,
                   seed=seed, backend="numpy").process(o.generate_frame(tx))


def assert_same_outcome(o, rx, chunk=1):
    """Compare V1 and V2 INCLUDING their failures.

    An uncorrectable codeword is a legitimate outcome, not a test error,
    so "both raised" counts as equivalent. Checking only the success path
    would quietly pass a V2 that fails everywhere V1 succeeds.
    """
    def attempt(fn):
        try:
            return "ok", fn()
        except (ValueError, NotImplementedError) as exc:
            return "raise", exc

    s1, r1 = attempt(lambda: o.rx_process(rx))
    s2, r2 = attempt(lambda: run_v2(o, rx, chunk))
    assert s1 == s2, f"V1 {s1} but V2 {s2} ({r1!r} / {r2!r})"
    if s1 == "raise":
        return
    np.testing.assert_array_equal(np.asarray(r2["bits"]), np.asarray(r1["bits"]))
    np.testing.assert_array_equal(np.asarray(r2["crc_valid"]), np.asarray(r1["crc_valid"]))
    np.testing.assert_allclose(np.asarray(r2["evm"]), np.asarray(r1["evm"]), rtol=1e-5)


@pytest.mark.parametrize("modem", ["qpsk", "qam16", "qam64"])
@pytest.mark.parametrize("dmrs_interval", [0, 32])
def test_full_chain_matches_on_a_clean_channel(modem, dmrs_interval):
    o = make_ofdm(modem, dmrs_interval)
    assert_same_outcome(o, o.generate_frame(payload()))


@pytest.mark.parametrize("soft", [True, False])
@pytest.mark.parametrize("interleaver2", ["block", "none"])
@pytest.mark.parametrize("scale", ["stream", "frame"])
def test_full_chain_matches_through_a_two_path_channel(soft, interleaver2, scale):
    """The arm that found the one real bug in this partition.

    With `interleaver2="block"` and `scale="frame"`, FD emits the whole
    frame as a single batch (frame scaling is non-causal), and BIT used to
    infer the inner interleaver's block size as "total bits / number of
    batches" -- which then became the whole frame and ran the inverse
    permutation on the wrong geometry. `interleaver2="none"` passed
    throughout, because with no permutation the block size does not
    matter, which is exactly why the bug needed this cross-product to
    show up. The block size now travels in `HeaderConfig`.
    """
    o = make_ofdm("qam16", 32, soft_decision=soft, interleaver2=interleaver2,
                  soft_llr_scale=scale)
    assert_same_outcome(o, two_path(o, payload()))


@pytest.mark.parametrize("chunk", [1, 4, 8])
def test_chunk_size_does_not_change_the_result(chunk):
    """How many symbols ride in one FIFO message is a throughput knob and
    must not be a correctness one. If this ever fails, some stage is
    carrying state across a batch boundary that it should be deriving
    per symbol."""
    o = make_ofdm("qam16", 32)
    assert_same_outcome(o, two_path(o, payload()), chunk=chunk)


def test_fpga_soft_profile_matches():
    """The `thresh_wq` metric is a separate path through FD's soft
    demapper -- the 4-bit table lookup the FPGA implements
    (`llr_weight.v`'s power-of-two weight) rather than max-log. It is the
    profile the RTL is verified against, so V2 has to reproduce it too,
    and it exercises `train_h2_sum` which the max-log path never touches.
    """
    o = make_ofdm("qam16", 32, soft_llr_metric="thresh_wq", soft_llr_bits=4,
                  soft_llr_clip=3.0, soft_llr_scale="stream")
    assert_same_outcome(o, two_path(o, payload()))


# ---------------------------------------------- abort / invalidation ----

def test_bit_domain_discards_chunks_that_arrive_after_an_abort():
    """The ordering invariant, and the reason it is not obvious.

    Control events travel on their own channel so a full data queue
    cannot block the abort that the full queue caused. But that makes
    control UNORDERED with respect to data: the abort for a frame arrives
    before some of that frame's chunks. So aborting cannot just drop what
    is currently held -- the id has to stay invalidated and discard
    arrivals, or the late chunks start a fresh accumulator and a partial
    frame reaches MAC.
    """
    # `scale="stream"` on purpose: the scenario needs FD to emit many
    # batches, and under `Ofdm`'s default "frame" scaling FD correctly
    # emits the whole frame as ONE batch (the scale is non-causal), which
    # would leave nothing to arrive after the abort.
    o = make_ofdm("qam16", soft_llr_scale="stream")
    env = PhyEnv.from_ofdm(o)
    bd = RxBitDomain(env)
    fd = RxFreqDomain(env, soft_config_from(o))
    fd.configure_expected_fec(o.fec, o.fec1)
    td = RxTimeDomain(env)

    rx = o.generate_frame(payload())
    d = td.detect(rx)
    fid = d["frame_id"]
    fd.training(td.symbols(fid, 0, env.n_training_symbols, SymbolType.TRAIN))
    cfg = fd.decode_header(td.symbols(
        fid, env.n_training_symbols, env.num_symbols_header, SymbolType.HEADER))
    base = env.n_training_symbols + env.num_symbols_header
    batches = []
    for i in range(cfg.n_total_slots):
        batches.extend(fd.body(td.symbols(fid, base + i, 1)))
    assert len(batches) >= 3

    # Partway through, then the abort arrives ahead of the rest.
    for llr in batches[:2]:
        assert bd.push(llr) is None
    bd.abort(fid)
    assert bd.stats["partial_frames_discarded"] == 1

    # Every late chunk -- including the one flagged `last`, which is what
    # would otherwise trigger a decode -- must be discarded, and nothing
    # may be returned to MAC.
    for llr in batches[2:]:
        assert bd.push(llr) is None
    assert bd.stats["frames_decoded"] == 0
    assert bd.stats["dropped_invalid"] == len(batches) - 2
    assert fid not in bd._frames

    # And the id is only released explicitly, not by the last chunk.
    bd.retire(fid)
    assert fid not in bd.invalidated


def test_freq_domain_also_honours_a_sticky_abort():
    """FD needs the same sticky invalidation as BIT: TD may already have
    published symbols for an aborted frame, and FD must not resurrect it
    by lazily re-creating frame state on the next arrival."""
    o = make_ofdm("qam16")
    env = PhyEnv.from_ofdm(o)
    td, fd = RxTimeDomain(env), RxFreqDomain(env, soft_config_from(o))
    fd.configure_expected_fec(o.fec, o.fec1)
    d = td.detect(o.generate_frame(payload()))
    fid = d["frame_id"]

    fd.abort(fid)
    fd.training(td.symbols(fid, 0, env.n_training_symbols, SymbolType.TRAIN))
    assert fid not in fd._frames
    assert fd.stats["dropped_invalid"] >= 1
