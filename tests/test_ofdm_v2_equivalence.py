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

import random
import time

import numpy as np
import pytest

from spectracuda.pipeline import Ofdm
from spectracuda.pipeline.v2.env import PhyConfig, PhyEnv
from spectracuda.pipeline.v2.flowgraph import RxFlowgraph
from spectracuda.pipeline.v2.rx_bit_domain import RxBitDomain
from spectracuda.pipeline.v2.rx_freq_domain import RxFreqDomain
from spectracuda.pipeline.v2.rx_pipeline import RxPipeline
from spectracuda.pipeline.v2.rx_time_domain import RxTimeDomain, TdStream
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


# ========================================================= streaming ====
#
# Step 2 of the plan, and it runs BEFORE threads on purpose: every failure
# below is a sequencing failure, and a thread would only make it
# intermittent. All four scenarios here found or guard a real bug.


def stream_pipeline(o, chunk_symbols=1):
    return RxPipeline(PhyEnv.from_ofdm(o), soft=soft_config_from(o),
                      expected_fec=(o.fec, o.fec1), chunk_symbols=chunk_symbols)


def drive(rx, sig, chunk=2048):
    out = []
    for i in range(0, sig.shape[-1], chunk):
        out += rx.feed(sig[:, i:i + chunk])
    return out


def gap(n):
    return np.zeros((1, n), dtype="complex64")


def concat(*parts):
    return np.concatenate([np.asarray(p) for p in parts], axis=-1)


def matches(results, tx):
    for r in results:
        b = np.asarray(r["bits"])[0]
        if b.shape[0] >= tx.shape[1] and np.array_equal(b[: tx.shape[1]], tx[0]):
            return True
    return False


@pytest.mark.parametrize("chunk", [256, 512, 2048, 8192, 10**6])
def test_back_to_back_frames_of_differing_lengths(chunk):
    """Three frames of 120 / 600 / 64 bytes, and the chunk size must not
    matter. This found TWO real bugs:

    * A retired frame's preamble was forgotten while its samples were
      still buffered, so the same preamble was detected and decoded a
      SECOND time -- three transmitted frames came out as four decodes,
      the last a byte-identical duplicate. Starts are now remembered
      until the samples are trimmed.
    * `sync.process()` returns one GLOBAL argmax, so one call finds one
      preamble and that one need not be the earliest. With a chunk
      carrying several frames this decoded two of three with the wrong
      payloads, and whole-buffer feeds decoded only the LAST frame. The
      buffer is now scanned in overlapping windows, in time order.

    `chunk=10**6` deliberately delivers the entire signal in one call,
    which is the case both bugs hid in.
    """
    o = make_ofdm("qam16", soft_llr_scale="stream")
    txs = [payload(n, seed=s) for s, n in ((1, 120), (2, 600), (3, 64))]
    parts = [gap(800)]
    for tx in txs:
        parts += [o.generate_frame(tx), gap(300)]
    parts.append(gap(3000))

    rx = stream_pipeline(o)
    got = drive(rx, concat(*parts), chunk)

    assert len(got) == len(txs), f"expected {len(txs)} decodes, got {len(got)}"
    for tx in txs:
        assert matches(got, tx)
    assert rx.stream.stats["detections"] == len(txs)


def test_an_undecodable_header_costs_one_frame_and_no_more():
    """A false sync detection's most common real outcome.

    The frame must be dropped -- never decoded into something
    plausible-looking -- and the NEXT frame must still be recovered. If a
    bad header could wedge the stream, a single noise burst would take the
    link down until a reset.

    The header is replaced with noise rather than attenuated: scaling BPSK
    by a positive real does not move any decision boundary, so an
    attenuated header decodes perfectly well. (That mistake made an
    earlier version of this test pass while proving nothing.)
    """
    o = make_ofdm("qam16", soft_llr_scale="stream")
    tx_bad, tx_good = payload(200, seed=1), payload(200, seed=2)

    frame = np.asarray(o.generate_frame(tx_bad)).copy()
    h0 = o.fft_size + o.n_training_symbols * o.slot_len
    n = o.num_symbols_header * o.slot_len
    rng = np.random.default_rng(99)
    frame[:, h0:h0 + n] = (rng.normal(size=(1, n))
                           + 1j * rng.normal(size=(1, n))).astype("complex64") * 0.3

    rx = stream_pipeline(o)
    got = drive(rx, concat(gap(800), frame, gap(300),
                           o.generate_frame(tx_good), gap(3000)))

    assert rx.stats["header_failed"] == 1
    assert not matches(got, tx_bad), "a corrupted header must not decode"
    assert matches(got, tx_good), "the next frame must still be recovered"


def test_a_stream_gap_aborts_what_is_in_flight_and_recovers_after():
    """A discontinuity invalidates symbol alignment for EVERY frame in
    flight, not just one -- "the next slot_len samples are the next
    symbol" stops being true. So the gap aborts all active frames, and
    the receiver has to resynchronize cleanly afterwards rather than
    decoding across the seam."""
    o = make_ofdm("qam16", soft_llr_scale="stream")
    tx_cut, tx_after = payload(300, seed=3), payload(300, seed=4)

    frame = np.asarray(o.generate_frame(tx_cut))
    rx = stream_pipeline(o)
    got = drive(rx, concat(gap(800), frame[:, : frame.shape[-1] // 2]))
    rx.gap(5000)
    got += drive(rx, concat(o.generate_frame(tx_after), gap(3000)))

    assert rx.stats["gaps"] == 1
    assert rx.stats["aborted"] >= 1, "the half-delivered frame must be aborted"
    assert not matches(got, tx_cut)
    assert matches(got, tx_after), "must resynchronize after the gap"


def test_a_false_peak_inside_a_frame_is_suppressed_not_promoted():
    """The invariant that gates the move to threads.

    A real preamble is written into the frame's TRAINING span -- inside
    its minimum extent, where the frame is certainly still running, so any
    peak there is a false correlation. TD must record such peaks as
    candidates and start NO new frame: acting on one would re-align
    mid-frame and corrupt a frame that was decoding fine.

    This asserts the sync decision only, not that the frame still decodes
    -- overwriting training symbols destroys the channel estimate by
    construction, so requiring a good decode here would be testing
    interference instead of the FSM.
    """
    o = make_ofdm("qam16", soft_llr_scale="stream")
    frame = np.asarray(o.generate_frame(payload(600, seed=5))).copy()
    preamble = frame[:, : o.fft_size].copy()
    frame[:, o.fft_size + o.slot_len: o.fft_size + o.slot_len + o.fft_size] = preamble

    td = RxTimeDomain(PhyEnv.from_ofdm(o))
    stream = TdStream(td)
    started = []
    sig = concat(gap(600), frame, gap(2000))
    for i in range(0, sig.shape[-1], 2048):
        started += stream.feed(sig[:, i:i + 2048])

    assert len(started) == 1, f"one frame, got {len(started)}"
    assert stream.stats["suppressed_in_frame"] > 0, "the false peaks must be seen and refused"
    assert td.stats["candidates_seen"] == stream.stats["suppressed_in_frame"]
    assert td.stats["frames_aborted"] == 0, "no frame may be abandoned by a false peak"


def test_pure_noise_produces_no_frames():
    """The null hypothesis. `sync.process()` is a best-window search, not
    a detector, so it always returns SOME candidate -- without the
    threshold gate a receiver fed noise invents frames and burns FEC on
    them."""
    o = make_ofdm("qam16", soft_llr_scale="stream")
    rng = np.random.default_rng(7)
    noise = (rng.normal(size=(1, 60000)) + 1j * rng.normal(size=(1, 60000))).astype("complex64")
    rx = stream_pipeline(o)
    got = drive(rx, noise)
    assert got == []
    assert rx.stats["completed"] == 0


@pytest.mark.parametrize("chunk_symbols", [1, 4, 8])
def test_streaming_recovers_the_payload_at_every_fd_batch_size(chunk_symbols):
    """`chunk_symbols` is how many symbols ride in one FIFO message -- a
    throughput knob (step 5 sweeps it) that must never be a correctness
    one."""
    o = make_ofdm("qam16", dmrs_interval=32, soft_llr_scale="stream")
    tx = payload(400, seed=11)
    rx = stream_pipeline(o, chunk_symbols=chunk_symbols)
    got = drive(rx, concat(gap(800), o.generate_frame(tx), gap(3000)))
    assert matches(got, tx)


# ------------------------------------------------- standalone build ----

def test_standalone_env_matches_the_borrowed_one():
    """V2 must be constructible WITHOUT an `Ofdm`, and the two routes
    must agree exactly.

    This is not a formality. `PhyEnv.build()` re-derives the preamble,
    the training symbol's content and the header's bit positions from
    seeds, and a mismatch there has no error path: a wrong
    `preamble_seed` means the receiver never syncs, a wrong
    `training_seed` means it syncs and decodes nothing. An earlier version
    of `PhyConfig` defaulted both seeds to 0 instead of `Ofdm`'s 123/999
    and produced exactly that silent dead link -- caught here, by
    comparing the derived arrays element by element rather than trusting
    that two copies of one derivation stayed in step.
    """
    o = make_ofdm("qam16")
    borrowed = PhyEnv.from_ofdm(o)
    standalone = PhyEnv.build(PhyConfig(
        fft_size=o.fft_size, cp_len=o.cp_len, n_data=o.grid.n_data,
        n_pilot=o.grid.n_pilot, n_training_symbols=o.n_training_symbols,
        interleaver=o.interleaver, interleaver_kwargs=dict(o.interleaver_kwargs),
        max_payload_symbols=o.MAX_PAYLOAD_SYMBOLS, backend="numpy"))

    assert standalone.slot_len == borrowed.slot_len
    assert standalone.timing_advance == borrowed.timing_advance
    assert standalone.num_symbols_header == borrowed.num_symbols_header
    assert standalone.header_wire_len == borrowed.header_wire_len
    for name in ("preamble_time", "pilot_values", "train_grid_freq",
                 "train_known_indices", "train_known_values",
                 "header_positions_flat"):
        np.testing.assert_array_equal(
            np.asarray(getattr(standalone, name)),
            np.asarray(getattr(borrowed, name)),
            err_msg=f"{name} differs between build() and from_ofdm()")
    np.testing.assert_array_equal(np.asarray(standalone.grid.data_indices),
                                  np.asarray(borrowed.grid.data_indices))
    np.testing.assert_array_equal(np.asarray(standalone.grid.pilot_indices),
                                  np.asarray(borrowed.grid.pilot_indices))


def test_a_standalone_pipeline_decodes_a_frame_from_ofdm():
    """The point of step 3: V2 stands on its own.

    A receiver built purely from `PhyConfig` -- never having seen the
    transmitter's object -- recovers a frame `Ofdm` generated, which is
    also what a real receiver has to do (it is a separate device).
    """
    o = make_ofdm("qam16", soft_llr_scale="stream")
    env = PhyEnv.build(PhyConfig(
        fft_size=o.fft_size, cp_len=o.cp_len, n_data=o.grid.n_data,
        n_pilot=o.grid.n_pilot, n_training_symbols=o.n_training_symbols,
        interleaver=o.interleaver, interleaver_kwargs=dict(o.interleaver_kwargs),
        max_payload_symbols=o.MAX_PAYLOAD_SYMBOLS, backend="numpy"))

    tx = payload(400, seed=5)
    rx = RxPipeline(env, soft=soft_config_from(o), expected_fec=(o.fec, o.fec1))
    got = drive(rx, concat(gap(800), o.generate_frame(tx), gap(3000)))
    assert matches(got, tx)


def test_stages_own_their_blocks_rather_than_sharing_them():
    """No stateful DSP block may be shared between stages or pipelines.

    "Stateless today" is not a property to rely on across a future
    optimization -- this project has already added caches to the
    interleaver, the FEC codecs and the LDPC construction, and
    `framing/header.py:69` keeps a module-level `_HEADER_PACKETIZER`
    while `ConvolutionalCode._native_soft` is built lazily with no lock.
    Sharing is how those become races once threads arrive, so it is
    checked structurally now rather than debugged later.
    """
    env = PhyEnv.from_ofdm(make_ofdm("qam16"))
    a, b = RxPipeline(env), RxPipeline(env)

    # Across pipelines.
    for stage, attr in ((("td",), "sync"), (("td",), "cfo"), (("td",), "demod"),
                        (("fd",), "equalizer"), (("fd",), "channel_estimator"),
                        (("fd",), "header_modem")):
        oa = getattr(getattr(a, stage[0]), attr)
        ob = getattr(getattr(b, stage[0]), attr)
        assert oa is not ob, f"{stage[0]}.{attr} is shared between pipelines"

    # And the header codec, which wraps a Packetizer.
    assert a.fd._header_codec is not b.fd._header_codec
    # TD and FD hold disjoint block sets, so nothing is shared across the
    # stage boundary either.
    assert {id(a.td.sync), id(a.td.cfo), id(a.td.demod)}.isdisjoint(
        {id(a.fd.equalizer), id(a.fd.channel_estimator), id(a.fd.header_modem)})


# ------------------------------------------------------- flowgraph -----
#
# Step 4: the three stages as threads behind bounded queues. Everything
# before this existed to make these tests meaningful -- the partition is
# bit-exact, the streaming FSM is robust, and no stateful block is shared
# -- so a failure here is a CONCURRENCY failure and nothing else.


def frames_signature(results):
    """A comparable fingerprint of a run's output.

    Keyed and sorted by `frame_id`, never by completion order: completion
    order is genuinely timing-dependent, so a caller who observed it
    could not be deterministic even though the pipeline is.
    """
    return [(r["frame_id"], np.asarray(r["bits"]).tobytes(),
             bool(np.asarray(r["crc_valid"])[0]))
            for r in sorted(results, key=lambda x: x["frame_id"])]


def three_frames(o):
    txs = [payload(n, seed=s) for s, n in ((1, 120), (2, 600), (3, 64))]
    parts = [gap(800)]
    for tx in txs:
        parts += [o.generate_frame(tx), gap(300)]
    parts.append(gap(3000))
    return txs, concat(*parts)


def run_threaded(o, sig, *, chunk=2048, stalls=None, queue_depth=64,
                 chunk_symbols=1):
    fg = RxFlowgraph(PhyEnv.from_ofdm(o), soft=soft_config_from(o),
                     expected_fec=(o.fec, o.fec1), chunk_symbols=chunk_symbols,
                     queue_depth=queue_depth).start()
    try:
        for i in range(0, sig.shape[-1], chunk):
            fg.feed(sig[:, i:i + chunk])
            if stalls is not None and stalls.random() < 0.4:
                time.sleep(stalls.uniform(0.001, 0.02))
        return fg.drain(), fg.report()
    finally:
        fg.stop()


def test_threaded_output_is_identical_to_single_threaded():
    """Concurrency must add throughput, not change answers."""
    o = make_ofdm("qam16", soft_llr_scale="stream")
    txs, sig = three_frames(o)

    st = RxPipeline(PhyEnv.from_ofdm(o), soft=soft_config_from(o),
                    expected_fec=(o.fec, o.fec1))
    reference = frames_signature(drive(st, sig))
    assert len(reference) == len(txs)

    got, _ = run_threaded(o, sig)
    assert frames_signature(got) == reference
    for tx in txs:
        assert matches(got, tx)


def test_threaded_runs_are_deterministic_including_under_stalls():
    """Output that depends on thread timing is broken, so this perturbs
    the interleaving on purpose and demands the same bytes anyway.

    Determinism holds structurally rather than by luck: each frame is
    processed independently, every stage keys its per-frame state by
    `frame_id`, and the queues preserve order within a stage.
    """
    o = make_ofdm("qam16", soft_llr_scale="stream")
    _, sig = three_frames(o)

    baseline, _ = run_threaded(o, sig)
    reference = frames_signature(baseline)
    assert reference, "nothing decoded, so the comparison would be vacuous"

    for trial in range(3):
        plain, _ = run_threaded(o, sig)
        assert frames_signature(plain) == reference
        stalled, _ = run_threaded(o, sig, stalls=random.Random(trial))
        assert frames_signature(stalled) == reference


@pytest.mark.parametrize("chunk_symbols", [1, 4])
def test_threaded_works_at_several_fd_batch_sizes(chunk_symbols):
    """`chunk_symbols=1` is what found the real bug in this step.

    FD's `training()` had summed only WITHIN one batch while dividing by
    `n_training_symbols`, so a one-symbol batch yielded half of one
    symbol's channel estimate and the next batch overwrote it. Nothing
    raised -- the estimate was simply wrong and the header then failed to
    decode. The single-threaded driver always requested the whole
    training run in one call, so it stayed latent until threads chunked
    it.
    """
    o = make_ofdm("qam16", soft_llr_scale="stream")
    txs, sig = three_frames(o)
    got, report = run_threaded(o, sig, chunk_symbols=chunk_symbols)
    for tx in txs:
        assert matches(got, tx)
    # No spurious failures either. TD overshoots every frame on purpose,
    # so surplus batches arrive after a frame completes; they must be
    # counted as surplus rather than mistaken for unknown frames. Before
    # FD distinguished "finished" from "forgotten", a perfect run
    # reported up to 3 aborts and 3 header failures.
    assert report["pipeline"]["header_failed"] == 0
    assert report["pipeline"]["aborted"] == 0


def test_internal_queues_apply_backpressure_rather_than_dropping():
    """Dropping is only correct where data would otherwise be lost.

    Between stages the producer still holds its input -- TD's samples are
    in TD's own buffer -- so stalling costs nothing while dropping
    destroys a frame. An earlier version dropped here and TD, which runs
    to its cap and therefore overshoots every short frame, filled the
    queue and aborted real frames with `i1 overflow`: 0 of 3 frames
    survived. A deliberately tiny queue now proves the opposite.
    """
    o = make_ofdm("qam16", soft_llr_scale="stream")
    txs, sig = three_frames(o)
    got, report = run_threaded(o, sig, queue_depth=2)

    for tx in txs:
        assert matches(got, tx), "backpressure must not cost a frame"
    assert report["pipeline"]["overflow_aborts"] == 0
    assert report["queues"]["i1"]["dropped"] == 0
    # And it really was constrained, so the test is not vacuous: the
    # queue reached its own ceiling. (`blocked_puts` is NOT a reliable
    # witness -- it only counts puts that waited out a full timeout, and
    # FD usually drains fast enough that none do.)
    assert report["queues"]["i1"]["high_water"] == report["queues"]["i1"]["maxsize"]


def test_report_exposes_per_stage_occupancy_and_queue_pressure():
    """A pipeline runs no faster than its slowest stage, so per-stage
    busy time and per-queue high-water are the measurements that make
    step 7 answerable. They ship with the flowgraph rather than being
    retrofitted, because otherwise the first numbers anyone collects are
    the ones nobody can explain."""
    o = make_ofdm("qam16", soft_llr_scale="stream")
    _, sig = three_frames(o)
    _, report = run_threaded(o, sig)

    assert set(report["stages"]) == {"td", "fd", "bit"}
    for name, s in report["stages"].items():
        assert s["items"] > 0, f"{name} did no work"
        assert s["busy_s"] >= 0.0
    for name in ("iq", "i1", "i2"):
        q = report["queues"][name]
        assert q["put"] > 0 and q["high_water"] <= q["maxsize"]


def test_detection_offset_does_not_depend_on_scan_alignment():
    """The regression test for the worst bug in this work.

    TD scans the buffer in fixed-stride windows, and a preamble near a
    window edge could be won by a partially-covered correlation in the
    EARLIER window and reported early. The grid's alignment depends on
    `base_offset`, which depends on when the buffer was trimmed, which
    under threading depends on scheduling -- so the same signal was
    detected at different offsets from run to run.

    With 12 back-to-back frames the threaded flowgraph placed several
    frames 24 samples early (harmless: inside the cp_len=32 cyclic
    prefix, absorbed by h_hat) and one 48 samples early, which is
    OUTSIDE the CP. That frame came out with ~11% of its bits wrong in
    every symbol and failed Reed-Solomon, intermittently, while the
    serial driver -- trimming at different moments -- got it right. It
    looked exactly like a data race and was not one.

    So: many frames, serial and threaded, and the detected starts must
    agree exactly. Not "within the CP" -- exactly, because tolerating
    drift here is what hid the bug.
    """
    o = make_ofdm("qam16", soft_llr_scale="stream")
    txs, parts = [], [gap(800)]
    for seed in range(8):
        tx = payload(600, seed=seed)
        txs.append(tx)
        parts += [o.generate_frame(tx), gap(300)]
    parts.append(gap(3000))
    sig = concat(*parts)

    def detected_starts(stream):
        found = {}
        original = stream.td.begin_stream_frame

        def spy(buf, rel, metric, *, base_offset):
            fid = original(buf, rel, metric, base_offset=base_offset)
            found[fid] = base_offset + rel
            return fid

        stream.td.begin_stream_frame = spy
        return found

    st = RxPipeline(PhyEnv.from_ofdm(o), soft=soft_config_from(o),
                    expected_fec=(o.fec, o.fec1), chunk_symbols=4)
    serial = detected_starts(st.stream)
    serial_frames = drive(st, sig)
    assert len(serial_frames) == len(txs)

    for _ in range(3):
        fg = RxFlowgraph(PhyEnv.from_ofdm(o), soft=soft_config_from(o),
                         expected_fec=(o.fec, o.fec1), chunk_symbols=4,
                         queue_depth=256)
        threaded = detected_starts(fg.stream)
        fg.start()
        try:
            for i in range(0, sig.shape[-1], 2048):
                fg.feed(sig[:, i:i + 2048])
            got = fg.drain()
        finally:
            fg.stop()
        assert sorted(threaded.values()) == sorted(serial.values()), \
            "detection offsets drifted between serial and threaded"
        assert len(got) == len(txs), f"lost frames: {len(got)}/{len(txs)}"
        for tx in txs:
            assert matches(got, tx)


@pytest.mark.parametrize("n_frames,chunk_symbols", [(8, 1), (8, 4), (16, 8)])
def test_many_back_to_back_frames_survive_the_flowgraph(n_frames, chunk_symbols):
    """Three frames was not enough to expose the detection-alignment bug.

    It needed many frames back to back, because the fault depended on
    where trimming had moved `base_offset` by the time a given preamble
    was scanned -- which only varies once frames keep arriving.
    """
    o = make_ofdm("qam16", soft_llr_scale="stream")
    txs, parts = [], [gap(800)]
    for seed in range(n_frames):
        tx = payload(400, seed=seed)
        txs.append(tx)
        parts += [o.generate_frame(tx), gap(300)]
    parts.append(gap(3000))

    got, report = run_threaded(o, concat(*parts), chunk_symbols=chunk_symbols,
                               queue_depth=256)
    assert len(got) == n_frames, f"{len(got)}/{n_frames} decoded"
    for tx in txs:
        assert matches(got, tx)
    assert report["pipeline"]["payload_failed"] == 0
