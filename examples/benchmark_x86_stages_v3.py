"""v3 of benchmark_x86_stages_v2.py, revised: stopwatch-timed stage
breakdown, not cProfile-timed.

v1 patched in a standalone Numba prototype (before Numba/native accel was
promoted into the library). v2 patched in a SEPARATE, hand-rolled
libcorrect binding (examples/libcorrect_backend.py) instead of using
spectracuda's own, already-fixed native path (spectracuda/fec/_native.py's
NativeConvolutional/NativeReedSolomon) -- and it turned out that separate
binding doesn't even carry the _DECODE_PAD_PAIRS decode-truncation fix
_native.py has. v2's own "decode check: all correct" only held by
coincidence (rs_m8's byte-granular output always lands conv_v27 on the one
T%8 residue that's safe even unpatched, see that investigation) -- not
because it was exercising the real fix.

That fix lives INSIDE spectracuda now, and activation is, by design,
"FULLY AUTOMATIC and TRANSPARENT, not a new constructor argument or config
flag" (fec/_native.py's own module docstring) -- same story for CRC's
Numba path (fec/crc.py's generate_key() calls numba_generate_key()
directly if available). Mac(mode=..., ofdm_kwargs=...) is built completely
normally, and whichever backend is actually active on this machine
(native C, Numba, or the pure-Python fallback if neither compiled) is
exactly what send_iq()/receive_iq() use.

This revision's own history: the ORIGINAL v3 got its per-stage breakdown
from cProfile/pstats reading the real call graph, specifically to avoid
having to patch anything (unlike v1/v2). That turned out to be a real
mistake, found by direct investigation (not assumed): cProfile's own
instrumentation overhead scales with the SIZE of the whole call graph
being traced (it hooks every function call/return in the process while
active, not just the ones a caller cares about), and that overhead does
NOT distribute evenly across stages -- it disproportionately inflates
stages built from many small nested calls. Reed-Solomon decode (~21 small
Python-level calls per frame: pack/unpack + a per-block ctypes call each)
was measured at ~3x its real cost under cProfile precisely because of
this, while Viterbi decode (1 big call per frame) was barely affected.
Verified conclusively: a plain time.perf_counter() stopwatch wrapped
directly around the SAME real calls in the SAME real pipeline (no
cProfile at all) consistently agreed with an isolated, out-of-pipeline
timing of the identical call -- cProfile alone was the outlier.

So this version DOES monkey-patch now (see _install_timing_patch()) --
but only to wrap each real method with a stopwatch, never to swap in a
different implementation (contrast with v1/v2's patches, which replaced
ConvolutionalCode/ReedSolomonCode's methods with a separate, non-
production binding). Whatever backend is really active (native C, Numba,
pure-Python fallback) still runs completely unchanged underneath every
wrapped call -- this changes how the number is MEASURED, not what code
runs. A stopwatch wrapper's own overhead (one timer read before the
call, one after, a dict increment) is orders of magnitude smaller than
cProfile's per-call-graph-wide hook, and critically does not compound
with how deep or wide the REST of the pipeline's call graph is -- which
is exactly the property that made cProfile's numbers untrustworthy here.

The headline TX/RX timing numbers are still measured on a plain,
UNINSTRUMENTED pass (nothing patched at all) -- the stopwatch-instrumented
pass, run separately over the same real inputs, is used only for the
stage-breakdown proportions, and is reconciled against that pristine
headline total (see "everything else" below) rather than trusted as a
total in its own right.

CPU affinity pinning: this process pins itself to a single CPU core at
startup (see _pin_to_one_core()) purely to cut run-to-run measurement
noise, not for any real-world deployment reason. Verified this actually
helps, not assumed: on this project's own dev machine (a WSL2 VM on a
hybrid P/E-core laptop chip), an isolated, sensitive probe (Viterbi
decode timing) run unpinned had std-dev ~0.45ms across repeated calls;
pinned to any single core (tried 8 different ones), std-dev dropped to
~0.06-0.13ms -- a real, ~4-8x tightening, though the specific core
chosen didn't matter much (no single core was a clear winner). This is
NOT a full fix for machine noise -- WSL2's guest Linux has no visibility
into the host's actual physical P/E-core assignment (/proc/cpuinfo
reports a flat, identical MHz for every logical CPU, and there's no
/sys/.../cpufreq info at all), so Windows' own hypervisor scheduler can
still move the pinned *virtual* CPU across different *physical* cores
underneath us -- pinning only stops OUR OWN process from migrating
across the 14 virtual CPUs Linux can see, which is still worth doing,
just not a guarantee of bare-metal-grade determinism. Override the core
via the SPECTRACUDA_BENCH_PIN_CORE env var (an integer core id, or "off"
to disable pinning entirely); defaults to core 2.

Usage:
    python examples/benchmark_x86_stages_v3.py
    python examples/benchmark_x86_stages_v3.py 32000            # override SDU bit count
    python examples/benchmark_x86_stages_v3.py qam16 32000      # + override modem scheme
    python examples/benchmark_x86_stages_v3.py 32000 qam16      # order doesn't matter
    SPECTRACUDA_BENCH_PIN_CORE=off python examples/benchmark_x86_stages_v3.py  # to compare
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np

from spectracuda.cfo.schmidl_cox import SchmidlCoxCFO
from spectracuda.channel.ls import LSChannelEstimator
from spectracuda.equalizer.mmse import MMSEEqualizer
from spectracuda.fec import _native, _numba_crc
from spectracuda.fec.crc import CRC
from spectracuda.fec.reed_solomon import ReedSolomonCode
from spectracuda.fec.viterbi import ConvolutionalCode
from spectracuda.modem import Modem
from spectracuda.mac import Mac
from spectracuda.mac.um import UmEntity
from spectracuda.ofdm.fft import OfdmDemodulator
from spectracuda.sync.schmidl_cox import SchmidlCoxSync

FFT_SIZE = 256
N_PILOT = 8
N_DATA = 216
CP_LEN_DEFAULT = 32  # overridable with cp=N -- see _parse_args()
N_ROUNDS = 30
N_WARMUP = 5
DEFAULT_PIN_CORE = 2
_VALID_MODEMS = {"bpsk", "qpsk", "qam16", "qam64", "qam256"}


def _parse_args():
    """Accepts a bit count (e.g. 32000), a modem scheme (e.g. qam16) and
    a cyclic-prefix length (e.g. cp=64) as positional args, IN ANY ORDER
    -- classified by shape (digits -> bit count, cp=N -> cyclic prefix,
    else -> modem scheme name), not position, so callers don't need to
    remember an argument order.

    `cp` exists because the project moved its reference numerology after
    this benchmark was written. cp=32 gives a 288-sample symbol (28.8 us
    at 10 MSps); cp=64 gives 320 samples, which at 20 MSps is 78.125 kHz
    subcarrier spacing, a 12.8 us symbol and a 3.2 us guard -- the
    802.11ax grid the Doppler and multipath characterization work all
    used. Measuring the old numerology and reasoning about the new one is
    how stale throughput numbers happen.

    `dmrs` turns on periodic channel refresh (dmrs=16/32/64, or 0 = off,
    the default this benchmark has always used). It is NOT free on either
    side: TX emits one extra training-like symbol per interval, and RX
    re-estimates H[k] at each of them and equalizes every payload symbol
    against its own segment's estimate instead of one frame-wide estimate.
    Leaving it off measures a receiver that does strictly less work than
    the one the Doppler/multipath characterization describes.

    `soft` turns on soft-decision Viterbi for the inner code. It is OFF
    by default in the library and here, and it is expensive today: the
    demapper does a max-log LLR pass instead of a nearest-point decision,
    and libcorrect has no fast/neon SOFT kernel so the decode forfeits
    the accelerated one. It buys large multipath robustness -- see
    docs/2026-09-21-multipath-severity-characterization.md -- so the
    question this flag exists to answer is what that robustness costs.

    Note the real-time budget MOVES with cp: it is frame_samples / 20 MSps,
    and a longer CP means more samples per frame, so more wall-clock to
    process it. cp=64 is therefore easier to hit than cp=32 at the same
    payload, which is a property of the budget, not of the receiver.
    """
    sdu_bits = 24000
    modem_scheme = "qpsk"
    cp_len = CP_LEN_DEFAULT
    dmrs_interval = 0
    soft_decision = False
    for arg in sys.argv[1:]:
        low = arg.lower()
        if arg.isdigit():
            sdu_bits = int(arg)
        elif low.startswith("cp=") and low[3:].isdigit():
            cp_len = int(low[3:])
        elif low.startswith("dmrs=") and low[5:].isdigit():
            dmrs_interval = int(low[5:])
        elif low in ("soft", "soft=1"):
            soft_decision = True
        elif low == "soft=0":
            soft_decision = False
        elif low in _VALID_MODEMS:
            modem_scheme = low
        else:
            raise SystemExit(
                f"Unrecognized argument {arg!r} -- expected a bit count (e.g. 32000), "
                f"a cyclic prefix (e.g. cp=64), a DMRS interval (e.g. dmrs=32), "
                f"`soft` for soft-decision Viterbi, or a modem scheme "
                f"(one of {sorted(_VALID_MODEMS)})"
            )
    if not 0 <= cp_len < FFT_SIZE:
        raise SystemExit(f"cp={cp_len} must be in [0, fft_size={FFT_SIZE})")
    if dmrs_interval not in (0, 16, 32, 64):
        raise SystemExit(f"dmrs={dmrs_interval} must be one of 0/16/32/64 "
                         f"(the 2-bit wire codes -- see framing/dmrs.py)")
    return sdu_bits, modem_scheme, cp_len, dmrs_interval, soft_decision


SDU_BITS, MODEM_SCHEME, CP_LEN, DMRS_INTERVAL, SOFT_DECISION = _parse_args()


def _pin_to_one_core() -> str:
    """Pin this process to a single CPU core to cut run-to-run
    measurement noise -- see module docstring for the measured
    before/after and its caveats (this is a benchmark-only concern, not
    a real-world deployment technique). Returns a short status string
    for the printed header; never raises -- a platform without
    sched_setaffinity (non-Linux), or a requested core id past this
    machine's actual core count, just leaves affinity untouched and
    says so, the same "fail silently, this is a transparent nicety, not
    a user-facing contract" pattern already used for native-backend
    detection elsewhere in this project."""
    override = os.environ.get("SPECTRACUDA_BENCH_PIN_CORE")
    if override is not None and override.strip().lower() == "off":
        return "disabled (SPECTRACUDA_BENCH_PIN_CORE=off)"
    if not hasattr(os, "sched_setaffinity"):
        return "unavailable (no os.sched_setaffinity on this platform)"
    try:
        core = int(override) if override is not None else DEFAULT_PIN_CORE
        available = os.sched_getaffinity(0)
        if core not in available:
            return f"skipped (core {core} not in this process's available set {sorted(available)})"
        os.sched_setaffinity(0, {core})
        return f"pinned to core {core}"
    except Exception as exc:  # pragma: no cover -- best-effort, never fatal to the benchmark itself
        return f"failed ({exc})"


def _timed(fn, bucket: str, timings: dict):
    """Wrap an existing (unbound) method with a plain stopwatch that
    accumulates elapsed wall-clock time into timings[bucket] -- the
    SAME technique examples/benchmark_x86_stages_v2.py used, but applied
    here to spectracuda's own real methods (never a swapped-in
    implementation) purely to time them. See module docstring for why
    this replaced cProfile."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        timings[bucket] += time.perf_counter() - start
        return result

    return wrapper


def _install_timing_patch(timings: dict):
    """Monkey-patches the REAL classes' REAL methods (not a different
    binding -- see module docstring) with _timed() stopwatch wrappers,
    covering both TX-side (encode) and RX-side (decode) stages so one
    installer serves both stage-breakdown passes below. Returns a
    restore() callable that puts every original method back -- always
    call it in a finally: block, even if send_iq()/receive_iq() raises,
    so a failed run never leaves the patch installed for anything after
    it (including, critically, the pristine unpatched headline passes
    this script also runs)."""
    orig = dict(
        conv_encode=ConvolutionalCode.encode, conv_decode=ConvolutionalCode.decode,
        rs_encode=ReedSolomonCode.encode, rs_decode=ReedSolomonCode.decode,
        crc_generate=CRC.generate_key,
        sync_process=SchmidlCoxSync.process,
        cfo_process=SchmidlCoxCFO.process, cfo_correct=SchmidlCoxCFO.correct,
        ofdm_demod_process=OfdmDemodulator.process,
        ls_process=LSChannelEstimator.process,
        mmse_process=MMSEEqualizer.process,
        um_receive=UmEntity.receive,
        # Soft-decision decode is a DIFFERENT method, so instrumenting only
        # decode() silently attributed the whole soft Viterbi cost to
        # "everything else" -- it read 0.04 ms of Viterbi against 7.6 ms
        # unbucketed on a soft run. Both are timed into the same bucket so
        # the breakdown means the same thing either way.
        conv_decode_soft=ConvolutionalCode.decode_soft,
        soft_demod=Modem.demodulate_soft,
    )
    ConvolutionalCode.encode = _timed(orig["conv_encode"], "conv_encode", timings)
    ConvolutionalCode.decode = _timed(orig["conv_decode"], "conv_decode", timings)
    ConvolutionalCode.decode_soft = _timed(orig["conv_decode_soft"], "conv_decode", timings)
    Modem.demodulate_soft = _timed(orig["soft_demod"], "soft_demod", timings)
    ReedSolomonCode.encode = _timed(orig["rs_encode"], "rs_encode", timings)
    ReedSolomonCode.decode = _timed(orig["rs_decode"], "rs_decode", timings)
    CRC.generate_key = _timed(orig["crc_generate"], "crc_generate", timings)
    SchmidlCoxSync.process = _timed(orig["sync_process"], "sync_cfo", timings)
    SchmidlCoxCFO.process = _timed(orig["cfo_process"], "sync_cfo", timings)
    SchmidlCoxCFO.correct = _timed(orig["cfo_correct"], "sync_cfo", timings)
    OfdmDemodulator.process = _timed(orig["ofdm_demod_process"], "ofdm_decode", timings)
    LSChannelEstimator.process = _timed(orig["ls_process"], "chanest_eq", timings)
    MMSEEqualizer.process = _timed(orig["mmse_process"], "chanest_eq", timings)
    UmEntity.receive = _timed(orig["um_receive"], "mac_decode", timings)

    def restore():
        ConvolutionalCode.encode = orig["conv_encode"]
        ConvolutionalCode.decode = orig["conv_decode"]
        ConvolutionalCode.decode_soft = orig["conv_decode_soft"]
        Modem.demodulate_soft = orig["soft_demod"]
        ReedSolomonCode.encode = orig["rs_encode"]
        ReedSolomonCode.decode = orig["rs_decode"]
        CRC.generate_key = orig["crc_generate"]
        SchmidlCoxSync.process = orig["sync_process"]
        SchmidlCoxCFO.process = orig["cfo_process"]
        SchmidlCoxCFO.correct = orig["cfo_correct"]
        OfdmDemodulator.process = orig["ofdm_demod_process"]
        LSChannelEstimator.process = orig["ls_process"]
        MMSEEqualizer.process = orig["mmse_process"]
        UmEntity.receive = orig["um_receive"]

    return restore


def run() -> None:
    pin_status = _pin_to_one_core()
    phy_kwargs = dict(
        fft_size=FFT_SIZE, n_pilot=N_PILOT, n_data=N_DATA, cp_len=CP_LEN,
        modem=MODEM_SCHEME, fec="rs_m8", fec1="conv_v27", crc="crc16",
        sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        dmrs_interval=DMRS_INTERVAL, soft_decision=SOFT_DECISION,
        backend="numpy",
    )
    print(f"=== v3 (stopwatch-timed stage breakdown -- spectracuda's own transparent "
          f"native/Numba acceleration, whatever's actually active on THIS machine) config: "
          f"fft_size={FFT_SIZE}, n_pilot={N_PILOT}, n_data={N_DATA}, cp_len={CP_LEN}, "
          f"modem={MODEM_SCHEME}, dmrs_interval={DMRS_INTERVAL}, soft_decision={SOFT_DECISION}, fec='rs_m8' (inner), fec1='conv_v27' (outer), crc=crc16, "
          f"sync=schmidl_cox, cfo=schmidl_cox, channel_estimator=ls, equalizer=mmse, "
          f"backend=numpy, sdu_bits={SDU_BITS} ===")
    print(f"    native FEC backend (Viterbi/RS, C, transparent, fec/_native.py): "
          f"{'ACTIVE' if _native.native_available() else 'inactive -- pure-Python fallback'}")
    print(f"    Numba CRC backend (transparent, fec/_numba_crc.py): "
          f"{'ACTIVE' if _numba_crc.numba_available() else 'inactive -- pure-Python fallback'}")
    print(f"    CPU affinity (benchmark-only noise reduction, see module docstring): {pin_status}")

    tx_probe_mac = Mac(mode="um", ofdm_kwargs=phy_kwargs)
    tx_probe_mac.bound = True  # never talks to a real peer -- see v1's own module docstring
    tx_mac = Mac(mode="um", ofdm_kwargs=phy_kwargs)
    rx_mac = Mac(mode="um", ofdm_kwargs=phy_kwargs)

    req = tx_mac.build_bind_request()
    resp = rx_mac.handle_bind_request_iq(req)
    assert tx_mac.handle_bind_response_iq(resp)

    rng = np.random.default_rng(0)
    sdu = rng.integers(0, 2, size=SDU_BITS).astype("uint8")

    ofdm = tx_probe_mac.ofdm
    pdus = tx_probe_mac._impl.transmit(sdu)
    samples_per_symbol = ofdm.fft_size + ofdm.cp_len
    one_frame_iq = ofdm.generate_frame(np.asarray(pdus[0], dtype="uint8")[None, :])
    total_samples = one_frame_iq.shape[-1]
    other_symbols = (total_samples - ofdm.fft_size) / samples_per_symbol
    payload_symbols = other_symbols - ofdm.n_training_symbols - ofdm.num_symbols_header
    print(f"\nSDU: {SDU_BITS} bits -> {len(pdus)} PDU(s), {len(pdus[0])} bits/PDU "
          f"(includes MAC header + FEC/CRC overhead)")
    print(f"One frame: {total_samples} IQ samples = "
          f"1 preamble symbol ({ofdm.fft_size} samples, no CP) + "
          f"{ofdm.n_training_symbols} training + {ofdm.num_symbols_header} header + "
          f"{payload_symbols:.0f} payload OFDM symbols ({samples_per_symbol} samples/symbol each)")

    # -- full TX chain: plain-timed, NOTHING patched -- this IS the
    # headline number, so it must not carry any instrumentation overhead --
    for _ in range(N_WARMUP):
        tx_probe_mac.send_iq(sdu)
    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        probe_iq_frames = tx_probe_mac.send_iq(sdu)
    tx_time_per_round = (time.perf_counter() - start) / N_ROUNDS
    n_pdus_per_round = len(probe_iq_frames)
    # Real bug found (and fixed the same way) in v2: tx_time_per_round
    # covers ALL PDUs/frames send_iq() produced for this SDU -- for an
    # SDU spanning >1 PDU (e.g. 30000 bits -> 2 PDUs), dividing the
    # per-frame throughput math by tx_time_per_round directly understates
    # TX Msps by ~n_pdus_per_round x. Must divide by n_pdus_per_round too
    # to get genuine per-FRAME time, matching total_samples (one frame's
    # sample count) below.
    tx_time = tx_time_per_round / n_pdus_per_round
    print(f"\nfull TX chain (send_iq(), {n_pdus_per_round} PDU(s)/SDU): "
          f"{tx_time_per_round * 1000:.4f} ms/SDU ({tx_time * 1000:.4f} ms/frame)")

    # -- TX stage breakdown: a SEPARATE stopwatch-instrumented pass over
    # the exact same real send_iq() call -- proportions only, reconciled
    # against the pristine headline tx_time above, not trusted as a total
    # of its own (see module docstring for why a stopwatch, not cProfile) --
    tx_timings: dict = defaultdict(float)
    restore_tx = _install_timing_patch(tx_timings)
    try:
        for _ in range(N_ROUNDS):
            tx_probe_mac.send_iq(sdu)
    finally:
        restore_tx()
    # /N_ROUNDS/n_pdus_per_round, not just /N_ROUNDS -- same per-round-vs-
    # per-frame fix as tx_time above (these functions run once per PDU).
    crc_ms = tx_timings["crc_generate"] / N_ROUNDS / n_pdus_per_round * 1000
    conv_enc_ms = tx_timings["conv_encode"] / N_ROUNDS / n_pdus_per_round * 1000
    rs_enc_ms = tx_timings["rs_encode"] / N_ROUNDS / n_pdus_per_round * 1000
    print("TX stage breakdown (stopwatch pass -- proportions, see module docstring):")
    accounted = 0.0
    for label, ms in [
        ("CRC key generation", crc_ms),
        ("FEC encode -- Viterbi (fec1, outer)", conv_enc_ms),
        ("FEC encode -- Reed-Solomon (fec0, inner)", rs_enc_ms),
    ]:
        accounted += ms
        print(f"  {label:>50}: {ms:.4f} ms")
    print(f"  {'everything else (modem/OFDM/resource-grid/header)':>50}: {tx_time * 1000 - accounted:.4f} ms")

    # -- RX: plain-timed headline pass + decode check. A FRESH batch of
    # frames per pass (tx_mac's SN counter keeps advancing, matching one
    # real receiver's lifetime) -- reusing one fixed batch across passes
    # would make the second pass see only duplicate-SN fast-rejects,
    # not genuine reassembly/delivery work (see v1's own module
    # docstring for why that matters here). --
    for _ in range(N_WARMUP):
        for iq in tx_mac.send_iq(sdu):
            rx_mac.receive_iq(iq)

    headline_rounds = [tx_mac.send_iq(sdu) for _ in range(N_ROUNDS)]
    n_delivered = 0
    n_bit_exact = 0
    n_calls = 0
    start = time.perf_counter()
    for iq_frames in headline_rounds:
        for iq in iq_frames:
            delivered = rx_mac.receive_iq(iq)
            n_delivered += len(delivered)
            n_bit_exact += sum(np.array_equal(d, sdu) for d in delivered)
            n_calls += 1
    rx_total = (time.perf_counter() - start) / n_calls * 1000  # ms/frame, headline

    print(f"\ndecode check: {n_delivered}/{N_ROUNDS} rounds delivered a SDU, "
          f"{n_bit_exact}/{n_delivered} of those bit-exact matches of the original "
          f"({'all correct' if n_bit_exact == N_ROUNDS else 'SOME ROUNDS FAILED -- see below'})")
    if n_bit_exact != N_ROUNDS:
        print("  NOTE: a round not delivering doesn't necessarily mean a decode error -- see v1's")
        print("  module docstring for the ReassemblyBuffer window-eviction explanation.")

    # -- RX stage breakdown: a SEPARATE stopwatch-instrumented pass, same
    # real call, fresh frames again -- proportions only, reconciled
    # against the pristine headline rx_total above, see module docstring --
    profiled_rounds = [tx_mac.send_iq(sdu) for _ in range(N_ROUNDS)]
    rx_timings: dict = defaultdict(float)
    restore_rx = _install_timing_patch(rx_timings)
    n_calls_profiled = 0
    try:
        for iq_frames in profiled_rounds:
            for iq in iq_frames:
                rx_mac.receive_iq(iq)
                n_calls_profiled += 1
    finally:
        restore_rx()

    print(f"\nRX stage breakdown (stopwatch pass -- proportions, averaged over "
          f"{n_calls_profiled} frames, see module docstring):")
    buckets = [
        ("sync detect + CFO", rx_timings["sync_cfo"]),
        ("OFDM decode (FFT+CP strip)", rx_timings["ofdm_decode"]),
        ("channel estimation + equalization", rx_timings["chanest_eq"]),
        ("FEC decode -- Viterbi (fec1, outer)", rx_timings["conv_decode"]),
        ("soft demapper (max-log LLR)", rx_timings.get("soft_demod", 0.0)),
        ("FEC decode -- Reed-Solomon (fec0, inner)", rx_timings["rs_decode"]),
        ("MAC decode", rx_timings["mac_decode"]),
    ]
    rx_accounted = 0.0
    for label, total_s in buckets:
        ms = total_s / n_calls_profiled * 1000
        rx_accounted += ms
        print(f"  {label:>50}: {ms:.4f} ms")
    print(f"  {'everything else (unbucketed -- header/resource-grid/reassembly/etc.)':>50}: "
          f"{max(rx_total - rx_accounted, 0.0):.4f} ms")

    # -- headline: what this actually means for sustained throughput,
    # using the PLAIN-TIMED numbers (tx_time/rx_total), never the
    # stopwatch-instrumented pass --
    pdu_bits = len(pdus[0])
    tx_mbps = pdu_bits / (tx_time * 1000) / 1000  # bits / ms -> Mbps
    rx_mbps = pdu_bits / rx_total / 1000
    tx_msps = total_samples / (tx_time * 1000) / 1000  # samples / ms -> Msps
    rx_msps = total_samples / rx_total / 1000
    budget_ms = total_samples / 20e6 * 1000  # ms/frame budget for the real target: 20 Msps sample rate
    print(f"\n=== Throughput (single-threaded, one frame at a time) ===")
    print(f"  TX: {tx_time * 1000:.4f} ms/frame -> ~{tx_mbps:.2f} Mbps -> ~{tx_msps:.2f} Msps "
          f"({'OK' if tx_time*1000 <= budget_ms else f'{budget_ms/(tx_time*1000):.2f}x short'} of 20 Msps, budget {budget_ms:.4f} ms)")
    print(f"  RX: {rx_total:.4f} ms/frame -> ~{rx_mbps:.2f} Mbps -> ~{rx_msps:.2f} Msps "
          f"({'OK' if rx_total <= budget_ms else f'{budget_ms/rx_total:.2f}x short'} of 20 Msps, budget {budget_ms:.4f} ms)")

    # -- Real SDU-level throughput: SDU_BITS (fixed, whatever payload the
    # caller asked for) over the TOTAL wall time across every frame the
    # SDU actually took, not just one frame in isolation. The two lines
    # above are legitimate PER-FRAME rates (bits carried by exactly one
    # frame / that frame's time) -- correct in isolation, but only
    # comparable ACROSS RUNS when every run uses the same PDU/frame count
    # per SDU. Modulation order changes how many PDUs a fixed SDU needs
    # (n_pdus_per_round above), which shifts how much per-frame header/FEC
    # overhead gets paid -- so the per-frame Mbps above can go DOWN when a
    # scheme is actually delivering the SDU faster overall. This line is
    # the one to compare across different modem/SDU-size runs.
    sdu_tx_ms = tx_time * 1000 * n_pdus_per_round
    sdu_rx_ms = rx_total * n_pdus_per_round
    real_tx_mbps = SDU_BITS / sdu_tx_ms / 1000
    real_rx_mbps = SDU_BITS / sdu_rx_ms / 1000
    print(f"\n=== Real SDU-level throughput ({SDU_BITS} payload bits over ALL "
          f"{n_pdus_per_round} frame(s)/SDU -- the number to compare ACROSS runs) ===")
    print(f"  TX: {SDU_BITS} bits / {sdu_tx_ms:.4f} ms -> ~{real_tx_mbps:.2f} Mbps")
    print(f"  RX: {SDU_BITS} bits / {sdu_rx_ms:.4f} ms -> ~{real_rx_mbps:.2f} Mbps")
    print(f"  (bottleneck: {'TX' if tx_time*1000 > rx_total else 'RX'})")


if __name__ == "__main__":
    run()
