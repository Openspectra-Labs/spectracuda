"""RX-only, no TX at all -- v2 of pluto_rx_standalone_test.py, fixing that
script's own architectural bug rather than its symptom.

v1's problem, confirmed on real Pi-5 hardware: a single thread does
`rx.rx()` then `rx_streaming()` on it then `rx.rx()` again, serially. Any
time decode runs a bit long, the NEXT `rx.rx()` call is late -- and since
nothing is draining the Pluto's own (small, fixed-size) hardware buffer
in the meantime, samples that arrive during that delay are silently
dropped at the DRIVER/hardware level, with zero application-level
visibility into what got dropped or when. Measured effect: ~92-93%
"samples looked at" coverage translated into only ~57-61% of sent packets
actually being decoded in a real two-Pluto capture (see
docs/2026-08-27-neon-viterbi-and-rx-throughput.md-style session notes /
this debug session's own history) -- because frame capture is all-or-
nothing per frame, a small, frequent timing gap (recurring roughly every
loop iteration, same order of magnitude as one frame's own duration)
clips a much larger fraction of frames than the raw missed-time
percentage would suggest.

Fix, reusing an ALREADY-PROVEN pattern from this codebase
(examples/pluto_channel.py's PlutoChannel._rx_loop(), not invented here):
split radio-reading and decoding into two threads connected by a bounded
queue.

    [_rx_loop thread]                      [main thread]
    rx.rx() -- NEVER blocked by decode --> queue.Queue --> rx_streaming()
    (drops OLDEST queued chunk on                          (however slow
     backpressure, not the live read)                       it needs to be)

This means the ADC is drained at its own pace regardless of how long
decode takes -- decode falling behind now shows up as an EXPLICIT,
COUNTED "chunks_dropped" number (real backpressure, measured directly)
instead of an invisible hardware-level overflow inferred indirectly from
a timing budget. Comparing chunks_produced vs chunks_dropped here is the
actual, precise version of what v1's "samples actually looked at %" was
only a rough proxy for.

Usage:
    python3 debug/pluto_rx_standalone_test_v2.py --uri-rx ip:192.168.3.1 --seconds 10
"""
import argparse
import gc
import queue
import sys
import threading
import time

import numpy as np
import adi

sys.path.insert(0, "/home/abhi/spectracuda")
from spectracuda.pipeline import Ofdm

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qam64",
    fec="rs_m8", fec1="conv_v27", crc="crc16",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
    backend="numpy",
    # See debug/pluto_rx_standalone_test.py's own PHY_KWARGS comment for
    # the full story -- unchanged here, this receiver still only ever
    # expects rs_m8/conv_v27.
    strict_fec_check=True,
)
RX_SAMPLES = 100_000  # one rx.rx() call's worth -- see v1's own comment
# for why big reads are used (amortizes per-call daemon overhead), still
# true here since it's the SAME reader-thread rx.rx() call, just no
# longer serialized against decode.
STREAM_CHUNK = 1024  # MUST be < rx_streaming()'s own SEEKING-state search cap
# (STREAM_SEARCH_WINDOW_SYMBOLS(8) * fft_size(256) = 2048) -- NOT equal to it.
# When chunk_size == cap, each SEEKING-state call's post-concatenation trim
# ("keep only the last `cap` samples") discards the ENTIRE previous chunk,
# leaving zero overlap between consecutive search windows. Any preamble that
# straddles a chunk boundary is then unrecoverable: truncated in the chunk
# it starts in, and the samples it needs before that point are already gone
# by the time the next chunk arrives. Confirmed empirically (pure-simulation,
# no real channel): with STREAM_CHUNK=2048 this missed 6/64 (9.4%) of tested
# frame-start alignments, specifically ones landing in the last ~192 samples
# before a chunk boundary; dropping to 1024 (half the cap, guaranteeing
# cap-chunk=1024 samples of retained overlap every call) measured 0/32
# missed. This -- not weak RF signal, not an RX/TX FEC-scheme mismatch --
# is the real explanation for frames that go missing entirely despite this
# script's own reader thread reporting ~100% ADC coverage and 0% queue
# drops: the loss happens inside rx_streaming()'s SEEKING state, downstream
# of everything this script can see.

ap = argparse.ArgumentParser()
ap.add_argument("--uri-rx", required=True)
ap.add_argument("--freq", type=float, default=2.425e9)
ap.add_argument("--rate", type=float, default=5e6)
ap.add_argument("--rx-gain", type=float, default=60.0)
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--queue-size", type=int, default=1024,
                 help="max buffered STREAM_CHUNK-sized pieces before the reader "
                      "thread starts dropping the OLDEST one to make room for new "
                      "samples (matches PlutoChannel's own default) -- at "
                      "STREAM_CHUNK=2048 and --rate 4e6 this is ~524ms of cushion")
args = ap.parse_args()

rf_bw = int(max(args.rate * 1.25, 5e6))

rx = adi.Pluto(uri=args.uri_rx)  # NO tx object opened at all in this script
rx.sample_rate = int(args.rate)
rx.rx_lo = int(args.freq)
rx.rx_rf_bandwidth = rf_bw
rx.gain_control_mode_chan0 = "manual"
rx.rx_hardwaregain_chan0 = args.rx_gain
rx.rx_buffer_size = RX_SAMPLES

ofdm_rx = Ofdm(**PHY_KWARGS)
ofdm_rx.reset_stream()

for _ in range(10):  # PySDR's own recommended flush before real measurement
    rx.rx()


class _ReaderStats:
    """Written only by the reader thread, read only by the main thread
    after stop_event is set -- a lock isn't strictly required for that
    single-writer/read-after-join pattern, but this is a measurement
    script and the counts are the whole point, so correctness is kept
    explicit rather than relying on CPython GIL incidentals."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.rx_calls = 0
        self.samples_read = 0
        self.chunks_produced = 0
        self.chunks_dropped = 0
        self.max_queue_depth = 0

    def record_rx_call(self, n_samples: int) -> None:
        with self._lock:
            self.rx_calls += 1
            self.samples_read += n_samples

    def record_chunk(self, dropped: bool, queue_depth_after: int) -> None:
        with self._lock:
            self.chunks_produced += 1
            if dropped:
                self.chunks_dropped += 1
            if queue_depth_after > self.max_queue_depth:
                self.max_queue_depth = queue_depth_after


chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=args.queue_size)
reader_stats = _ReaderStats()
stop_event = threading.Event()


rx_call_gaps_us = []  # written only by the reader thread; time spent
# BETWEEN the end of one rx.rx() call and the start of the next -- i.e.
# the reader loop's own per-iteration overhead (slicing + queue puts).
# If the Pluto's internal DMA buffer isn't tolerant of that gap, this is
# real ADC time nothing is listening during -- a smaller-scale repeat of
# the exact same class of loss v1 had, one level down (between reader-
# thread rx.rx() calls, not between decode iterations).
_prev_call_end = [None]


def _rx_loop() -> None:
    """Mirrors examples/pluto_channel.py's PlutoChannel._rx_loop() --
    same drop-OLDEST-on-backpressure logic (see that module's own
    comment for why: never stall the live ADC read behind a slow
    consumer -- only fresh incoming samples matter, stale backlog is
    what should be sacrificed under sustained decode backpressure)."""
    while not stop_event.is_set():
        t_call_start = time.perf_counter()
        if _prev_call_end[0] is not None:
            rx_call_gaps_us.append((t_call_start - _prev_call_end[0]) * 1e6)
        buf = np.asarray(rx.rx(), dtype="complex64")
        _prev_call_end[0] = time.perf_counter()
        reader_stats.record_rx_call(len(buf))
        for start in range(0, len(buf), STREAM_CHUNK):
            chunk = buf[start:start + STREAM_CHUNK]
            if len(chunk) == 0:
                continue
            dropped = False
            try:
                chunk_queue.put_nowait(chunk)
            except queue.Full:
                dropped = True
                try:
                    chunk_queue.get_nowait()  # drop the oldest queued chunk
                except queue.Empty:
                    pass
                chunk_queue.put_nowait(chunk)
            reader_stats.record_chunk(dropped, chunk_queue.qsize())


reader_thread = threading.Thread(target=_rx_loop, daemon=True)

decode_us = []
n_decoded = 0
n_crc_valid = 0
n_consumed = 0
SLOW_CHUNK_THRESHOLD_US = 50_000  # a single STREAM_CHUNK's rx_streaming() call
# taking >50ms would itself be a red flag (the old LDPC-construction bug
# would have shown up exactly like this, per-chunk, before it was fixed --
# see debug/pluto_rx_standalone_test.py's PHY_KWARGS comment) -- kept as a
# regression tripwire, not expected to ever fire now.

print(f"[rx-standalone-v2] uri_rx={args.uri_rx} rate={args.rate/1e6:.1f}Msps rx_gain={args.rx_gain:+.1f}dB "
      f"rx_samples={RX_SAMPLES} stream_chunk={STREAM_CHUNK} queue_size={args.queue_size} "
      f"running for {args.seconds:.1f}s ...")

reader_thread.start()
t_end = time.perf_counter() + args.seconds
while time.perf_counter() < t_end:
    try:
        chunk = chunk_queue.get(timeout=0.5)
    except queue.Empty:
        continue  # reader thread hasn't produced anything yet this tick -- keep waiting for t_end
    n_consumed += 1

    t0 = time.perf_counter()
    result = ofdm_rx.rx_streaming(chunk[None, :])
    chunk_decode_us = (time.perf_counter() - t0) * 1e6
    decode_us.append(chunk_decode_us)
    if chunk_decode_us > SLOW_CHUNK_THRESHOLD_US:
        print(f"  [SLOW] single-chunk rx_streaming() took {chunk_decode_us:.0f}us "
              f"(> {SLOW_CHUNK_THRESHOLD_US}us tripwire) -- unexpected, worth investigating")

    if result is not None:
        n_decoded += 1
        crc_ok = result.get("crc_valid") is not None and bool(np.asarray(result["crc_valid"])[0])
        if crc_ok:
            n_crc_valid += 1
        evm = float(np.asarray(result["evm"])[0]) if result.get("evm") is not None else float("nan")
        rssi_db = float(np.asarray(result["rssi_db"])[0]) if result.get("rssi_db") is not None else float("nan")
        print(f"  [decoded] #{n_decoded} crc_valid={crc_ok} evm={evm:.5f} rssi_db={rssi_db:.1f}")

stop_event.set()
reader_thread.join(timeout=5.0)

# Explicit, deterministic teardown -- same fix as debug/pluto_rx_pingpong_test.py's
# own (already-proven) one, for the same root cause: the reader thread's closure
# keeps `rx` reachable in a way plain refcounting doesn't clear until a cyclic GC
# pass runs, and that pass can destroy the libiio Buffer/Device/Context objects in
# an arbitrary order (e.g. Buffer after its parent Context), a use-after-free in
# libiio's C code that segfaults on interpreter exit. Drop every reference and
# collect here instead, while nothing else has started tearing down.
del reader_thread
chunk_queue = None
rx = None
gc.collect()

dec_arr = np.array(decode_us) if decode_us else np.array([0.0])
adc_coverage_pct = 100 * reader_stats.samples_read / (args.seconds * args.rate)
drop_pct = 100 * reader_stats.chunks_dropped / max(reader_stats.chunks_produced, 1)
consume_pct = 100 * n_consumed / max(reader_stats.chunks_produced, 1)

gap_arr = np.array(rx_call_gaps_us) if rx_call_gaps_us else np.array([0.0])
nominal_buffer_us = RX_SAMPLES / args.rate * 1e6

print(f"\n=== SUMMARY over {args.seconds:.1f}s ===")
print(f"[reader thread -- never blocked by decode]")
print(f"  rx.rx() calls: {reader_stats.rx_calls}  samples read: {reader_stats.samples_read} of "
      f"{int(args.seconds * args.rate)} ({adc_coverage_pct:.2f}% ADC coverage -- should be ~100% now)")
print(f"  gap between consecutive rx.rx() calls (reader loop's own overhead -- real ADC time "
      f"unlistened-to if the HW buffer can't absorb it): min={gap_arr.min():7.1f}us  "
      f"median={np.median(gap_arr):7.1f}us  mean={gap_arr.mean():7.1f}us  max={gap_arr.max():7.1f}us  "
      f"(one {RX_SAMPLES}-sample buffer = {nominal_buffer_us:.0f}us of real time)")
print(f"  STREAM_CHUNK pieces produced: {reader_stats.chunks_produced}  "
      f"dropped under backpressure: {reader_stats.chunks_dropped} ({drop_pct:.2f}%)")
print(f"  max queue depth observed: {reader_stats.max_queue_depth} of {args.queue_size}")
print(f"[main thread -- decode, however slow it needs to be]")
print(f"  chunks consumed: {n_consumed} of {reader_stats.chunks_produced} produced ({consume_pct:.2f}%)")
print(f"  time per single-chunk rx_streaming() call: min={dec_arr.min():8.1f}us  "
      f"median={np.median(dec_arr):8.1f}us  mean={dec_arr.mean():8.1f}us  max={dec_arr.max():8.1f}us")
print(f"Frames completed decode: {n_decoded}")
print(f"CRC-valid: {n_crc_valid}/{n_decoded}" + (f" ({100 * n_crc_valid / n_decoded:.1f}%)" if n_decoded else " (n/a -- no frames completed decode)"))
