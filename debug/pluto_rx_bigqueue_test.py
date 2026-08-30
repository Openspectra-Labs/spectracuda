"""Plain queue.Queue producer/consumer RX test (no hand-rolled ping-pong):
a capture thread does nothing but call rx.rx() and .put() each 100k-sample
buffer onto a bounded queue.Queue; the main thread .get()s buffers and
feeds them to rx_streaming() in 2048-sample chunks.

Queue capacity is specified in IQ *samples*, not buffer count -- e.g.
--queue-samples 1000000 with RX_SAMPLES=100_000 per buffer means
maxsize=10 (10 whole buffers = 1,000,000 samples of producer/consumer
slack) before .put() blocks the capture thread.

A bigger queue absorbs *transient* decode stalls (e.g. GIL-scheduling
jitter) without dropping samples, at the cost of decode falling further
behind on stale data if it ever lags on average, not just transiently --
it's a shock absorber, not a fix for a sustained rate mismatch.

Usage:
    python3 debug/pluto_rx_bigqueue_test.py --uri-rx ip:192.168.3.1 --rate 5e6 --queue-samples 1000000 --seconds 8
"""
import argparse
import os
import queue
import sys
import threading
import time

import numpy as np
import adi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spectracuda.pipeline import Ofdm

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="rs_m8", fec1="conv_v27", crc="crc16",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
    backend="numpy",
)
RX_SAMPLES = 100_000
STREAM_CHUNK = 2048

ap = argparse.ArgumentParser()
ap.add_argument("--uri-rx", required=True)
ap.add_argument("--freq", type=float, default=2.425e9)
ap.add_argument("--rate", type=float, default=5e6)
ap.add_argument("--rx-gain", type=float, default=60.0)
ap.add_argument("--seconds", type=float, default=10.0)
ap.add_argument("--queue-samples", type=int, default=1_000_000,
                 help="queue capacity in IQ samples (converted to buffer count internally)")
args = ap.parse_args()

queue_depth_buffers = max(1, args.queue_samples // RX_SAMPLES)

rf_bw = int(max(args.rate * 1.25, 5e6))

rx = adi.Pluto(uri=args.uri_rx)
rx.sample_rate = int(args.rate)
rx.rx_lo = int(args.freq)
rx.rx_rf_bandwidth = rf_bw
rx.gain_control_mode_chan0 = "manual"
rx.rx_hardwaregain_chan0 = args.rx_gain
rx.rx_buffer_size = RX_SAMPLES

ofdm_rx = Ofdm(**PHY_KWARGS)
ofdm_rx.reset_stream()

for _ in range(10):
    rx.rx()

BUDGET_US = RX_SAMPLES / args.rate * 1e6

buf_queue = queue.Queue(maxsize=queue_depth_buffers)
stop_event = threading.Event()
capture_us = []
n_dropped = 0


def capture_loop():
    global n_dropped
    while not stop_event.is_set():
        t1 = time.perf_counter()
        raw = rx.rx()
        t2 = time.perf_counter()
        buf = np.asarray(raw, dtype="complex64")
        capture_us.append((t2 - t1) * 1e6)
        try:
            buf_queue.put(buf, timeout=1.0)
        except queue.Full:
            n_dropped += 1


print(f"[rx-bigqueue] uri_rx={args.uri_rx} rate={args.rate/1e6:.1f}Msps rx_gain={args.rx_gain:+.1f}dB "
      f"rx_samples={RX_SAMPLES} stream_chunk={STREAM_CHUNK} "
      f"queue_samples={args.queue_samples} (={queue_depth_buffers} buffers) "
      f"budget={BUDGET_US:.0f}us running for {args.seconds:.1f}s ...")

cap_thread = threading.Thread(target=capture_loop, daemon=True)
cap_thread.start()

decode_us = []
n_decoded = 0
n_buffers_seen = 0
max_queue_depth = 0

t_start = time.perf_counter()
t_end = t_start + args.seconds
while time.perf_counter() < t_end:
    try:
        buf = buf_queue.get(timeout=0.5)
    except queue.Empty:
        continue
    max_queue_depth = max(max_queue_depth, buf_queue.qsize() + 1)
    t1 = time.perf_counter()
    results = []
    for start in range(0, len(buf), STREAM_CHUNK):
        piece = buf[start:start + STREAM_CHUNK]
        results.append(ofdm_rx.rx_streaming(piece[None, :]))
    t2 = time.perf_counter()
    decode_us.append((t2 - t1) * 1e6)
    n_buffers_seen += 1
    for result in results:
        if result is not None:
            n_decoded += 1

elapsed = time.perf_counter() - t_start
stop_event.set()
cap_thread.join(timeout=2.0)

cap_arr = np.array(capture_us) if capture_us else np.array([0.0])
dec_arr = np.array(decode_us) if decode_us else np.array([0.0])
total_samples_seen = n_buffers_seen * RX_SAMPLES
coverage_pct = 100 * total_samples_seen / (elapsed * args.rate)
n_over_budget = int(np.sum(dec_arr > BUDGET_US))

print(f"\nbuffers decoded: {n_buffers_seen}  buffers captured: {len(cap_arr)}  dropped (queue full): {n_dropped}")
print(f"  decode alone over {BUDGET_US:.0f}us budget: {n_over_budget} ({100 * n_over_budget / max(n_buffers_seen, 1):.2f}%)")
print(f"  rx.rx() alone:   min={cap_arr.min():8.1f}us  median={np.median(cap_arr):8.1f}us  "
      f"mean={cap_arr.mean():8.1f}us  max={cap_arr.max():8.1f}us")
print(f"  rx_streaming():  min={dec_arr.min():8.1f}us  median={np.median(dec_arr):8.1f}us  "
      f"mean={dec_arr.mean():8.1f}us  max={dec_arr.max():8.1f}us")
print(f"  max queue depth seen: {max_queue_depth} / {queue_depth_buffers} buffers "
      f"({max_queue_depth * RX_SAMPLES} / {args.queue_samples} samples)")
print(f"  samples actually looked at: {total_samples_seen} of {int(elapsed*args.rate)} "
      f"({coverage_pct:.2f}% real-time coverage)")
print(f"  frames completed decode (no TX active, so expected ~0): {n_decoded}")
