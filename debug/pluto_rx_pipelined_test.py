"""Pipelined variant of pluto_rx_standalone_test.py: a dedicated capture
thread does nothing but call rx.rx() in a tight loop and push each raw
buffer onto a bounded queue; the main thread drains the queue and calls
rx_streaming() -- decode of buffer N overlaps with capture of buffer N+1,
instead of the serial capture-then-decode loop in the standalone test.

rx.rx() is a blocking libiio/USB read (a C extension call) and releases
the GIL while blocked on I/O, so the decode thread genuinely runs
concurrently with the next capture rather than just interleaving.

Point: at 5 Msps, decode (~5.6-6.5ms/100k-sample buffer) has ~3.5x
headroom against the ~18.7-18.9ms capture window. Run serially, the two
stack (~24.3ms > 20ms budget -> ~80% coverage, see the standalone test).
Overlapped, decode should cost nothing and coverage should recover
toward ~100% at 5 Msps, since the loop period converges to
max(capture, decode) instead of capture + decode.

Usage:
    python3 debug/pluto_rx_pipelined_test.py --uri-rx ip:192.168.3.1 --seconds 10
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
ap.add_argument("--queue-size", type=int, default=4,
                 help="bounded queue depth between capture and decode threads")
args = ap.parse_args()

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

for _ in range(10):  # same pre-flush as the standalone test
    rx.rx()

BUDGET_US = RX_SAMPLES / args.rate * 1e6

buf_queue = queue.Queue(maxsize=args.queue_size)
stop_event = threading.Event()
capture_us = []
capture_lock = threading.Lock()
n_dropped = 0
drop_lock = threading.Lock()


def capture_loop():
    global n_dropped
    while not stop_event.is_set():
        t1 = time.perf_counter()
        raw = rx.rx()
        t2 = time.perf_counter()
        buf = np.asarray(raw, dtype="complex64")
        with capture_lock:
            capture_us.append((t2 - t1) * 1e6)
        try:
            buf_queue.put(buf, timeout=1.0)
        except queue.Full:
            with drop_lock:
                n_dropped += 1


print(f"[rx-pipelined] uri_rx={args.uri_rx} rate={args.rate/1e6:.1f}Msps rx_gain={args.rx_gain:+.1f}dB "
      f"rx_samples={RX_SAMPLES} stream_chunk={STREAM_CHUNK} queue_size={args.queue_size} "
      f"running for {args.seconds:.1f}s ...")

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
print(f"  max queue depth seen: {max_queue_depth} / {args.queue_size}")
print(f"  samples actually looked at: {total_samples_seen} of {int(elapsed*args.rate)} "
      f"({coverage_pct:.2f}% real-time coverage)")
print(f"  frames completed decode (no TX active, so expected ~0): {n_decoded}")
