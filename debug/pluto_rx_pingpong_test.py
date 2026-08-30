"""Ping-pong (2-slot double-buffer) variant of the pipelined RX test.

Instead of a general N-deep queue.Queue between capture and decode threads
(pluto_rx_pipelined_test.py, which saturated its depth-4 queue and produced
wild decode-time variance -- mean 29ms vs median 5.3ms, one 3.1s outlier),
this uses exactly 2 fixed slots with a strict alternation, the classic
double-buffering discipline: capture fills slot A while decode drains slot
B, then they swap and wait on each other via Events -- no unbounded slack,
no queue-internal locking, capture can never get more than one buffer
ahead of decode.

Usage:
    python3 debug/pluto_rx_pingpong_test.py --uri-rx ip:192.168.3.1 --rate 4e6 --seconds 8
"""
import argparse
import gc
import os
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
# Matches debug/pluto_tx_standalone_test.py's PAYLOAD_BYTES exactly -- the
# fixed 64-byte pattern the TX side sends every --interval, so a decoded
# frame's content can be verified byte-for-byte, not just counted.
EXPECTED_PAYLOAD_BYTES = bytes(range(64))

ap = argparse.ArgumentParser()
ap.add_argument("--uri-rx", required=True)
ap.add_argument("--freq", type=float, default=2.425e9)
ap.add_argument("--rate", type=float, default=4e6)
ap.add_argument("--rx-gain", type=float, default=60.0)
ap.add_argument("--seconds", type=float, default=10.0, help="0 = run until Ctrl-C")
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

for _ in range(10):
    rx.rx()

BUDGET_US = RX_SAMPLES / args.rate * 1e6

# Exactly 2 slots. buf_ready[i]: capture -> decode ("slot i has fresh data").
# buf_free[i]: decode -> capture ("slot i has been drained, safe to overwrite").
slots = [None, None]
buf_ready = [threading.Event(), threading.Event()]
buf_free = [threading.Event(), threading.Event()]
for e in buf_free:
    e.set()  # both slots start empty/free
stop_event = threading.Event()

capture_us = []
n_capture_waits = 0  # times capture had to wait for decode to free a slot


def capture_loop():
    global n_capture_waits
    i = 0
    while not stop_event.is_set():
        if not buf_free[i].is_set():
            n_capture_waits += 1
        if not buf_free[i].wait(timeout=2.0):
            continue
        buf_free[i].clear()
        t1 = time.perf_counter()
        raw = rx.rx()
        t2 = time.perf_counter()
        slots[i] = np.asarray(raw, dtype="complex64")
        capture_us.append((t2 - t1) * 1e6)
        buf_ready[i].set()
        i ^= 1


print(f"[rx-pingpong] uri_rx={args.uri_rx} rate={args.rate/1e6:.1f}Msps rx_gain={args.rx_gain:+.1f}dB "
      f"rx_samples={RX_SAMPLES} stream_chunk={STREAM_CHUNK} budget={BUDGET_US:.0f}us "
      f"running for {args.seconds:.1f}s ...")

cap_thread = threading.Thread(target=capture_loop, daemon=True)
cap_thread.start()

decode_us = []
n_decoded = 0
n_crc_valid = 0
n_bytes_match = 0
n_bytes_mismatch = 0
n_buffers_seen = 0
n_decode_waits = 0

t_start = time.perf_counter()
t_end = float("inf") if args.seconds <= 0 else t_start + args.seconds
i = 0
try:
    while time.perf_counter() < t_end:
        if not buf_ready[i].is_set():
            n_decode_waits += 1
        if not buf_ready[i].wait(timeout=2.0):
            continue
        buf_ready[i].clear()
        buf = slots[i]
        t1 = time.perf_counter()
        results = []
        for start in range(0, len(buf), STREAM_CHUNK):
            piece = buf[start:start + STREAM_CHUNK]
            results.append(ofdm_rx.rx_streaming(piece[None, :]))
        t2 = time.perf_counter()
        decode_us.append((t2 - t1) * 1e6)
        n_buffers_seen += 1
        for result in results:
            if result is None:
                continue
            n_decoded += 1
            crc_ok = bool(np.asarray(result["crc_valid"])[0]) if result["crc_valid"] is not None else None
            if crc_ok:
                n_crc_valid += 1
            decoded_bits = np.asarray(result["bits"])[0].astype("uint8")
            decoded_bytes = np.packbits(decoded_bits).tobytes()
            match = decoded_bytes[:64] == EXPECTED_PAYLOAD_BYTES
            if match:
                n_bytes_match += 1
            else:
                n_bytes_mismatch += 1
            evm = float(np.asarray(result["evm"])[0]) if result["evm"] is not None else float("nan")
            print(f"  [decoded] #{n_decoded} crc_valid={crc_ok} bytes_match_expected={match} "
                  f"evm={evm:.5f} rssi_db={float(np.asarray(result['rssi_db'])[0]):.1f} "
                  f"first16={decoded_bytes[:16].hex()}")
        buf_free[i].set()
        i ^= 1
except KeyboardInterrupt:
    print(f"\n[rx-pingpong] stopped by user")

elapsed = time.perf_counter() - t_start
stop_event.set()
for e in buf_free:  # wake capture thread if it's blocked waiting on a slot
    e.set()
cap_thread.join(timeout=3.0)

# Explicit, deterministic teardown -- don't rely on the interpreter's own
# exit-time (cyclic) GC pass to destroy the libiio Buffer/Device/Context
# objects. The capture thread's closure keeps `rx` reachable in a way
# plain refcounting alone doesn't clear until a cyclic collection runs,
# and that collection can destroy objects in an arbitrary order -- e.g. a
# Buffer after its parent Context is already gone, a use-after-free in
# libiio's C code that segfaults (confirmed via PYTHONFAULTHANDLER:
# "Garbage-collecting" -> iio.py Buffer.__del__ -> _buffer_destroy).
# Drop every reference and collect here instead, while nothing else has
# started tearing down, so ordinary refcounting handles it safely.
del cap_thread
slots[0] = None
slots[1] = None
rx = None
gc.collect()

cap_arr = np.array(capture_us) if capture_us else np.array([0.0])
dec_arr = np.array(decode_us) if decode_us else np.array([0.0])
total_samples_seen = n_buffers_seen * RX_SAMPLES
coverage_pct = 100 * total_samples_seen / (elapsed * args.rate)
n_over_budget = int(np.sum(dec_arr > BUDGET_US))

print(f"\nbuffers decoded: {n_buffers_seen}  buffers captured: {len(cap_arr)}")
print(f"  capture had to wait for a free slot: {n_capture_waits}x   decode had to wait for fresh data: {n_decode_waits}x")
print(f"  decode alone over {BUDGET_US:.0f}us budget: {n_over_budget} ({100 * n_over_budget / max(n_buffers_seen, 1):.2f}%)")
print(f"  rx.rx() alone:   min={cap_arr.min():8.1f}us  median={np.median(cap_arr):8.1f}us  "
      f"mean={cap_arr.mean():8.1f}us  max={cap_arr.max():8.1f}us")
print(f"  rx_streaming():  min={dec_arr.min():8.1f}us  median={np.median(dec_arr):8.1f}us  "
      f"mean={dec_arr.mean():8.1f}us  max={dec_arr.max():8.1f}us")
print(f"  samples actually looked at: {total_samples_seen} of {int(elapsed*args.rate)} "
      f"({coverage_pct:.2f}% real-time coverage)")
print(f"  frames completed decode: {n_decoded}  crc_valid: {n_crc_valid}  "
      f"bytes exactly matched expected 64-byte payload: {n_bytes_match}  mismatched: {n_bytes_mismatch}")
