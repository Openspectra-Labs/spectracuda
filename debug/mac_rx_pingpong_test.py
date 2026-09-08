"""MAC-layer counterpart to pluto_rx_pingpong_test.py -- identical 2-slot
double-buffer capture/decode threading (see that script's module
docstring for the full reasoning), but the decode side now goes through
Mac(mode="um") instead of a bare Ofdm: real 32-bit PDU header decode
(TYPE/SI/SN/SO) and UM reassembly, feeding Mac's OWN internal
result-handling instead of a hand-rolled bit-compare.

The key move, per this session's conversation: Ofdm.rx_streaming()
returns the EXACT SAME result-dict shape as rx_process() (confirmed from
its own docstring) -- Mac already splits "get a result dict" from "act on
a result dict" into _rx_process_only()/_apply_rx_result()/_deliver()
(see mac.py, built that way specifically so receive_iq_batch() could
reuse the pieces on its own terms). Mac.receive_iq() happens to wire the
first piece to ofdm.rx_process() (batch), but nothing requires that --
swapping ONLY that one piece for ofdm.rx_streaming() and feeding its
result into the same two downstream methods is enough:

    result = mac.ofdm.rx_streaming(chunk)      # None, or the same dict rx_process() returns
    if result is not None:
        bits = mac._apply_rx_result(result)     # quality tracking + crc/frame_found check
        sdus = mac._deliver(bits)                # UM: [] or completed SDU(s) -- header stripped here

No custom dispatcher, no PDU-type routing -- that complexity only exists
for a real bind() handshake (BIND_RESPONSE vs DATA), which this link
deliberately skips (see mac_tx_standalone_test.py's module docstring):
every arriving frame is DATA, so _deliver() always routes to
UmEntity.receive().

Usage:
    python3 debug/mac_rx_pingpong_test.py --uri-rx ip:192.168.3.1 --rate 4e6 --seconds 0
"""
import argparse
import gc
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spectracuda.mac import Mac
from spectracuda.mac.pdu import HEADER_LEN_BITS, decode_header
from pluto_common import pluto_rx_init

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="rs_m8", fec1="conv_v27", crc="crc16",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
    backend="numpy",
)
RX_SAMPLES = 100_000
STREAM_CHUNK = 2048
# Matches debug/mac_tx_standalone_test.py's PAYLOAD_BYTES exactly.
EXPECTED_PAYLOAD_BYTES = bytes(range(64))

ap = argparse.ArgumentParser()
ap.add_argument("--uri-rx", required=True)
ap.add_argument("--freq", type=float, default=2.425e9)
ap.add_argument("--rate", type=float, default=4e6)
ap.add_argument("--rx-gain", type=float, default=60.0)
ap.add_argument("--seconds", type=float, default=10.0, help="0 = run until Ctrl-C")
args = ap.parse_args()

rx = pluto_rx_init(args.uri_rx, args.freq, args.rate, args.rx_gain, RX_SAMPLES)

mac = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
mac.bound = True  # see mac_tx_standalone_test.py's module docstring -- no real bind on this one-way link
mac.ofdm.reset_stream()

for _ in range(10):
    rx.rx()

BUDGET_US = RX_SAMPLES / args.rate * 1e6

# Exactly 2 slots -- see pluto_rx_pingpong_test.py for the full reasoning
# (strict 1-buffer bound, not a general N-deep queue).
slots = [None, None]
buf_ready = [threading.Event(), threading.Event()]
buf_free = [threading.Event(), threading.Event()]
for e in buf_free:
    e.set()
stop_event = threading.Event()

capture_us = []
n_capture_waits = 0


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


print(f"[mac-rx] uri_rx={args.uri_rx} rate={args.rate/1e6:.1f}Msps rx_gain={args.rx_gain:+.1f}dB "
      f"rx_samples={RX_SAMPLES} stream_chunk={STREAM_CHUNK} budget={BUDGET_US:.0f}us "
      f"max_segment_bits={mac.max_segment_bits} -- listening via Mac(mode='um')/rx_streaming()...")

cap_thread = threading.Thread(target=capture_loop, daemon=True)
cap_thread.start()

decode_us = []
n_pdus_decoded = 0
n_sdus_completed = 0
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
        for start in range(0, len(buf), STREAM_CHUNK):
            piece = buf[start:start + STREAM_CHUNK]
            result = mac.ofdm.rx_streaming(piece[None, :])  # <-- the ONE swapped piece: streaming, not batch
            if result is None:
                continue
            bits = mac._apply_rx_result(result)  # quality tracking + crc/frame_found check -> PDU bits or None
            if bits is None:
                continue
            n_pdus_decoded += 1
            header = decode_header(np.asarray(bits)[:HEADER_LEN_BITS].astype("uint8"))
            sdus = mac._deliver(bits)  # UM: [] or completed SDU(s), header already stripped
            for sdu_bits in sdus:
                n_sdus_completed += 1
                decoded_bytes = np.packbits(np.asarray(sdu_bits).astype("uint8")).tobytes()
                match = decoded_bytes[:64] == EXPECTED_PAYLOAD_BYTES
                if match:
                    n_bytes_match += 1
                else:
                    n_bytes_mismatch += 1
                print(f"  [sdu] #{n_sdus_completed} pdu_type={header['pdu_type']} si={header['si']} "
                      f"sn={header['sn']} so={header['so']} bytes_match_expected={match} "
                      f"first16={decoded_bytes[:16].hex()}")
        t2 = time.perf_counter()
        decode_us.append((t2 - t1) * 1e6)
        n_buffers_seen += 1
        buf_free[i].set()
        i ^= 1
except KeyboardInterrupt:
    print(f"\n[mac-rx] stopped by user")

elapsed = time.perf_counter() - t_start
stop_event.set()
for e in buf_free:
    e.set()
cap_thread.join(timeout=3.0)

# Explicit, deterministic teardown -- same libiio Buffer.__del__ shutdown
# segfault fix as pluto_rx_pingpong_test.py (see that file for the full
# PYTHONFAULTHANDLER-confirmed root cause).
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
print(f"  rx_streaming(): min={dec_arr.min():8.1f}us  median={np.median(dec_arr):8.1f}us  "
      f"mean={dec_arr.mean():8.1f}us  max={dec_arr.max():8.1f}us")
print(f"  samples actually looked at: {total_samples_seen} of {int(elapsed*args.rate)} "
      f"({coverage_pct:.2f}% real-time coverage)")
print(f"  PDUs decoded (CRC-valid, post-FEC): {n_pdus_decoded}  SDUs completed: {n_sdus_completed}  "
      f"bytes exactly matched expected payload: {n_bytes_match}  mismatched: {n_bytes_mismatch}")
print(f"  mac.quality: {mac.quality.report_dict()}")
