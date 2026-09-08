"""AM-mode two-node real-hardware test -- NODE B (receiver role for the
DATA direction, sender role for the STATUS/ACK direction). Run this on
the OTHER Pi5+Pluto rig from abhi/mac_am_node_a_test.py, with --tx-freq/
--rx-freq SWAPPED relative to node A's (node A's --tx-freq = this node's
--rx-freq, and vice versa). See that file's own module docstring for the
full design writeup (why two scripts, the exact API calls used and why,
what was verified before writing this) -- not repeated here.

This node's own API usage, the STATUS-direction counterpart to node A's:
  - mac.ofdm.rx_streaming(chunk) + mac._apply_rx_result(result) -> CRC-
    valid PDU bits or None, identical pattern to every real-hardware RX
    script this session.
  - decode_header(bits)["pdu_type"] == TYPE_DATA check, then
    mac._deliver(bits) -> [] or completed SDU(s) with the MAC header
    already stripped (am.py: _deliver() routes to AmEntity.receive_data()
    for mode="am", confirmed by reading mac.py's _deliver() body directly,
    not assumed from UM's own behavior).
  - mac.build_status() -> this node's OWN outgoing STATUS pdu bits
    (reports which SNs it has received, for node A's receive_status() to
    act on) -> mac.ofdm.generate_frame() -> sent back to node A, sent
    periodically (--status-interval) rather than once per received PDU,
    same "batch the ACKs" precedent as real RLC/802.11 Block-ACK.

bind() skipped, same documented reason/precedent as node A's script.

Usage (frequencies SWAPPED relative to node A):
    python3 abhi/mac_am_node_b_test.py --uri ip:192.168.2.1 --rate 4e6 \\
        --tx-freq 5.870e9 --rx-freq 5.860e9 --tx-gain -10 --rx-gain 60 \\
        --status-interval 0.2 --seconds 30
"""
import argparse
import gc
import queue
import threading
import time

import numpy as np
import adi

from spectracuda.mac import Mac
from spectracuda.mac.pdu import HEADER_LEN_BITS, TYPE_DATA, decode_header

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="rs_m8", fec1="conv_v27", crc="crc16",
    interleaver="block", interleaver_kwargs={"unit_bits": 8},  # MUST match node A exactly -- see abhi/pluto_rx_standalone_v2.py's own PHY_KWARGS comment for why
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
    backend="numpy",
)
RX_SAMPLES = 100_000
STREAM_CHUNK = 1024  # see node A's own comment -- NOT 2048, boundary bug

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--uri", required=True)
ap.add_argument("--rate", type=float, default=4e6, help="Hz -- must equal node A's --rate")
ap.add_argument("--tx-freq", type=float, required=True, help="Hz -- this node's own TX (= node A's --rx-freq)")
ap.add_argument("--rx-freq", type=float, required=True, help="Hz -- this node's own RX (= node A's --tx-freq)")
ap.add_argument("--tx-gain", type=float, default=-10.0, help="dB, PlutoSDR TX attenuation (0=max power)")
ap.add_argument("--rx-gain", type=float, default=60.0, help="dB, manual RX gain")
ap.add_argument("--status-interval", type=float, default=0.2, help="seconds between STATUS sends (default 200ms)")
ap.add_argument("--seconds", type=float, default=30.0, help="0 = run until Ctrl-C")
ap.add_argument("--queue-size", type=int, default=1024)
args = ap.parse_args()

rf_bw = int(max(args.rate * 1.25, 5e6))

sdr = adi.Pluto(uri=args.uri)
sdr.sample_rate = int(args.rate)
sdr.tx_lo = int(args.tx_freq)
sdr.tx_rf_bandwidth = rf_bw
sdr.tx_hardwaregain_chan0 = float(args.tx_gain)
sdr.tx_cyclic_buffer = False
sdr.rx_lo = int(args.rx_freq)
sdr.rx_rf_bandwidth = rf_bw
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = args.rx_gain
sdr.rx_buffer_size = RX_SAMPLES

mac = Mac(mode="am", ofdm_kwargs=PHY_KWARGS)
mac.bound = True  # see node A's / debug/mac_tx_standalone_test.py's module docstring
mac.ofdm.reset_stream()

for _ in range(10):
    sdr.rx()

print(f"[mac-am-node-b] uri={args.uri} rate={args.rate/1e6:.1f}Msps tx_freq={args.tx_freq/1e9:.4f}GHz "
      f"rx_freq={args.rx_freq/1e9:.4f}GHz gap={abs(args.tx_freq-args.rx_freq)/1e6:.1f}MHz "
      f"max_segment_bits={mac.max_segment_bits} status_interval={args.status_interval*1000:.0f}ms")


def _to_iq_and_send(iq_frame) -> None:
    samples = np.asarray(iq_frame)
    if samples.ndim == 2:
        samples = samples[0]
    peak = np.max(np.abs(samples))
    scaled = (samples * (2 ** 13 / peak)).astype("complex64") if peak > 1e-12 else (samples * 0).astype("complex64")
    sdr.tx(scaled)


chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=args.queue_size)
stop_event = threading.Event()


def _rx_loop() -> None:
    while not stop_event.is_set():
        buf = np.asarray(sdr.rx(), dtype="complex64")
        for start in range(0, len(buf), STREAM_CHUNK):
            chunk = buf[start:start + STREAM_CHUNK]
            if len(chunk) == 0:
                continue
            try:
                chunk_queue.put_nowait(chunk)
            except queue.Full:
                try:
                    chunk_queue.get_nowait()
                except queue.Empty:
                    pass
                chunk_queue.put_nowait(chunk)


reader_thread = threading.Thread(target=_rx_loop, daemon=True)
reader_thread.start()

n_pdus_decoded = 0
n_sdus_completed = 0
n_gaps_detected = 0
last_counter_seen = None
next_status_t = time.perf_counter()
t_start = time.perf_counter()
t_end = float("inf") if args.seconds <= 0 else t_start + args.seconds

try:
    while time.perf_counter() < t_end:
        now = time.perf_counter()

        # 1) drain RX -- this is the DATA direction (node A's send_iq() output).
        drained_any = False
        while True:
            try:
                chunk = chunk_queue.get_nowait()
            except queue.Empty:
                break
            drained_any = True
            result = mac.ofdm.rx_streaming(chunk[None, :])
            if result is None:
                continue
            bits = mac._apply_rx_result(result)
            if bits is None:
                continue
            bits = np.asarray(bits).astype("uint8")
            header = decode_header(bits[:HEADER_LEN_BITS])
            if header["pdu_type"] != TYPE_DATA:
                continue  # not expected on this node's RX in this one-direction test -- ignore rather than crash
            n_pdus_decoded += 1
            sdus = mac._deliver(bits)  # AM: _deliver() routes to receive_data() for mode="am" -- [] or completed SDU(s)
            for sdu_bits in sdus:
                n_sdus_completed += 1
                decoded_bytes = np.packbits(np.asarray(sdu_bits).astype("uint8")).tobytes()
                counter = int.from_bytes(decoded_bytes[:4], "big")
                gap = ""
                if last_counter_seen is not None and counter != last_counter_seen + 1:
                    n_gaps_detected += 1
                    gap = f"  <-- GAP (expected {last_counter_seen + 1})"
                last_counter_seen = counter
                print(f"  [sdu] #{n_sdus_completed} counter={counter} pdu_type={header['pdu_type']} "
                      f"si={header['si']} sn={header['sn']} so={header['so']} "
                      f"first16={decoded_bytes[:16].hex()}{gap}")

        # 2) send a STATUS report periodically -- batches ACK/NACK info
        # rather than one STATUS per received PDU (real RLC/Block-ACK precedent).
        if now >= next_status_t:
            status_bits = mac.build_status()
            iq = mac.ofdm.generate_frame(np.asarray(status_bits, dtype="uint8")[None, :])
            _to_iq_and_send(iq)
            next_status_t += args.status_interval

        if not drained_any:
            time.sleep(0.001)
except KeyboardInterrupt:
    print(f"\n[node-b] stopped by user")

stop_event.set()
reader_thread.join(timeout=5.0)

del reader_thread
chunk_queue = None
sdr = None
gc.collect()

print(f"\n=== SUMMARY ===")
print(f"PDUs decoded (CRC-valid, post-FEC): {n_pdus_decoded}  SDUs completed: {n_sdus_completed}  "
      f"order gaps detected: {n_gaps_detected}  last counter seen: {last_counter_seen}")
print(f"mac.quality: {mac.quality.report_dict()}")
