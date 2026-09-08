"""AM-mode two-node real-hardware test -- NODE A (sender role for the DATA
direction, receiver role for the STATUS/ACK direction). Run this on one
Pi5+Pluto rig; copy abhi/mac_am_node_b_test.py to the other rig and run it
there with --tx-freq/--rx-freq SWAPPED relative to this node's.

Why two scripts, not one bidirectional script: Mac(mode="am")'s own class
docstring is explicit that a full-duplex link needs ONE AmEntity per
DIRECTION per endpoint (spectracuda/mac/am.py's own docstring: "a
full-duplex link needs one AmEntity per direction per endpoint, wired
together by MacLink ... not a single shared instance") -- this test
deliberately exercises ONE direction of real ARQ traffic (A sends DATA,
B ACKs/NACKs back to A) rather than building the full 4-Mac/4-Ofdm
bidirectional wiring docs/mac.md describes, since the actual question
right now is "does AM's retransmission mechanism work over a real,
self-interference-checked full-duplex RF link" -- not the fuller
bidirectional-data-both-ways design. Confirmed both Ofdm's own CPE-drift
fix and the 10MHz TX/RX frequency gap this session's own
pluto_self_interference_test.py measured (~0.0-0.06dB delta, no self-
interference at either -10dBm or 0dBm) before writing this, not assumed.

API used, verified directly against spectracuda/mac/mac.py and am.py
(not guessed):
  - mac.send_iq(sdu_bits) -> list of IQ frames for a NEW SDU (requires
    mac.bound; see below).
  - mac.ofdm.rx_streaming(chunk) -> a rx_process()-shaped result dict or
    None, exactly like every other real-hardware RX script this session.
  - mac._apply_rx_result(result) -> decoded PDU bits (CRC-valid) or None
    -- PDU-type-agnostic (works for STATUS just as well as DATA; verified
    by reading its body directly).
  - decode_header(bits)["pdu_type"] used to recognize a STATUS pdu
    arriving on THIS node's own RX chain (the peer's build_status()
    output).
  - mac.receive_status(status_pdu_bits) -> list of PDU-bit-arrays that
    need retransmitting THIS round (am.py's own docstring) -- each is
    turned back into IQ via mac.ofdm.generate_frame() and resent, exactly
    like a fresh send_iq() PDU.

bind() deliberately skipped, same documented precedent as
debug/mac_tx_standalone_test.py: Mac.send_iq() requires self.bound, a
real bind() needs its own BIND_REQUEST/BIND_RESPONSE round trip this
first test isn't trying to prove yet. `mac.bound = True` forced directly,
same as that file. Worth doing for real once ARQ itself is confirmed
working.

Usage (frequencies must be SWAPPED on node B):
    python3 abhi/mac_am_node_a_test.py --uri ip:192.168.3.1 --rate 4e6 \\
        --tx-freq 5.860e9 --rx-freq 5.870e9 --tx-gain -10 --rx-gain 60 \\
        --interval 0.1 --seconds 30
"""
import argparse
import gc
import queue
import threading
import time

import numpy as np
import adi

from spectracuda.mac import Mac
from spectracuda.mac.pdu import HEADER_LEN_BITS, TYPE_STATUS, decode_header

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="rs_m8", fec1="conv_v27", crc="crc16",
    interleaver="block", interleaver_kwargs={"unit_bits": 8},  # MUST match node B exactly -- see abhi/pluto_rx_standalone_v2.py's own PHY_KWARGS comment for why
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
    backend="numpy",
)
RX_SAMPLES = 100_000
STREAM_CHUNK = 1024  # MUST be < STREAM_SEARCH_WINDOW_SYMBOLS(8)*fft_size(256)=2048 -- see
# abhi/pluto_rx_standalone_v2.py's own comment for the full boundary-bug writeup; NOT 2048.

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--uri", required=True)
ap.add_argument("--rate", type=float, default=4e6, help="Hz -- must equal node B's --rate")
ap.add_argument("--tx-freq", type=float, required=True, help="Hz -- this node's own TX (= node B's --rx-freq)")
ap.add_argument("--rx-freq", type=float, required=True, help="Hz -- this node's own RX (= node B's --tx-freq)")
ap.add_argument("--tx-gain", type=float, default=-10.0, help="dB, PlutoSDR TX attenuation (0=max power)")
ap.add_argument("--rx-gain", type=float, default=60.0, help="dB, manual RX gain")
ap.add_argument("--interval", type=float, default=0.1, help="seconds between new SDUs (default 100ms)")
ap.add_argument("--payload-bytes", type=int, default=64, help="SDU size in bytes (must fit in one segment -- max_segment_bits/8; 1024B is well within that)")
ap.add_argument("--seconds", type=float, default=30.0, help="0 = run until Ctrl-C")
ap.add_argument("--queue-size", type=int, default=1024)
args = ap.parse_args()

rf_bw = int(max(args.rate * 1.25, 5e6))

# One physical Pluto, two independent LOs -- same full-duplex FDD pattern
# already verified real (examples/drone_tui/pluto.py's pluto_init(), and
# this session's own pluto_self_interference_test.py measuring ~0dB
# leakage at this exact kind of frequency gap) -- not a new/unproven setup.
sdr = adi.Pluto(uri=args.uri)
sdr.sample_rate = int(args.rate)
sdr.tx_lo = int(args.tx_freq)
sdr.tx_rf_bandwidth = rf_bw
sdr.tx_hardwaregain_chan0 = float(args.tx_gain)
sdr.tx_cyclic_buffer = False  # one-shot burst per tx() call, matches every TX script this session
sdr.rx_lo = int(args.rx_freq)
sdr.rx_rf_bandwidth = rf_bw
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = args.rx_gain
sdr.rx_buffer_size = RX_SAMPLES

mac = Mac(mode="am", ofdm_kwargs=PHY_KWARGS)
mac.bound = True  # see module docstring -- real bind() not attempted in this first ARQ-focused test
mac.ofdm.reset_stream()

for _ in range(10):  # PySDR's own recommended flush before real measurement, same as every other script this session
    sdr.rx()

print(f"[mac-am-node-a] uri={args.uri} rate={args.rate/1e6:.1f}Msps tx_freq={args.tx_freq/1e9:.4f}GHz "
      f"rx_freq={args.rx_freq/1e9:.4f}GHz gap={abs(args.tx_freq-args.rx_freq)/1e6:.1f}MHz "
      f"max_segment_bits={mac.max_segment_bits} interval={args.interval*1000:.0f}ms")


def _to_iq_and_send(iq_frame) -> None:
    """peak=2**13 (12dB below full scale) -- same DAC-scaling convention as
    every other TX script this session (examples/pluto_channel.py's
    send_frame(), debug/mac_tx_standalone_test.py)."""
    samples = np.asarray(iq_frame)
    if samples.ndim == 2:
        samples = samples[0]
    peak = np.max(np.abs(samples))
    scaled = (samples * (2 ** 13 / peak)).astype("complex64") if peak > 1e-12 else (samples * 0).astype("complex64")
    sdr.tx(scaled)


# -- reader thread: drains the ADC into a queue, never blocked by decode --
# same proven pattern as abhi/pluto_rx_standalone_v2.py's own _rx_loop().
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

n_sent = 0
n_retransmitted = 0
n_status_seen = 0
next_send_t = time.perf_counter()
t_start = time.perf_counter()
t_end = float("inf") if args.seconds <= 0 else t_start + args.seconds

try:
    while time.perf_counter() < t_end:
        now = time.perf_counter()

        # 1) send a new SDU if it's due -- payload = 4-byte big-endian
        # counter + filler, so node B can print exactly what arrived and
        # in what order (visible proof of correct/complete delivery, not
        # just a CRC-valid count).
        if now >= next_send_t:
            payload = n_sent.to_bytes(4, "big") + bytes((n_sent + i) % 256 for i in range(args.payload_bytes - 4))
            sdu_bits = np.unpackbits(np.frombuffer(payload, dtype="uint8"))
            pdus_iq = mac.send_iq(sdu_bits)
            for iq in pdus_iq:
                _to_iq_and_send(iq)
            n_sent += 1
            next_send_t += args.interval
            if n_sent % 20 == 0 or n_sent <= 3:
                print(f"[node-a] sent SDU #{n_sent}  pending_acks={len(mac.pending_pdus)}  "
                      f"retransmitted_so_far={n_retransmitted}")

        # 2) drain whatever RX chunks are available -- this is the STATUS
        # direction (node B's build_status() output arriving here).
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
            bits = mac._apply_rx_result(result)  # CRC-valid PDU bits, or None -- PDU-type-agnostic
            if bits is None:
                continue
            bits = np.asarray(bits).astype("uint8")
            header = decode_header(bits[:HEADER_LEN_BITS])
            if header["pdu_type"] != TYPE_STATUS:
                continue  # not expected on this node's RX in this one-direction test -- ignore rather than crash
            n_status_seen += 1
            to_retransmit = mac.receive_status(bits)
            for pdu_bits in to_retransmit:
                iq = mac.ofdm.generate_frame(np.asarray(pdu_bits, dtype="uint8")[None, :])
                _to_iq_and_send(iq)
                n_retransmitted += 1
            if n_status_seen <= 5 or n_status_seen % 20 == 0:
                print(f"[node-a] status #{n_status_seen} base_sn={header['sn']}  "
                      f"retransmitting_now={len(to_retransmit)}  failed_sns={sorted(mac.failed_sns)}")

        if not drained_any:
            time.sleep(0.001)  # avoid a hot spin-loop when nothing's queued yet
except KeyboardInterrupt:
    print(f"\n[node-a] stopped by user")

stop_event.set()
reader_thread.join(timeout=5.0)

# Explicit, deterministic teardown -- same fix as every other real-hardware
# script this session (libiio Buffer/Context destroy-order segfault on
# interpreter exit otherwise -- see abhi/pluto_rx_standalone_v2.py's own
# comment for the full root cause).
del reader_thread
chunk_queue = None
sdr = None
gc.collect()

print(f"\n=== SUMMARY ===")
print(f"SDUs sent: {n_sent}  STATUS pdus seen: {n_status_seen}  PDUs retransmitted: {n_retransmitted}  "
      f"still pending ack at exit: {len(mac.pending_pdus)}  permanently failed (max_retries exceeded): {sorted(mac.failed_sns)}")
