"""MAC-layer counterpart to pluto_tx_standalone_test.py -- same fixed
64-byte payload, same PHY_KWARGS, same one-way hardware topology (this
runs on the transmitting Pi5+Pluto, debug/mac_rx_pingpong_test.py on the
receiving one), but the SDU now goes through Mac(mode="um") instead of
Ofdm.generate_frame() directly: real 32-bit PDU header (TYPE/SI/SN/SO),
segmentation (a no-op here -- 512 bits fits in one segment given this
PHY's capacity), all real MAC framing, not just raw payload bits.

bind() deliberately SKIPPED, not built: Mac.send_iq() requires
self.bound, and a real bind() is a genuine bidirectional BIND_REQUEST/
BIND_RESPONSE exchange -- our hardware topology is one-way (this Pi never
listens, the RX Pi never transmits), so there's no return path for a
BIND_RESPONSE to travel on. `mac.bound = True` is forced directly instead
(see conversation this was decided in) -- exercises UM's real
segmentation/SN/reassembly framing without inventing a fake handshake
neither side can actually complete.

Each call to send_iq() increments the entity's own SN, so repeated sends
of the SAME payload are still genuinely distinct PDUs on the wire (real
MAC behavior, not just a repeated raw frame) -- unlike the raw-Ofdm TX
script, the frame is regenerated every send here, not built once and
replayed.

Usage:
    python3 debug/mac_tx_standalone_test.py --uri ip:192.168.3.1 --rate 4e6
"""
import argparse
import time

import numpy as np

from spectracuda.mac import Mac
from pluto_common import pluto_tx_init

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="rs_m8", fec1="conv_v27", crc="crc16",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
    backend="numpy",
)
PAYLOAD_BYTES = bytes(range(64))  # fixed 64-byte SDU, 0x00..0x3F -- matches the raw-Ofdm test's payload

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--uri", required=True)
ap.add_argument("--freq", type=float, default=2.425e9, help="Hz -- must equal the RX side's --freq")
ap.add_argument("--rate", type=float, default=4e6, help="Hz -- must equal the RX side's --rate")
ap.add_argument("--tx-gain", type=float, default=-10.0, help="dB, PlutoSDR attenuation (0=max power)")
ap.add_argument("--interval", type=float, default=0.1, help="seconds between sends (default 100ms)")
ap.add_argument("--count", type=int, default=0, help="number of SDUs to send, 0=run until Ctrl-C")
args = ap.parse_args()

sdu_bits = np.unpackbits(np.frombuffer(PAYLOAD_BYTES, dtype="uint8"))

# -- build the Mac, skip the (unreachable, one-way-topology) bind handshake --
mac = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
mac.bound = True  # see module docstring -- no real bind() possible on this one-way link
print(f"[mac-tx] max_segment_bits={mac.max_segment_bits}  SDU={len(sdu_bits)} bits "
      f"+ 32-bit header -- expect 1 segment/PDU per SDU")

# -- configure the real Pluto TX chain (same as pluto_tx_standalone_test.py) --
sdr = pluto_tx_init(args.uri, args.freq, args.rate, args.tx_gain)

print(f"[mac-tx] uri={args.uri} freq={args.freq/1e9:.4f}GHz rate={args.rate/1e6:.1f}Msps "
      f"tx_gain={args.tx_gain:+.1f}dB interval={args.interval*1000:.0f}ms count={args.count or 'inf'} "
      f"-- sending fixed 64-byte SDU via Mac(mode='um').send_iq()...")

n_sent = 0
tx_call_us = []
next_t = time.perf_counter()
try:
    while args.count == 0 or n_sent < args.count:
        now = time.perf_counter()
        sleep_for = next_t - now
        if sleep_for > 0:
            time.sleep(sleep_for)

        sn_before = mac._next_tx_sn
        pdus_iq = mac.send_iq(sdu_bits)  # fresh SN each call -- a genuinely new PDU, not a replay

        t1 = time.perf_counter()
        for iq in pdus_iq:
            samples = np.asarray(iq)[0]
            peak = np.max(np.abs(samples))
            scaled = (samples * (2 ** 13 / peak)).astype("complex64") if peak > 1e-12 else (samples * 0).astype("complex64")
            sdr.tx(scaled)
        t2 = time.perf_counter()
        tx_call_us.append((t2 - t1) * 1e6)
        n_sent += 1
        next_t += args.interval

        if n_sent % 50 == 0 or n_sent <= 3:
            arr = np.array(tx_call_us[-50:])
            print(f"[mac-tx] sent SDU #{n_sent}  sn={sn_before}  pdus={len(pdus_iq)}  "
                  f"tx() last{len(arr)}: median={np.median(arr):.1f}us max={arr.max():.1f}us")
except KeyboardInterrupt:
    print(f"\n[mac-tx] stopped by user after {n_sent} SDUs sent")

if tx_call_us:
    arr = np.array(tx_call_us)
    print(f"[mac-tx] done. {n_sent} SDUs sent. tx() call time: median={np.median(arr):.1f}us "
          f"mean={arr.mean():.1f}us max={arr.max():.1f}us")
