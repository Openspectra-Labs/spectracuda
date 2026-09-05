"""TX-only counterpart to pluto_rx_pingpong_test.py / pluto_rx_standalone_
test.py -- runs on the OTHER physical Pi5+Pluto (2 Pi + 2 Pluto setup, one
node transmits, the other receives; this is a one-way simplex link, both
ends on the SAME center frequency, unlike pluto_channel.py's full-duplex
FDD air/ground design). Must use the exact same PHY_KWARGS as the RX side
-- Ofdm's own header carries mod_scheme/fec0/fec1/crc dynamically resolved
from the wire (see spectracuda/pipeline/ofdm.py's class docstring), but
fft_size/n_pilot/n_data/cp_len/sync/cfo/channel_estimator/equalizer are
NOT signaled over the air and must already match out-of-band on both ends.

fec/fec1 set to rs_m8/conv_v27 here to match the RX side's own PHY_KWARGS
(debug/pluto_rx_pingpong_test.py, debug/pluto_rx_standalone_test_v2.py) --
the RX side runs with strict_fec_check=True and its own PHY_KWARGS comment
says it "only ever expects rs_m8/conv_v27", so a frame sent with fec=none/
fec1=none (this script's ORIGINAL setting) gets rejected by that strict
check even though the header-resolved scheme is otherwise decodable --
confirmed as the root cause of a ~22-34% "frames just never decoded"
pattern on a real two-Pi5-Pluto RF link during this project's own 2026-09
debug session (RSSI/EVM on the frames that DID get through were healthy,
ruling out a weak-signal explanation).

Generates ONE fixed 64-byte payload (0x00..0x3F, easy to eyeball-verify on
decode) ONCE via generate_frame(), then just re-sends that same waveform
every --interval seconds -- no need to re-encode a buffer that never
changes. Verified round-trip losslessly in an ideal (no-channel) test
before this script was written (crc_valid=True, bits match exactly, 4
payload OFDM symbols at this fec/n_data/modem).

Scaling (peak -> 2**13, i.e. 12dB below full scale, room for OFDM PAPR)
and tx_cyclic_buffer=False (one-shot burst per tx() call, not a repeating
cyclic buffer) match examples/pluto_channel.py's own send_frame() --
reused, not re-derived.

Usage:
    python3 debug/pluto_tx_standalone_test.py --uri ip:192.168.3.1 --rate 4e6
"""
import argparse
import time

import numpy as np
import adi

from spectracuda.pipeline import Ofdm

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="rs_m8", fec1="conv_v27", crc="crc16",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
    backend="numpy",
)
PAYLOAD_BYTES = bytes(range(64))  # fixed 64-byte buffer, 0x00..0x3F

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--uri", required=True)
ap.add_argument("--freq", type=float, default=2.425e9, help="Hz -- must equal the RX side's --freq")
ap.add_argument("--rate", type=float, default=4e6, help="Hz -- must equal the RX side's --rate")
ap.add_argument("--tx-gain", type=float, default=-10.0, help="dB, PlutoSDR attenuation (0=max power); tune for your link distance")
ap.add_argument("--interval", type=float, default=0.1, help="seconds between sends (default 100ms)")
ap.add_argument("--count", type=int, default=0, help="number of frames to send, 0=run until Ctrl-C")
args = ap.parse_args()

# -- build + verify the frame in an ideal loopback before ever touching hardware --
ofdm = Ofdm(**PHY_KWARGS)
payload_bits = np.unpackbits(np.frombuffer(PAYLOAD_BYTES, dtype="uint8"))[None, :]
tx_iq = ofdm.generate_frame(payload_bits)

_check = Ofdm(**PHY_KWARGS)  # separate object -- simulates "a separate device" per Ofdm's own class docstring
_result = _check.rx_process(tx_iq)
if not _result["frame_found"] or not bool(np.asarray(_result["crc_valid"])[0]):
    raise RuntimeError(
        f"self-check failed BEFORE touching hardware: frame_found={_result['frame_found']} "
        f"crc_valid={_result['crc_valid']} -- refusing to transmit a frame that doesn't even "
        f"decode in an ideal (no-channel) loopback"
    )
print(f"[tx] self-check OK: {len(PAYLOAD_BYTES)}-byte fixed payload round-trips losslessly "
      f"(n_payload_symbols={_result['n_payload_symbols']}, evm={float(np.asarray(_result['evm'])[0]):.5f})")

# -- scale for real DAC output, matching examples/pluto_channel.py's send_frame() --
samples = np.asarray(tx_iq)[0]
peak = np.max(np.abs(samples))
scaled = (samples * (2 ** 13 / peak)).astype("complex64")
print(f"[tx] frame: {len(scaled)} samples, peak|iq| pre-scale={peak:.4f}, scaled peak={np.max(np.abs(scaled)):.1f} "
      f"(int16 full scale=32767)")

# -- configure the real Pluto TX chain --
rf_bw = int(max(args.rate * 1.25, 5e6))
sdr = adi.Pluto(uri=args.uri)
sdr.sample_rate = int(args.rate)
sdr.tx_lo = int(args.freq)
sdr.tx_rf_bandwidth = rf_bw
sdr.tx_hardwaregain_chan0 = float(args.tx_gain)
sdr.tx_cyclic_buffer = False  # one-shot burst per tx() call, not a repeating cyclic buffer

print(f"[tx] uri={args.uri} freq={args.freq/1e9:.4f}GHz rate={args.rate/1e6:.1f}Msps tx_gain={args.tx_gain:+.1f}dB "
      f"interval={args.interval*1000:.0f}ms count={args.count or 'inf'} -- sending fixed 64-byte payload...")

n_sent = 0
tx_call_us = []
next_t = time.perf_counter()
try:
    while args.count == 0 or n_sent < args.count:
        now = time.perf_counter()
        sleep_for = next_t - now
        if sleep_for > 0:
            time.sleep(sleep_for)
        t1 = time.perf_counter()
        sdr.tx(scaled)
        t2 = time.perf_counter()
        tx_call_us.append((t2 - t1) * 1e6)
        n_sent += 1
        next_t += args.interval
        if n_sent % 50 == 0:
            arr = np.array(tx_call_us[-50:])
            print(f"[tx] sent {n_sent} frames  tx() last50: median={np.median(arr):.1f}us max={arr.max():.1f}us")
except KeyboardInterrupt:
    print(f"\n[tx] stopped by user after {n_sent} frames")

if tx_call_us:
    arr = np.array(tx_call_us)
    print(f"[tx] done. {n_sent} frames sent. tx() call time: median={np.median(arr):.1f}us "
          f"mean={arr.mean():.1f}us max={arr.max():.1f}us")
