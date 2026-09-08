"""Self-interference sanity check: does THIS Pluto's own TX leak into its
OWN RX when both run at once (full-duplex FDD), at whatever tx-freq/rx-freq
gap you're planning to use for real AM-mode ACK traffic?

Single physical Pluto, one adi.Pluto object with BOTH tx_lo and rx_lo set
independently -- same "one Pluto, two LOs" pattern already used for real
full-duplex in examples/drone_tui/pluto.py's pluto_init() (verified against
that file directly, not assumed), just simplified here: no OFDM frames, no
Mac, just raw noise on TX and a raw RX power reading, because the only
question this script answers is "does turning MY OWN TX on measurably raise
MY OWN RX's noise floor" -- decoding real frames would only make that harder
to see clearly under everything else already investigated this session.

Method: transmits continuous band-filling complex noise (tx_cyclic_buffer=True
-- Pluto's own hardware repeats the DMA buffer autonomously, no Python loop
needed to "hold TX on") in alternating ON/OFF windows, while continuously
reading RX power the whole time. At the end, compares mean RX noise-floor
power during TX-ON windows vs TX-OFF windows -- if TX turning on doesn't
move RX's own noise floor, the frequency gap (+ antenna separation) you're
using is fine for a same-board full-duplex ACK channel. If it does, that's
direct, measured evidence of self-interference, not a guess from a spec
sheet.

Usage:
    python3 abhi/pluto_self_interference_test.py --uri ip:192.168.3.1 \\
        --rate 4e6 --rx-freq 5.860e9 --tx-freq 5.870e9 \\
        --tx-gain -10 --rx-gain 60 --on-seconds 3 --off-seconds 3 --cycles 5
"""
import argparse
import gc
import time

import numpy as np
import adi

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--uri", required=True)
ap.add_argument("--rate", type=float, default=4e6, help="Hz -- shared TX/RX sample rate")
ap.add_argument("--rx-freq", type=float, required=True, help="Hz -- this Pluto's own RX frequency")
ap.add_argument("--tx-freq", type=float, required=True, help="Hz -- this Pluto's own TX frequency (>=10MHz away from --rx-freq per this session's plan)")
ap.add_argument("--rx-gain", type=float, default=60.0, help="dB, manual RX gain (same convention as the other debug scripts)")
ap.add_argument("--tx-gain", type=float, default=-10.0, help="dB, PlutoSDR TX attenuation (0=max power) -- match whatever real power you'd actually run ACKs at")
ap.add_argument("--rx-samples", type=int, default=100_000, help="one rx() call's buffer size")
ap.add_argument("--on-seconds", type=float, default=3.0, help="how long TX stays on per cycle")
ap.add_argument("--off-seconds", type=float, default=3.0, help="how long TX stays off per cycle")
ap.add_argument("--cycles", type=int, default=5, help="number of on/off cycles")
args = ap.parse_args()

freq_gap_mhz = abs(args.tx_freq - args.rx_freq) / 1e6
rf_bw = int(max(args.rate * 1.25, 5e6))

print(f"[self-interference-test] uri={args.uri} rate={args.rate/1e6:.1f}Msps "
      f"rx_freq={args.rx_freq/1e9:.4f}GHz tx_freq={args.tx_freq/1e9:.4f}GHz "
      f"gap={freq_gap_mhz:.1f}MHz tx_gain={args.tx_gain:+.1f}dB rx_gain={args.rx_gain:+.1f}dB")

# Same "one Pluto, two independent LOs" full-duplex FDD setup as
# examples/drone_tui/pluto.py's pluto_init() (verified against that file),
# just with tx_cyclic_buffer=True instead of False -- this script wants
# continuous TX during each "on" window without a Python-side loop holding
# it there, unlike a real frame-by-frame TX which sends one shot per call.
sdr = adi.Pluto(uri=args.uri)
sdr.sample_rate = int(args.rate)
sdr.tx_lo = int(args.tx_freq)
sdr.tx_rf_bandwidth = rf_bw
sdr.tx_hardwaregain_chan0 = float(args.tx_gain)
sdr.tx_cyclic_buffer = True
sdr.rx_lo = int(args.rx_freq)
sdr.rx_rf_bandwidth = rf_bw
sdr.gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan0 = args.rx_gain
sdr.rx_buffer_size = args.rx_samples

for _ in range(10):  # PySDR's own recommended flush before real measurement, same as every other script this session
    sdr.rx()

# TX waveform: full-occupied-bandwidth complex noise, not a real OFDM frame
# -- this test only cares about "is there energy on the air at tx_freq",
# not what it says. peak=2**13 (12dB below full scale) matches this
# codebase's own established DAC-scaling convention (examples/pluto_channel.py's
# send_frame(), abhi/pluto_tx_standalone_test.py) -- reused for consistency,
# not because OFDM's own PAPR headroom specifically matters for noise.
_rng = np.random.RandomState(0)
_n_tx_samples = max(int(args.rate * 0.01), 1024)  # 10ms buffer; tx_cyclic_buffer=True repeats it in hardware indefinitely
_noise = (_rng.randn(_n_tx_samples) + 1j * _rng.randn(_n_tx_samples)).astype("complex64")
_noise *= (2 ** 13) / np.max(np.abs(_noise))


def _tx_on() -> None:
    sdr.tx(_noise)


def _tx_off() -> None:
    sdr.tx_destroy_buffer()


_tx_off()  # start OFF, deterministic known state

cycle_len = args.on_seconds + args.off_seconds
total_seconds = cycle_len * args.cycles
print(f"running {args.cycles} on/off cycles ({args.on_seconds:.1f}s on / {args.off_seconds:.1f}s off each, "
      f"{total_seconds:.0f}s total) ...")

readings_t = []
readings_power_db = []
readings_tx_on = []
readings_peak = []

tx_state = False
t_start = time.perf_counter()
while True:
    t = time.perf_counter() - t_start
    if t >= total_seconds:
        break
    phase_t = t % cycle_len
    should_be_on = phase_t < args.on_seconds
    if should_be_on != tx_state:
        (_tx_on if should_be_on else _tx_off)()
        tx_state = should_be_on

    buf = np.asarray(sdr.rx(), dtype="complex64")
    power_db = 10.0 * np.log10(np.mean(np.abs(buf) ** 2) + 1e-20)
    readings_t.append(t)
    readings_power_db.append(power_db)
    readings_tx_on.append(should_be_on)
    readings_peak.append(float(np.max(np.abs(buf))))
    print(f"  t={t:5.1f}s  tx={'ON ' if should_be_on else 'OFF'}  rx_power={power_db:6.2f}dB  rx_peak={readings_peak[-1]:8.1f}")

_tx_off()  # leave TX off when done, don't leave the board transmitting after the script exits

# Explicit, deterministic teardown -- same fix as abhi/pluto_rx_standalone_v2.py's
# own (already-proven) one, for the same root cause: keeping `sdr` reachable
# lets the interpreter's own cyclic-GC pass destroy libiio's Buffer/Context
# objects in an arbitrary order at exit, a use-after-free in libiio's C code
# that segfaults on interpreter exit. Drop the reference and collect here,
# before any of the result-printing below.
sdr = None
gc.collect()

power_db_arr = np.asarray(readings_power_db)
tx_on_arr = np.asarray(readings_tx_on)
peak_arr = np.asarray(readings_peak)

off_power = power_db_arr[~tx_on_arr]
on_power = power_db_arr[tx_on_arr]
off_peak = peak_arr[~tx_on_arr]
on_peak = peak_arr[tx_on_arr]

print(f"\n=== RESULTS ===")
if len(off_power) == 0 or len(on_power) == 0:
    print("Not enough samples in one phase or the other -- increase --on-seconds/--off-seconds/--cycles and rerun.")
else:
    print(f"RX noise floor, TX OFF: mean={off_power.mean():6.2f}dB  min={off_power.min():6.2f}  max={off_power.max():6.2f}  (n={len(off_power)})")
    print(f"RX noise floor, TX ON:  mean={on_power.mean():6.2f}dB  min={on_power.min():6.2f}  max={on_power.max():6.2f}  (n={len(on_power)})")
    delta = float(on_power.mean() - off_power.mean())
    print(f"Delta (TX-ON minus TX-OFF): {delta:+.2f}dB")
    print(f"RX peak sample magnitude, TX OFF: mean={off_peak.mean():8.1f}  max={off_peak.max():8.1f}")
    print(f"RX peak sample magnitude, TX ON:  mean={on_peak.mean():8.1f}  max={on_peak.max():8.1f}  "
          f"(compare to TX-OFF's own max, not an absolute full-scale claim -- if this is dramatically "
          f"higher, the RX front end may be saturating during TX-ON, which would make the dB numbers "
          f"above unreliable rather than reassuring)")

    if delta > 10:
        verdict = ("STRONG self-interference -- your own TX is heavily raising your own RX noise floor "
                   "at this frequency gap/power. More separation, lower TX power, or physical shielding/"
                   "antenna placement changes are needed before trusting an AM-mode ACK channel here.")
    elif delta > 3:
        verdict = ("NOTICEABLE self-interference -- some leakage is present. May still be fine for a "
                   "strong desired signal but could bury a weak one; worth a real link test (actual "
                   "OFDM decode with TX on) before committing to this gap/power for ACKs.")
    else:
        verdict = ("No significant self-interference detected -- RX noise floor barely moves when TX "
                   "turns on, at this frequency gap and TX power.")
    print(f"\nVERDICT: {verdict}")
