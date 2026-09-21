"""Full phase-2 matrix with interleaver2 on, against the committed
interleaver2-off numbers. Same seeds, same channel, 100 frames/cell."""
import sys, json, time; sys.path.insert(0, "examples")
import numpy as np
from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel
import dmrs_doppler_study as H

FS, SNR, PB = 20e6, 15.0, 5575
AMPS = (0.2, 0.4, 0.6, 0.8, 1.0)
DELAYS = (50, 100, 200, 500, 1000)
bits = np.random.default_rng(1).integers(0, 2, size=(1, PB*8)).astype("uint8")

def mk(il2, soft):
    o = Ofdm(fft_size=256, n_pilot=8, n_data=216, cp_len=64, modem="qam16",
             fec="rs_m8", fec1="conv_v27", interleaver="block",
             interleaver_kwargs={"unit_bits": 8}, crc="crc16",
             sync="schmidl_cox", cfo="schmidl_cox", n_training_symbols=2,
             dmrs_interval=32, soft_decision=soft, interleaver2=il2)
    o.MAX_PAYLOAD_SYMBOLS = 256
    return o

def run(a, d, il2, soft, n=100, seed0=9000):
    ok = 0
    with H.interval_ctx(32):
        for t in range(n):
            rng = np.random.default_rng(seed0+t)
            taps, dop, _ = Channel.paths_to_taps([
                {"amplitude": 1.0, "delay_ns": 0},
                {"amplitude": a, "delay_ns": d,
                 "phase_rad": float(rng.uniform(0, 2*np.pi))}], FS)
            o = mk(il2, soft)
            rx = Channel(snr_db=SNR, multipath_taps=taps, tap_doppler_hz=dop,
                         sample_rate_hz=FS, tail_samples=4096,
                         noise_draw_len=300_000, seed=seed0+t,
                         backend="numpy").process(o.generate_frame(bits))
            try:
                r = o.rx_process(rx); ok += int(bool(np.asarray(r["crc_valid"])[0]))
            except ValueError:
                pass
    return ok

rows = []
t0 = time.time()
for soft in (False, True):
    print(f"\n=== soft_decision={soft} === (of 100)")
    print("%7s |"%"a" + "".join("%18s"%("%dns"%d) for d in DELAYS))
    for a in AMPS:
        cells = []
        for d in DELAYS:
            off = run(a, d, "none", soft); on = run(a, d, "block", soft)
            cells.append((off, on)); rows.append(dict(a=a, delay_ns=d, soft=soft, off=off, on=on))
        print("%7.1f |"%a + "".join("%8d ->%6d "%(o, n) for o, n in cells), flush=True)
json.dump(rows, open("debug/freq_interleaver/matrix.json", "w"), indent=1)
print(f"\nwrote debug/freq_interleaver/matrix.json  [{time.time()-t0:.0f}s]")
