"""thresh_wq (the 2-BRAM table, exactly as the FPGA will compute it) vs
full-precision thresh_w: frame error rate on the same frames/channels,
100 frames per point."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel

def mk(mod, metric):
    return Ofdm(fft_size=256, cp_len=32, n_data=216, n_pilot=8, modem=mod, fec1="conv_v27",
                interleaver="block", interleaver_kwargs={"unit_bits": 8}, soft_llr_bits=4,
                soft_llr_metric=metric)

def one(a):
    mod, snr, taps, t = a
    b = np.random.default_rng(t).integers(0, 2, (1, 8000)).astype("uint8")
    f = np.asarray(mk(mod, "thresh_w").generate_frame(b))
    f = np.concatenate([np.zeros((1, 200), f.dtype), f, np.zeros((1, 200), f.dtype)], -1)
    y = np.asarray(Channel(snr_db=snr, multipath_taps=taps, seed=1000 + t).process(f))
    res = []
    for m in ("thresh_w", "thresh_wq"):
        try:
            r = mk(mod, m).rx_process(y)
            ok = bool(r.get("frame_found")) and np.array_equal(np.asarray(r["bits"]).ravel()[:8000], b.ravel())
        except Exception:
            ok = False
        res.append(not ok)
    return res

if __name__ == "__main__":
    mp = np.array([1.0, 0, 0, 0.5j, 0, 0, 0, 0.3], dtype=np.complex64); mp /= np.sqrt(np.sum(np.abs(mp) ** 2))
    deep = np.array([1.0, 0, 0.9, 0, 0, -0.6j, 0, 0, 0.4], dtype=np.complex64); deep /= np.sqrt(np.sum(np.abs(deep) ** 2))
    pts = [("qpsk", 6, None, "AWGN"), ("qam16", 12, None, "AWGN"), ("qam64", 18, None, "AWGN"),
           ("qpsk", 8, mp, "multipath"), ("qam16", 14, mp, "multipath"), ("qam64", 20, mp, "multipath"),
           ("qam16", 16, deep, "deep fades"), ("qam16", 18, deep, "deep fades")]
    print(f"{'channel':>11s} {'mod':>6s} {'SNR':>4s} {'thresh_w':>9s} {'table':>9s}")
    with ProcessPoolExecutor(8) as ex:
        for mod, snr, tp, lab in pts:
            rs = list(ex.map(one, [(mod, snr, tp, t) for t in range(100)]))
            print(f"{lab:>11s} {mod:>6s} {snr:>4d} {sum(r[0] for r in rs)/100:9.2f} {sum(r[1] for r in rs)/100:9.2f}", flush=True)
