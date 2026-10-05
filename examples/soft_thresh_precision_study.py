"""How much precision does the "thresh_w" soft demapper need?

Exact thresh_w (Modem.demodulate_soft_thresh, the committed reference)
vs a reduced-precision model of a small FPGA implementation:
  * xn = x / norm quantized to F fractional bits (round to nearest) and
    clipped to +/- XCLIP * m (m = levels per axis)
  * t = (L1 - L0)(2 xn - L0 - L1) from the QUANTIZED xn (integer math)
  * q = clip(round(7 t 2^k / 4), -7, 7) -- unchanged
  * k exact (the RTL compares wide enough fields; not the cost driver)
Frame error rate on the same frames/channels, 100 frames per point, at the
SNRs where the curves fall (from examples/soft_metric_study.py).

    .venv/bin/python examples/soft_thresh_precision_study.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel
from spectracuda.modem import Modem

_real = Modem.demodulate_soft_thresh
CFG = {"F": None, "XCLIP": 1.0, "KBITS": None}
SCHEMES = [("exact", None, None), ("F5", 5, None), ("F4", 4, None), ("F3", 3, None), ("F2", 2, None)]


def thresh_q(self, symbols, weight_exp, llr_bits=4):
    F = CFG["F"]
    if F is None:
        return _real(self, symbols, weight_exp, llr_bits)
    y = np.asarray(symbols)
    if y.ndim == 1:
        y = y[None, :]
    pts, labels = self._point_table()
    lev = np.unique(np.round(pts.real, 6))
    norm = float(np.min(np.abs(lev)))
    nlev = len(lev)                                 # levels per axis
    half = int(np.log2(nlev))
    m = self.bits_per_symbol
    k = np.asarray(weight_exp, dtype=np.int64)
    out = np.empty(y.shape + (m,), dtype=np.int64)
    L = np.arange(nlev) * 2 - (nlev - 1)            # odd levels
    gray = L * 0 + np.array([a ^ (a >> 1) for a in range(nlev)])
    for ax, comp in ((0, y.real), (1, y.imag)):
        xq = np.clip(np.rint(comp / norm * 2 ** F), -CFG["XCLIP"] * nlev * 2 ** F, CFG["XCLIP"] * nlev * 2 ** F)
        for j in range(half):
            bit = (gray >> (half - 1 - j)) & 1
            # nearest level with bit 0 / bit 1 (integer distances on the quantized axis)
            d = np.abs(xq[..., None] - L[None, None, :] * 2 ** F)
            l0 = L[np.argmin(np.where(bit[None, None, :] == 0, d, 1 << 40), axis=-1)]
            l1 = L[np.argmin(np.where(bit[None, None, :] == 1, d, 1 << 40), axis=-1)]
            t = (l1 - l0) * (2 * xq - (l0 + l1) * 2 ** F)          # Q(F)
            # q = round(7 t 2^k / 4 / 2^F), half to even
            v = 7.0 * t * np.ldexp(1.0, k) / (4.0 * 2 ** F)
            out[..., ax * half + j] = np.rint(np.clip(v, -7, 7))
    Lq = float(2 ** (llr_bits - 1) - 1)
    soft = np.clip(np.rint(128.0 + 127.0 * out / Lq), 0, 255).astype("uint8")
    return soft.reshape(y.shape[0], -1)


def kexp_reduced(h2, ref, bits):
    """k = -3 + #{m: n_data*h2 >= ref*2^m}, compared on `bits`-bit leading fields
    of the reference (truncate both sides below the reference's top bits)."""
    out = np.full(h2.shape, -3, dtype=np.int64)
    for mm in (-3, -2, -1, 0):
        T = ref * 2.0 ** mm
        s = np.floor(np.log2(np.maximum(T, 1e-30))) - (bits - 1)
        out += np.floor(h2 / 2.0 ** s) >= np.floor(T / 2.0 ** s)
    return out


def mk(mod, metric="thresh_w", soft=True):
    return Ofdm(fft_size=256, cp_len=32, n_data=216, n_pilot=8, sync="schmidl_cox", cfo="schmidl_cox",
                channel_estimator="ls", equalizer="mmse", modem=mod, fec1="conv_v27",
                interleaver="block", interleaver_kwargs={"unit_bits": 8},
                soft_decision=soft, soft_llr_bits=4, soft_llr_metric=metric)


def one(args):
    mod, snr, taps, t, bits = args
    Modem.demodulate_soft_thresh = thresh_q
    tx = mk(mod)
    b = np.random.default_rng(t).integers(0, 2, (1, bits)).astype("uint8")
    f = np.asarray(tx.generate_frame(b))
    f = np.concatenate([np.zeros((1, 200), f.dtype), f, np.zeros((1, 200), f.dtype)], -1)
    y = np.asarray(Channel(snr_db=snr, multipath_taps=taps, seed=1000 + t).process(f))
    res = {}
    for name, F, kb in SCHEMES:
        CFG["F"] = F
        o = mk(mod)
        try:
            r = o.rx_process(y)
            ok = bool(r.get("frame_found")) and np.array_equal(np.asarray(r["bits"]).ravel()[:bits], b.ravel())
        except Exception:
            ok = False
        res[name] = not ok
    return res


POINTS = [
    ("qpsk", 6, None, "AWGN"), ("qam16", 12, None, "AWGN"), ("qam64", 18, None, "AWGN"),
    ("qpsk", 8, "mp", "multipath"), ("qpsk", 10, "mp", "multipath"),
    ("qam16", 14, "mp", "multipath"), ("qam16", 16, "mp", "multipath"),
    ("qam64", 20, "mp", "multipath"), ("qam64", 22, "mp", "multipath"),
    ("qam16", 16, "deep", "deep fades"), ("qam16", 18, "deep", "deep fades"),
]

if __name__ == "__main__":
    mp = np.array([1.0, 0, 0, 0.5j, 0, 0, 0, 0.3], dtype=np.complex64); mp /= np.sqrt(np.sum(np.abs(mp) ** 2))
    deep = np.array([1.0, 0, 0.9, 0, 0, -0.6j, 0, 0, 0.4], dtype=np.complex64); deep /= np.sqrt(np.sum(np.abs(deep) ** 2))
    chans = {None: None, "mp": mp, "deep": deep}
    n = 100
    print("frame error rate, 100 frames of 8000 bits per point (lower is better)")
    print(f"{'channel':>11s} {'mod':>6s} {'SNR':>4s} " + "".join(f"{s[0]:>8s}" for s in SCHEMES))
    with ProcessPoolExecutor(8) as ex:
        for mod, snr, ch, label in POINTS:
            rs = list(ex.map(one, [(mod, snr, chans[ch], t, 8000) for t in range(n)]))
            print(f"{label:>11s} {mod:>6s} {snr:>4d} " + "".join(f"{sum(r[s[0]] for r in rs) / n:8.2f}" for s in SCHEMES))
            sys.stdout.flush()
