"""Which soft-decision metric does the FPGA receiver actually need?

Same frames, same channels, five ways of turning equalized symbols into the
Viterbi's soft values (frame error rate, lower is better):

  ours       max-log for every bit, weighted by |H[k]|^2/(mean|H|^2 * noise),
             4-bit (Ofdm soft_llr_scale="stream", clip 3) -- the current RTL
  openofdm   OpenOFDM style (reference/openofdm demodulate.v/deinterleave.v):
             fixed thresholds, 3 bits, NO channel weighting, and only the bit
             nearest its decision boundary on each axis is soft; the other
             bits of that axis are sent as fully confident
  thresh     fixed thresholds for EVERY bit, no weighting, 4 bits
  thresh+w   thresh, times a coarse channel weight: |H[k]|^2/mean rounded to
             a power of two in [1/8, 2] (a shift in hardware, no multiplier)
  hard       hard decision (floor)

"Fixed thresholds" = the max-log value with a constant scale instead of the
measured noise: for Gray QAM max-log is piecewise linear in the received
value with breakpoints at the constellation's decision boundaries, so this
is exactly comparators + shifts in hardware. Saturates (full confidence) at
the distance of the constellation point nearest the boundary.

    .venv/bin/python examples/soft_metric_study.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel
from spectracuda.modem import Modem

SCHEMES = ["ours", "openofdm", "thresh", "thresh+w", "hard"]
_real_soft = Modem.demodulate_soft
_current = {"scheme": None, "ofdm": None}


def _raw_maxlog(self, y):
    """raw max-log per bit, constellation units (> 0 means bit 1)."""
    pts, labels = self._point_table()
    d = np.abs(y[..., None] - pts[None, None, :]) ** 2
    m = self.bits_per_symbol
    out = np.empty(y.shape + (m,), dtype=np.float64)
    for b in range(m):
        out[..., b] = d[..., labels[:, b] == 0].min(-1) - d[..., labels[:, b] == 1].min(-1)
    return out


def _to_bytes(q, L, shape0):
    return np.clip(np.rint(128.0 + 127.0 * q / L), 0, 255).astype(np.uint8).reshape(shape0, -1)


def _soft(self, symbols, weight=None, llr_clip=6.0, llr_bits=None, sigma2=None):
    scheme = _current["scheme"]
    if scheme == "ours":
        return _real_soft(self, symbols, weight=weight, llr_clip=llr_clip, llr_bits=llr_bits, sigma2=sigma2)
    y = np.asarray(symbols)
    if y.ndim == 1:
        y = y[None, :]
    raw = _raw_maxlog(self, y)
    # constant scale: full confidence at the boundary-nearest point,
    # t = (L1-L0)(2xn - L0 - L1) = 4 there, raw = 4 * norm^2
    pts, _ = self._point_table()
    norm2 = float(np.min(np.abs(np.unique(np.round(pts.real, 6)))) ** 2)
    v = raw / (4.0 * norm2)                      # 1.0 = "at the nearest point"
    if scheme == "thresh+w":
        o = _current["ofdm"]
        # weight passed in = |H|^2 / (mean_train|H|^2 * noise); undo the noise
        w = np.asarray(weight) * float(np.asarray(o._stream_noise).ravel()[0])
        w = np.clip(2.0 ** np.round(np.log2(np.maximum(w, 1e-9))), 1 / 8, 2.0)
        v = v * w[..., None]
    if scheme == "openofdm":
        L = 3.0
        q = np.rint(np.clip(v, -1, 1) * L)
        m = q.shape[-1]
        half = m // 2
        if half > 1:   # QPSK: every bit is the axis's only bit, all soft
            for ax in (slice(0, half), slice(half, m)):
                blk = q[..., ax]
                keep = np.argmin(np.abs(v[..., ax]), axis=-1)       # bit nearest its boundary
                hard = np.sign(blk); hard[hard == 0] = 1
                full = hard * L
                idx = np.arange(blk.shape[-1])
                soft_mask = idx[None, None, :] == keep[..., None]
                q[..., ax] = np.where(soft_mask, blk, full)
        return _to_bytes(q, L, y.shape[0])
    L = 7.0
    q = np.rint(np.clip(v, -1, 1) * L)
    return _to_bytes(q, L, y.shape[0])


def mk(mod, soft):
    return Ofdm(fft_size=256, cp_len=32, n_data=216, n_pilot=8, sync="schmidl_cox", cfo="schmidl_cox",
                channel_estimator="ls", equalizer="mmse", modem=mod, fec1="conv_v27",
                interleaver="block", interleaver_kwargs={"unit_bits": 8},
                soft_decision=soft, soft_llr_bits=4 if soft else None,
                soft_llr_scale="stream", soft_llr_clip=3.0)


def one(args):
    mod, snr, taps, t, bits = args
    Modem.demodulate_soft = _soft
    tx = mk(mod, False)
    b = np.random.default_rng(t).integers(0, 2, (1, bits)).astype("uint8")
    f = np.asarray(tx.generate_frame(b))
    f = np.concatenate([np.zeros((1, 200), f.dtype), f, np.zeros((1, 200), f.dtype)], -1)
    y = np.asarray(Channel(snr_db=snr, multipath_taps=taps, seed=1000 + t).process(f))
    res = {}
    for s in SCHEMES:
        o = mk(mod, s != "hard")
        _current["scheme"], _current["ofdm"] = s, o
        try:
            r = o.rx_process(y)
            ok = bool(r.get("frame_found")) and np.array_equal(np.asarray(r["bits"]).ravel()[:bits], b.ravel())
        except Exception:
            ok = False
        res[s] = not ok
    return res


def run(ex, mod, snrs, taps, label, n=100, bits=8000):
    print(f"\n{mod}, {label}: frame error rate, {n} frames of {bits} bits each")
    print("  SNR dB " + "".join(f"{s:>11s}" for s in SCHEMES))
    for snr in snrs:
        rs = list(ex.map(one, [(mod, snr, taps, t, bits) for t in range(n)]))
        print(f"  {snr:6.1f} " + "".join(f"{sum(r[s] for r in rs) / n:11.3f}" for s in SCHEMES))
        sys.stdout.flush()


if __name__ == "__main__":
    taps = np.array([1.0, 0, 0, 0.5j, 0, 0, 0, 0.3], dtype=np.complex64)
    taps = taps / np.sqrt(np.sum(np.abs(taps) ** 2))
    deep = np.array([1.0, 0, 0.9, 0, 0, -0.6j, 0, 0, 0.4], dtype=np.complex64)   # deeper fades
    deep = deep / np.sqrt(np.sum(np.abs(deep) ** 2))
    with ProcessPoolExecutor(8) as ex:
        run(ex, "qpsk", [3, 4, 5, 6], None, "AWGN")
        run(ex, "qam16", [9, 10, 11, 12], None, "AWGN")
        run(ex, "qam64", [15, 16, 17, 18], None, "AWGN")
        run(ex, "qpsk", [6, 8, 10, 12], taps, "3-tap multipath")
        run(ex, "qam16", [12, 14, 16, 18], taps, "3-tap multipath")
        run(ex, "qam64", [18, 20, 22, 24], taps, "3-tap multipath")
        run(ex, "qam16", [14, 16, 18, 20, 22], deep, "deep-fade multipath")
