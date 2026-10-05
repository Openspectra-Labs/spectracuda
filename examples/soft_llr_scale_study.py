"""Does soft_llr_scale="stream" (noise from the header, |H|^2 normalized by
the training estimate -- what a streaming FPGA receiver can know) lose
anything against the default "frame" scaling? 4-bit LLRs (the RTL's width),
hard decision as the floor. Frame error rate vs SNR, AWGN and a 3-tap
frequency-selective channel, DMRS every 16 data symbols.

    .venv/bin/python examples/soft_llr_scale_study.py
"""
import sys
import numpy as np
from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel


def mk(mod, soft, scale):
    return Ofdm(fft_size=256, cp_len=32, n_data=216, n_pilot=8, sync="schmidl_cox", cfo="schmidl_cox",
                channel_estimator="ls", equalizer="mmse", modem=mod, fec1="conv_v27",
                interleaver="block", interleaver_kwargs={"unit_bits": 8}, interleaver2="block",
                soft_decision=soft, soft_llr_bits=4 if soft else None, soft_llr_scale=scale,
                dmrs_interval=16)


def run(mod, snrs, taps, n=40, bits=8000):
    modes = [("hard", False, "frame"), ("soft-frame", True, "frame"), ("soft-stream", True, "stream")]
    tx = mk(mod, False, "frame")
    rx = {name: mk(mod, s, sc) for name, s, sc in modes}
    print(f"\n{mod}, {'3-tap multipath' if taps is not None else 'AWGN'}: frame error rate over {n} frames of {bits} bits")
    print("  SNR dB  " + "  ".join(f"{m[0]:>11s}" for m in modes))
    for snr in snrs:
        err = {m[0]: 0 for m in modes}
        for t in range(n):
            rng = np.random.default_rng(t)
            b = rng.integers(0, 2, (1, bits)).astype("uint8")
            f = np.asarray(tx.generate_frame(b))
            f = np.concatenate([np.zeros((1, 200), f.dtype), f, np.zeros((1, 200), f.dtype)], -1)
            y = np.asarray(Channel(snr_db=snr, multipath_taps=taps, seed=1000 + t).process(f))
            for name, o in rx.items():
                try:
                    r = o.rx_process(y)
                    ok = r.get("frame_found") and np.array_equal(np.asarray(r["bits"]).ravel()[:bits], b.ravel())
                except Exception:
                    ok = False
                err[name] += not ok
        print(f"  {snr:6.1f}  " + "  ".join(f"{err[m[0]] / n:11.3f}" for m in modes))
        sys.stdout.flush()


if __name__ == "__main__":
    taps = np.array([1.0, 0.0, 0.0, 0.5j, 0.0, 0.0, 0.0, 0.3], dtype=np.complex64)
    taps = taps / np.sqrt(np.sum(np.abs(taps) ** 2))
    run("qpsk", [2, 3, 4, 5], None)
    run("qam16", [8, 9, 10, 11], None)
    run("qam64", [14, 15, 16, 17], None)
    run("qam16", [10, 12, 14, 16], taps)
    run("qam64", [16, 18, 20, 22], taps)
