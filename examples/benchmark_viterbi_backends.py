"""Same-machine A/B of every compiled Viterbi decode backend on the real
PDU sizes this project decodes (24040 bits = QPSK, 32032 = QAM16/QAM64
at a 32000-bit SDU -- see benchmark_x86_stages_v3.py), on random
codewords with 3% bit errors, pinned to one core.

Drives the native classes DIRECTLY (fec/_native.py), not through
ConvolutionalCode's dispatch, so it reports every backend that compiles
on this machine regardless of which one the dispatch currently prefers
-- that is exactly what's needed to decide the dispatch order for a new
machine type (e.g. "fast" vs "neon" on the Pi 5), per this project's
measured-not-assumed rule. Every backend's output is checked against the
portable build before it is timed.

Usage:  python examples/benchmark_viterbi_backends.py
"""
import os
import platform
import statistics
import time

import numpy as np

from spectracuda.fec import _native

try:
    os.sched_setaffinity(0, {int(os.environ.get("SPECTRACUDA_BENCH_PIN_CORE", "2"))})
except Exception:
    pass

backends = {"portable": _native.NativeConvolutional()}
if _native.sse_available():
    backends["sse4.1"] = _native.NativeConvolutionalSSE()
if _native.neon_available():
    backends["neon"] = _native.NativeConvolutionalNEON()
if _native.fast_available():
    backends["fast"] = _native.NativeConvolutionalFast()
print(f"machine={platform.machine()}  compiled backends: {list(backends)}")

rng = np.random.default_rng(0)
ref = backends["portable"]
for k in (24040, 32032):
    msg = rng.integers(0, 2, size=(1, k)).astype("uint8")
    enc = ref.encode(msg)
    rx = (enc ^ (rng.random(enc.shape) < 0.03).astype("uint8")).astype("uint8")
    expected = ref.decode(rx)
    cells = []
    for name, be in backends.items():
        assert np.array_equal(be.decode(rx), expected), f"{name} is NOT bit-exact with portable -- do not trust its timing"
        for _ in range(5):
            be.decode(rx)
        ts = []
        for _ in range(60):
            t0 = time.perf_counter()
            be.decode(rx)
            ts.append(time.perf_counter() - t0)
        ms = statistics.median(ts) * 1e3
        cells.append(f"{name}: {ms:.3f} ms ({ms * 1e6 / k:.0f} ns/bit)")
    print(f"k={k}: " + " | ".join(cells))
