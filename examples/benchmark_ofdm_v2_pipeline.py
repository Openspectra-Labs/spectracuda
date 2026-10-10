"""Does the OfdmV2 flowgraph actually use more than one core?

HOW THIS AVOIDS LYING TO YOU. Two things make a naive benchmark
worthless here:

* Msps on this dev machine swings 3-5x run to run (WSL2 vCPU
  scheduling; pinning does not isolate it). So every arm runs
  INTERLEAVED in one process, repeated, and the median is reported --
  never a single number from a single run.
* Wall-clock speedup conflates "the pipeline got faster" with "this run
  got luckier". The honest measure of parallelism is

      overlap = sum(per-stage busy time) / wall time

  which is 1.0 when nothing overlaps and approaches the number of
  stages when they run concurrently. It is computed from the
  flowgraph's own per-stage counters, so it does not depend on the
  machine being quiet.

WHY THE ANSWER IS NOT OBVIOUS. The stages spend most of their time in
GIL-releasing native code -- the numba sync/CFO/mapper kernels are
`nogil=True`, libcorrect is bound through ctypes, numpy's FFT releases
the GIL -- so overlap is possible. But the Python-level array plumbing
between those kernels still holds the GIL, and the one prior data point
in this project is `Mac.receive_iq_batch` with >1 worker being a NET
LOSS on x86 from GIL contention (while the same work scaled 1.96x on a
free-threaded build). That was data-parallel rather than
pipeline-parallel, so it does not transfer directly -- which is exactly
why this measures instead of assuming.

    python examples/benchmark_ofdm_v2_pipeline.py
    python examples/benchmark_ofdm_v2_pipeline.py --frames 24 --repeats 5
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from spectracuda.pipeline import Ofdm
from spectracuda.pipeline.v2.env import PhyEnv
from spectracuda.pipeline.v2.flowgraph import RxFlowgraph
from spectracuda.pipeline.v2.rx_pipeline import RxPipeline
from spectracuda.pipeline.v2.stage_if import SoftConfig

CHUNK = 2048


def make_ofdm(modem: str, dmrs_interval: int = 32) -> Ofdm:
    o = Ofdm(fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem=modem,
             fec="rs_m8", fec1="conv_v27",
             interleaver="block", interleaver_kwargs={"unit_bits": 8},
             crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
             n_training_symbols=2, dmrs_interval=dmrs_interval,
             soft_llr_scale="stream", backend="numpy")
    o.MAX_PAYLOAD_SYMBOLS = 256
    return o


def soft_of(o: Ofdm) -> SoftConfig:
    return SoftConfig(enabled=o.soft_decision_active, metric=o.soft_llr_metric,
                      bits=o.soft_llr_bits, clip=o.soft_llr_clip,
                      scale=o.soft_llr_scale)


def build_stream(o: Ofdm, n_frames: int, payload_bytes: int, seed: int = 0):
    """One long capture of back-to-back frames, plus the payloads."""
    rng = np.random.default_rng(seed)
    txs, parts = [], [np.zeros((1, 800), dtype="complex64")]
    for _ in range(n_frames):
        tx = rng.integers(0, 2, size=(1, payload_bytes * 8)).astype("uint8")
        txs.append(tx)
        parts.append(np.asarray(o.generate_frame(tx)))
        parts.append(np.zeros((1, 300), dtype="complex64"))
    parts.append(np.zeros((1, 3000), dtype="complex64"))
    return txs, np.concatenate(parts, axis=-1)


def run_v1(o: Ofdm, sig) -> tuple:
    """`Ofdm.rx_streaming` -- the shipping single-threaded receiver."""
    o.reset_stream()
    t0 = time.perf_counter()
    n = 0
    for i in range(0, sig.shape[-1], CHUNK):
        if o.rx_streaming(sig[:, i:i + CHUNK]) is not None:
            n += 1
    return time.perf_counter() - t0, n, None


def run_v2_serial(o: Ofdm, sig, chunk_symbols: int) -> tuple:
    rx = RxPipeline(PhyEnv.from_ofdm(o), soft=soft_of(o),
                    expected_fec=(o.fec, o.fec1), chunk_symbols=chunk_symbols)
    t0 = time.perf_counter()
    n = 0
    for i in range(0, sig.shape[-1], CHUNK):
        n += len(rx.feed(sig[:, i:i + CHUNK]))
    return time.perf_counter() - t0, n, None


def run_v2_threaded(o: Ofdm, sig, chunk_symbols: int, expect: int) -> tuple:
    """Timed to LAST FRAME OUT, not to drain().

    `drain()` deliberately waits for quiescence -- four consecutive quiet
    polls -- because declaring "done" too early was itself a frame-loss
    bug. But that is a fixed ~80 ms confirmation delay, not processing
    time, and on a half-second run it would overstate the threading
    penalty by a sixth. So the clock stops when the expected frames have
    all arrived; `drain()` still runs, just outside the measurement.
    """
    fg = RxFlowgraph(PhyEnv.from_ofdm(o), soft=soft_of(o),
                     expected_fec=(o.fec, o.fec1),
                     chunk_symbols=chunk_symbols, queue_depth=256).start()
    t0 = time.perf_counter()
    try:
        for i in range(0, sig.shape[-1], CHUNK):
            fg.feed(sig[:, i:i + CHUNK])
        deadline = t0 + 60.0
        while time.perf_counter() < deadline:
            with fg._out_lock:
                if len(fg._out) >= expect:
                    break
            time.sleep(0.0005)
        dt = time.perf_counter() - t0
        got = fg.drain()
    finally:
        fg.stop()
    return dt, len(got), fg.report()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--bytes", type=int, default=600)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--modem", default="qam16")
    args = ap.parse_args()

    o = make_ofdm(args.modem)
    txs, sig = build_stream(o, args.frames, args.bytes)
    n_samples = sig.shape[-1]
    print(f"{args.frames} frames x {args.bytes} B {args.modem}, "
          f"{n_samples} samples, {args.repeats} interleaved repeats\n")

    arms = {
        "V1 rx_streaming": lambda: run_v1(o, sig),
        "V2 serial  cs=1": lambda: run_v2_serial(o, sig, 1),
        "V2 serial  cs=8": lambda: run_v2_serial(o, sig, 8),
        "V2 threads cs=1": lambda: run_v2_threaded(o, sig, 1, args.frames),
        "V2 threads cs=4": lambda: run_v2_threaded(o, sig, 4, args.frames),
        "V2 threads cs=8": lambda: run_v2_threaded(o, sig, 8, args.frames),
        "V2 threads cs=16": lambda: run_v2_threaded(o, sig, 16, args.frames),
    }
    results = {k: [] for k in arms}
    reports = {}
    # Interleaved: every arm sees the same scheduling weather.
    for _ in range(args.repeats):
        for name, fn in arms.items():
            dt, n, rep = fn()
            results[name].append((dt, n))
            if rep is not None:
                reports[name] = rep

    print(f"{'arm':18s} {'median s':>9s} {'Msps':>8s} {'frames':>7s} {'overlap':>8s}")
    print("-" * 56)
    for name in arms:
        runs = results[name]
        med = statistics.median(dt for dt, _ in runs)
        frames = min(n for _, n in runs)
        msps = n_samples / med / 1e6
        rep = reports.get(name)
        if rep is None:
            overlap = "-"
        else:
            busy = sum(s["busy_s"] for s in rep["stages"].values())
            overlap = f"{busy / med:.2f}x"
        flag = "" if frames == args.frames else f"  <-- {frames}/{args.frames}!"
        print(f"{name:18s} {med:9.3f} {msps:8.3f} {frames:7d} {overlap:>8s}{flag}")

    print("\nPer-stage busy time (threaded arms) -- the pipeline runs no")
    print("faster than its slowest stage:")
    for name, rep in reports.items():
        st = rep["stages"]
        tot = sum(s["busy_s"] for s in st.values()) or 1.0
        shares = "  ".join(f"{k}={s['busy_s']:.3f}s ({100*s['busy_s']/tot:.0f}%)"
                           for k, s in st.items())
        print(f"  {name:18s} {shares}")
        q = rep["queues"]
        print(f"{'':20s} high-water i1={q['i1']['high_water']}/{q['i1']['maxsize']}"
              f"  i2={q['i2']['high_water']}/{q['i2']['maxsize']}"
              f"  blocked={q['i1']['blocked_puts']}"
              f"  surplus={rep['fd']['dropped_surplus']}")

    print("\nReminder: `overlap` = sum(stage busy) / wall. 1.00x means the")
    print("stages did not overlap at all (GIL-bound); higher means they did.")
    print("Msps here is NOT a hardware figure -- this machine's throughput")
    print("swings 3-5x run to run, which is why arms are interleaved and")
    print("medians reported.")


if __name__ == "__main__":
    main()
