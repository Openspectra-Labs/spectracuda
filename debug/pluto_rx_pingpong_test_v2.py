"""Multiprocessing variant of pluto_rx_pingpong_test.py -- swaps the
threading-based ping-pong (capture thread / decode thread sharing one
process's GIL) for a genuine 2-process design: capture runs in its own
OS process with its own GIL, decode stays in this (parent) process.

Root cause this fixes (see conversation this was diagnosed in): with
BOTH capture and decode as threading.Thread in the same process, the
ip:-backend rx.rx()/refill() call does real Python-level bookkeeping
(channel buffer parsing, I+Q complex combine in pyadi-iio's compat.py)
WHILE HOLDING THE GIL -- a concurrently running CPU-heavy decode thread
competes for that same lock and measurably stretches the wall-clock
duration of rx() itself. Confirmed empirically with a standalone probe:
the exact same 100k-sample rx() call took 24982us alone vs 79561us with
a synthetic CPU-bound thread running concongruently -- a 3.18x stretch,
with the inter-call GAP staying negligible in both cases (so the loss
is inside the call, not idle time between calls). That stretch is real
physical RX time during which the AD9363's hardware-side buffer keeps
filling regardless of what Python is doing -- once it overflows before
the next refill() drains it, real RF samples are gone before Python
ever sees them. A separate PROCESS has its own interpreter and its own
GIL, so decode's CPU load can never again steal time from capture's
rx() call, no matter how heavy the decode workload gets.

The 2-slot strict-alternation discipline itself (see
pluto_rx_pingpong_test.py's docstring for why: a general queue.Queue
depth produced wild decode-time variance) is UNCHANGED here -- only the
underlying primitives cross a process boundary instead of a thread
boundary:
    slots[]            -> multiprocessing.shared_memory.SharedMemory
                           (numpy views onto the SAME physical memory,
                           zero-copy between the two processes)
    buf_ready/buf_free  -> multiprocessing.Event (OS semaphore-backed,
                           same .set()/.clear()/.wait() API as
                           threading.Event)

Also bumps RX_SAMPLES 100k -> 250k (2.5x more slack per buffer) per
this session's request, on top of the process-isolation fix.

A second, separate bug was caught building this: the first version
measured only ~88% coverage even after the process split -- turned out
to be a self-inflicted timing bug, not a real capture problem (see the
capture_ready handshake in capture_process()/main() below for the
full explanation). With that fixed, measured coverage is ~99.9%.

Bonus from the process split: the libiio Buffer/Context teardown
segfault noted in v1's module docstring is now isolated to the capture
child process -- if it happens, it can no longer take the decode side
(this process) down with it.

Usage:
    python3 debug/pluto_rx_pingpong_test_v2.py --uri-rx ip:192.168.3.1 --rate 4e6 --seconds 60
"""
import argparse
import gc
import multiprocessing as mp
import os
import sys
import time
from multiprocessing import shared_memory

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
    fec="rs_m8", fec1="conv_v27", crc="crc16",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
    backend="numpy",
)
RX_SAMPLES = 250_000  # was 100_000 in v1 -- 2.5x more slack per buffer
STREAM_CHUNK = 2048
# Matches debug/pluto_tx_standalone_test.py's PAYLOAD_BYTES exactly.
EXPECTED_PAYLOAD_BYTES = bytes(range(64))


def capture_process(uri_rx, freq, rate, rx_gain, shm_names, buf_ready, buf_free, stop_event, stats_q, capture_ready):
    """Runs in its own OS process -- own interpreter, own GIL. The Pluto
    handle (and its open network socket to iiod) is opened HERE and must
    never cross into the decode side."""
    from pluto_common import pluto_rx_init  # only this process needs adi

    # Pin to a dedicated core, as cheap defensive headroom against OS
    # scheduler contention on this 4-core box under heavier real decode
    # load (CRC/FEC on actual traffic, not just idle correlator search).
    # NOTE: measured to make no difference at the load levels tested so
    # far -- coverage was ~99.88% both with and without this pin. The
    # real fix for the ~88%-coverage regression seen before this was the
    # capture_ready handshake below: without it, the parent's timer
    # started before this process finished its Pluto connect/config +
    # warm-up, and that setup latency was being counted as dropped
    # coverage even though nothing was actually droppable yet.
    try:
        os.sched_setaffinity(0, {0})
    except (AttributeError, OSError):
        pass  # not available on this platform -- proceed unpinned

    rx = pluto_rx_init(uri_rx, freq, rate, rx_gain, RX_SAMPLES)
    for _ in range(10):
        rx.rx()

    shms = [shared_memory.SharedMemory(name=n) for n in shm_names]
    views = [np.ndarray((RX_SAMPLES,), dtype="complex64", buffer=s.buf) for s in shms]

    # Signal the parent only NOW -- Pluto connect/config round-trips and
    # the warm-up captures above are done. Without this handshake, the
    # parent's t_start (and therefore its coverage-pct accounting) would
    # start ticking the instant cap_proc.start() returns, silently
    # counting all of this setup latency as "missed real-time coverage"
    # even though nothing was actually droppable yet -- v1 avoided this
    # by doing all of this BEFORE starting the capture thread at all.
    capture_ready.set()

    capture_us = []
    n_capture_waits = 0
    i = 0
    while not stop_event.is_set():
        if not buf_free[i].is_set():
            n_capture_waits += 1
        if not buf_free[i].wait(timeout=2.0):
            continue
        buf_free[i].clear()
        t1 = time.perf_counter()
        raw = rx.rx()
        t2 = time.perf_counter()
        views[i][:] = np.asarray(raw, dtype="complex64")
        capture_us.append((t2 - t1) * 1e6)
        buf_ready[i].set()
        i ^= 1

    # Send stats back BEFORE any teardown -- if libiio's shutdown segfaults
    # (see pluto_rx_pingpong_test.py's teardown comment), it now only takes
    # down THIS process, never the parent/decode side, but we still want
    # the stats to have made it out first regardless.
    stats_q.put((capture_us, n_capture_waits))

    del rx
    for s in shms:
        s.close()
    gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri-rx", required=True)
    ap.add_argument("--freq", type=float, default=2.425e9)
    ap.add_argument("--rate", type=float, default=4e6)
    ap.add_argument("--rx-gain", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=10.0, help="0 = run until Ctrl-C")
    args = ap.parse_args()

    from spectracuda.pipeline import Ofdm  # only the parent (decode side) needs this

    ofdm_rx = Ofdm(**PHY_KWARGS)
    ofdm_rx.reset_stream()

    nbytes = RX_SAMPLES * 8  # complex64 = 8 bytes/sample
    shms = [shared_memory.SharedMemory(create=True, size=nbytes) for _ in range(2)]
    shm_names = [s.name for s in shms]
    views = [np.ndarray((RX_SAMPLES,), dtype="complex64", buffer=s.buf) for s in shms]

    buf_ready = [mp.Event(), mp.Event()]
    buf_free = [mp.Event(), mp.Event()]
    for e in buf_free:
        e.set()  # both slots start empty/free
    stop_event = mp.Event()
    stats_q = mp.Queue()
    capture_ready = mp.Event()

    BUDGET_US = RX_SAMPLES / args.rate * 1e6

    cap_proc = mp.Process(
        target=capture_process,
        args=(args.uri_rx, args.freq, args.rate, args.rx_gain, shm_names, buf_ready, buf_free, stop_event, stats_q, capture_ready),
        daemon=True,
    )
    cap_proc.start()

    # Keep decode (this process) off core 0 so it can never contend with
    # capture's dedicated core -- see the matching comment in
    # capture_process() above.
    try:
        os.sched_setaffinity(0, set(range(1, os.cpu_count())) or {0})
    except (AttributeError, OSError):
        pass

    # Wait for the child to finish Pluto connect/config + warm-up before
    # starting the measurement clock -- see the comment at
    # capture_ready.set() in capture_process(). Poll in short slices so a
    # crashed child (e.g. import error, bad URI) is noticed quickly
    # instead of the parent hanging for the full timeout.
    waited = 0.0
    while not capture_ready.wait(timeout=0.5):
        waited += 0.5
        if not cap_proc.is_alive():
            raise RuntimeError(f"capture process died before signaling ready (exitcode={cap_proc.exitcode})")
        if waited >= 30.0:
            raise RuntimeError("capture process did not signal ready within 30s")

    print(f"[rx-pingpong-v2] uri_rx={args.uri_rx} rate={args.rate/1e6:.1f}Msps rx_gain={args.rx_gain:+.1f}dB "
          f"rx_samples={RX_SAMPLES} stream_chunk={STREAM_CHUNK} budget={BUDGET_US:.0f}us "
          f"capture_pid={cap_proc.pid} running for {args.seconds:.1f}s ...")

    decode_us = []
    n_decoded = 0
    n_crc_valid = 0
    n_bytes_match = 0
    n_bytes_mismatch = 0
    n_buffers_seen = 0
    n_decode_waits = 0

    t_start = time.perf_counter()
    t_end = float("inf") if args.seconds <= 0 else t_start + args.seconds
    i = 0
    try:
        while time.perf_counter() < t_end:
            if not buf_ready[i].is_set():
                n_decode_waits += 1
            if not buf_ready[i].wait(timeout=2.0):
                continue
            buf_ready[i].clear()
            buf = views[i]
            t1 = time.perf_counter()
            results = []
            for start in range(0, len(buf), STREAM_CHUNK):
                piece = buf[start:start + STREAM_CHUNK]
                results.append(ofdm_rx.rx_streaming(piece[None, :]))
            t2 = time.perf_counter()
            decode_us.append((t2 - t1) * 1e6)
            n_buffers_seen += 1
            for result in results:
                if result is None:
                    continue
                n_decoded += 1
                crc_ok = bool(np.asarray(result["crc_valid"])[0]) if result["crc_valid"] is not None else None
                if crc_ok:
                    n_crc_valid += 1
                decoded_bits = np.asarray(result["bits"])[0].astype("uint8")
                decoded_bytes = np.packbits(decoded_bits).tobytes()
                match = decoded_bytes[:64] == EXPECTED_PAYLOAD_BYTES
                if match:
                    n_bytes_match += 1
                else:
                    n_bytes_mismatch += 1
                evm = float(np.asarray(result["evm"])[0]) if result["evm"] is not None else float("nan")
                print(f"  [decoded] #{n_decoded} crc_valid={crc_ok} bytes_match_expected={match} "
                      f"evm={evm:.5f} rssi_db={float(np.asarray(result['rssi_db'])[0]):.1f} "
                      f"first16={decoded_bytes[:16].hex()}")
            buf_free[i].set()
            i ^= 1
    except KeyboardInterrupt:
        print(f"\n[rx-pingpong-v2] stopped by user")

    elapsed = time.perf_counter() - t_start
    stop_event.set()
    for e in buf_free:  # wake capture process if it's blocked waiting on a slot
        e.set()

    try:
        capture_us, n_capture_waits = stats_q.get(timeout=5.0)
    except Exception:
        capture_us, n_capture_waits = [], 0
    cap_proc.join(timeout=3.0)
    if cap_proc.is_alive():
        cap_proc.terminate()

    for s in shms:
        s.close()
        s.unlink()

    cap_arr = np.array(capture_us) if capture_us else np.array([0.0])
    dec_arr = np.array(decode_us) if decode_us else np.array([0.0])
    total_samples_seen = n_buffers_seen * RX_SAMPLES
    coverage_pct = 100 * total_samples_seen / (elapsed * args.rate)
    n_over_budget = int(np.sum(dec_arr > BUDGET_US))

    print(f"\nbuffers decoded: {n_buffers_seen}  buffers captured: {len(cap_arr)}")
    print(f"  capture had to wait for a free slot: {n_capture_waits}x   decode had to wait for fresh data: {n_decode_waits}x")
    print(f"  decode alone over {BUDGET_US:.0f}us budget: {n_over_budget} ({100 * n_over_budget / max(n_buffers_seen, 1):.2f}%)")
    print(f"  rx.rx() alone:   min={cap_arr.min():8.1f}us  median={np.median(cap_arr):8.1f}us  "
          f"mean={cap_arr.mean():8.1f}us  max={cap_arr.max():8.1f}us")
    print(f"  rx_streaming():  min={dec_arr.min():8.1f}us  median={np.median(dec_arr):8.1f}us  "
          f"mean={dec_arr.mean():8.1f}us  max={dec_arr.max():8.1f}us")
    print(f"  samples actually looked at: {total_samples_seen} of {int(elapsed*args.rate)} "
          f"({coverage_pct:.2f}% real-time coverage)")
    print(f"  frames completed decode: {n_decoded}  crc_valid: {n_crc_valid}  "
          f"bytes exactly matched expected 64-byte payload: {n_bytes_match}  mismatched: {n_bytes_mismatch}")


if __name__ == "__main__":
    main()
