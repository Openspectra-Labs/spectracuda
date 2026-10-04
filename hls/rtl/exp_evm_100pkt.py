"""100 packets x EVM sweep, 2048 bytes, QAM64, fabric config (no RS/CRC).

Parallel: every frame at a given payload size has the SAME sample count,
so one compiled binary per worker serves all its jobs -- only the
stimulus file contents change. Each worker gets its own stim/unit paths
because those are compile-time defines in the testbench.
"""
from __future__ import annotations

import os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

# The pinned golden model, NOT the working-tree spectracuda -- see golden_ref.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden_ref  # noqa: E402
golden_ref.use()
from spectracuda.interleaver.base import _PermutationInterleaverBase  # noqa: E402
from spectracuda.pipeline import Ofdm                                 # noqa: E402

HERE  = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")
GUARD, FS = 200, (1 << 15) - 1
NBYTES, PN_ALPHA, WORKERS = 2048, 0.9995, 8
EVMS = [0.12, 0.14, 0.16, 0.18]
NPKT = 100

SRCS = ["tb/rx_top_tb.v", "src/rx_top.v", "src/rx_time_domain.v", "src/rx_freq_domain.v", "src/sync_fifo_fwft.v", "src/rx_bit_domain.v", "src/sc_sync_rtl.v", "src/frame_sync.v",
        "src/cfo_estimate.v", "src/cfo_correct.v", "src/cordic_rot.v",
        "src/cordic_vec.v", "src/cp_fft.v", "src/grid_extract.v",
        "src/ls_chanest.v", "src/mmse_eq.v", "src/pilot_cpe.v",
        "src/header_decode.v", "src/demapper.v", "src/viterbi_dec.v",
        "src/deinterleaver.v", "tb/stubs/xfft_256.v"]


def cfg():
    return dict(fft_size=256, cp_len=32, n_data=216, n_pilot=8,
                sync="schmidl_cox", cfo="schmidl_cox",
                channel_estimator="ls", equalizer="mmse",
                modem="qam64", fec1="conv_v27",
                interleaver="block", interleaver_kwargs={"unit_bits": 8})


def impair(rx, evm, seed):
    part = evm / np.sqrt(2.0)
    n = rx.shape[-1]
    sig = float(np.sqrt(np.mean(np.abs(rx) ** 2)))
    nr = np.random.default_rng(seed + 500)
    sigma = part * sig
    rx = rx + (nr.normal(0, sigma / np.sqrt(2), n)
               + 1j * nr.normal(0, sigma / np.sqrt(2), n))[None, :]
    w = np.random.default_rng(seed + 1000).normal(0.0, 1.0, n)
    acc, drive = 0.0, part * np.sqrt(1.0 - PN_ALPHA ** 2)
    phi = np.empty(n)
    for i in range(n):
        acc = PN_ALPHA * acc + drive * w[i]
        phi[i] = acc
    return rx * np.exp(1j * phi)[None, :]


# The worker slot must belong to the PROCESS, not to the job index.
# ProcessPoolExecutor.map hands jobs to whichever process is free, so
# labelling jobs with i % WORKERS let two processes share the same
# stimulus and output files -- they clobbered each other mid-read and the
# unit file came back full of NULs.
_SLOT = None


def _init(q):
    global _SLOT
    _SLOT = q.get()


def job(args):
    evm, seed = args
    wid = _SLOT
    o = Ofdm(**cfg())
    nbits = NBYTES * 8
    bits = np.random.default_rng(seed).integers(0, 2, size=(1, nbits)).astype("uint8")
    tx = np.asarray(o.generate_frame(bits))
    pad = np.zeros((1, GUARD), dtype=tx.dtype)
    rx = impair(np.concatenate([pad, tx, pad], axis=-1), evm, seed)

    seen = {}
    orig = _PermutationInterleaverBase.decode
    def hook(self, b):
        out = orig(self, b)
        seen.setdefault("out", np.array(out, copy=True))
        return out
    _PermutationInterleaverBase.decode = hook
    try:
        res = o.rx_process(rx)
    except Exception:
        _PermutationInterleaverBase.decode = orig
        return (evm, seed, False, False, float("nan"))
    finally:
        _PermutationInterleaverBase.decode = orig

    if not res["frame_found"] or "out" not in seen:
        return (evm, seed, False, False, float("nan"))

    py_ok = bool(np.array_equal(np.asarray(res["bits"]).ravel()[:nbits], bits.ravel()))
    meas = float(np.asarray(res["evm"]).ravel()[0])
    py_units = np.packbits(seen["out"].ravel().astype(np.uint8))

    stim = os.path.join(BUILD, f"w{wid}_stim.hex")
    unitp = os.path.join(BUILD, f"w{wid}_units.txt")
    with open(stim, "w") as f:
        for z in rx[0]:
            i = int(np.clip(round(float(np.real(z)) * FS), -32768, 32767))
            q = int(np.clip(round(float(np.imag(z)) * FS), -32768, 32767))
            f.write(f"{((i & 0xFFFF) << 16) | (q & 0xFFFF):08x}\n")
    subprocess.run([os.path.join(BUILD, f"vsim_w{wid}", f"vsim{wid}")],
                   cwd=HERE, capture_output=True, text=True, check=True)
    raw = [l.split() for l in open(unitp) if l.strip()]
    vals = [int(r[0]) for r in raw if r[0] != "OVF"]
    rtl_units = np.array(vals, dtype=int)
    m = min(len(rtl_units), len(py_units))
    rtl_ok = (len(rtl_units) == len(py_units)) and m > 0 and \
             bool(np.array_equal(rtl_units[:m], py_units[:m]))
    return (evm, seed, py_ok, rtl_ok, meas)


def main() -> None:
    o = Ofdm(**cfg())
    nbits = NBYTES * 8
    probe = np.asarray(o.generate_frame(
        np.zeros((1, nbits), dtype="uint8")))
    nsamp = probe.shape[-1] + 2 * GUARD
    enc = o.packetizer.encoded_length(nbits)
    n_units = (enc // 2 + 6) // 8          # viterbi output bytes
    # exact figure from one real decode
    rx0 = impair(np.concatenate(
        [np.zeros((1, GUARD), dtype=probe.dtype), probe,
         np.zeros((1, GUARD), dtype=probe.dtype)], axis=-1), 0.01, 7)
    seen = {}
    orig = _PermutationInterleaverBase.decode
    def hook(self, b):
        out = orig(self, b); seen.setdefault("in", np.array(b, copy=True)); return out
    _PermutationInterleaverBase.decode = hook
    try:
        o.rx_process(rx0)
    finally:
        _PermutationInterleaverBase.decode = orig
    n_units = seen["in"].ravel().size // 8
    M = 1 + int(np.floor(np.sqrt(n_units))); N = -(-n_units // M)
    print(f"2048 B qam64: {nsamp} samples, enc={enc}, di_units={n_units} "
          f"(M={M} N={N}), {WORKERS} workers")

    for w in range(WORKERS):
        with open(os.path.join(BUILD, f"w{w}_params.vh"), "w") as f:
            f.write(f'`define RXT_STIM_PATH "{os.path.join(BUILD, f"w{w}_stim.hex")}"\n')
            f.write(f'`define RXT_HDR_PATH "{os.path.join(BUILD, f"w{w}_hdr.txt")}"\n')
            f.write(f'`define RXT_UNIT_PATH "{os.path.join(BUILD, f"w{w}_units.txt")}"\n')
            f.write(f"`define RXT_NSAMP {nsamp}\n`define RXT_DRAIN 400000\n")
            f.write(f"`define RXT_ENC_BITS {enc}\n`define RXT_DI_UNITS {n_units}\n")
            f.write(f"`define RXT_DI_ROWS {M}\n`define RXT_DI_COLS {N}\n")
        tb = os.path.join(BUILD, f"w{w}_tb.v")
        src = open(os.path.join(HERE, "tb/rx_top_tb.v")).read()
        src = src.replace('`include "build/rxtop_tb_params.vh"',
                          f'`include "build/w{w}_params.vh"')
        src = src.replace("module rx_top_tb", f"module rx_top_tb{w}")
        open(tb, "w").write(src)
        r = subprocess.run(
            ["verilator", "--binary", "--timing", "-Wno-WIDTHEXPAND",
             "-Wno-WIDTHTRUNC", "-Isrc/generated", "-Isrc", "-I.",
             "--top-module", f"rx_top_tb{w}", "-o", f"vsim{w}",
             "--Mdir", os.path.join(BUILD, f"vsim_w{w}")]
            + [f"build/w{w}_tb.v"] + SRCS[1:], cwd=HERE,
            capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
            raise SystemExit(f"build failed for worker {w}")
    print("workers built")

    jobs = [(e, s) for e in EVMS for s in range(NPKT)]
    out = []
    q = mp.Queue()
    for w in range(WORKERS):
        q.put(w)
    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=_init, initargs=(q,)) as ex:
        for k, r in enumerate(ex.map(job, jobs), 1):
            out.append(r)
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)} done", flush=True)

    print(f"\n{'EVM':>5} {'meas':>7} {'python OK':>10} {'rtl==py':>9} "
          f"{'rtl OK':>8}")
    print("-" * 45)
    for e in EVMS:
        rows = [r for r in out if r[0] == e]
        meas = np.nanmean([r[4] for r in rows])
        pyok = sum(1 for r in rows if r[2])
        match = sum(1 for r in rows if r[3])
        rtlok = sum(1 for r in rows if r[2] and r[3])
        print(f"{e:>5.2f} {meas:>7.4f} {pyok:>7}/{len(rows)} "
              f"{match:>6}/{len(rows)} {rtlok:>5}/{len(rows)}")


if __name__ == "__main__":
    main()
