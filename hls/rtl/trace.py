"""Capture spectracuda's RX intermediates at every block boundary.

This does NOT reimplement the receive chain. It wraps the real block
objects' methods on a live Ofdm instance and then calls the library's own
rx_process() unchanged, so whatever lands in the trace is by construction
what the pipeline actually computed -- a re-expressed chain could drift
from rx_process() and we would be chasing our own paraphrase instead of
the golden model.

Frame is CLEAN: no CFO, no AWGN. That is deliberate. The first end-to-end
run is looking for plumbing bugs (Q-format, bit order, valid timing,
symbol boundaries) between blocks that were each verified in isolation;
mixing in channel impairments would make a convention bug and a DSP bug
indistinguishable. Impairments come after this passes.

Usage:
    python -m hls.rtl.trace
"""
from __future__ import annotations

import json
import os
import types
from typing import Any, Dict, List

import numpy as np

# The pinned golden model, NOT the working-tree spectracuda -- see golden_ref.py.
from hls.rtl import golden_ref
golden_ref.use()

from spectracuda.fec.fec import FEC
from spectracuda.interleaver.base import _PermutationInterleaverBase
from spectracuda.modem.mapper import Modem
from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel
from hls.gen.golden import CFG, GUARD, PAYLOAD_BITS, SEED

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "build", "trace")


class Recorder:
    """Wraps bound methods on real block objects and logs every call."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self._undo: List[tuple] = []

    def hook_class(self, cls: Any, name: str, label: str) -> None:
        """Hook an unbound method so EVERY instance is captured.

        The payload modem and both FEC codecs are constructed inside
        rx_process() from the DECODED header -- they do not exist yet when
        we set hooks up, so instance hooks miss them entirely.
        """
        orig = getattr(cls, name)

        def wrapper(inner_self, *a, **kw):
            out = orig(inner_self, *a, **kw)
            self.events.append({
                "label": f"{label}:{getattr(inner_self, 'scheme', None) or type(inner_self).__name__}",
                "args": [_snap(x) for x in a],
                "kwargs": {k: _snap(v) for k, v in kw.items()},
                "out": _snap(out),
            })
            return out

        setattr(cls, name, wrapper)
        self._undo.append((cls, name, orig))

    def hook(self, obj: Any, name: str, label: str) -> None:
        if obj is None or not hasattr(obj, name):
            return
        orig = getattr(obj, name)

        def wrapper(*a, **kw):
            out = orig(*a, **kw)
            self.events.append({
                "label": label,
                "args": [_snap(x) for x in a],
                "kwargs": {k: _snap(v) for k, v in kw.items()},
                "out": _snap(out),
            })
            return out

        setattr(obj, name, wrapper)
        self._undo.append((obj, name, orig))

    def restore(self) -> None:
        for obj, name, orig in self._undo:
            setattr(obj, name, orig)
        self._undo.clear()


def _snap(x: Any) -> Any:
    """Detach a value from the pipeline so later in-place work can't edit it."""
    if isinstance(x, dict):
        return {k: _snap(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_snap(v) for v in x)
    # NOT duck-typed on shape/dtype: the numpy MODULE itself has both
    # (np.shape, np.dtype), and Grid.extract_data(xp, grid) is called with
    # xp as its first argument -- np.array(module) yields a 0-d object
    # array that np.save then chokes on.
    if isinstance(x, types.ModuleType):
        return repr(x)
    if isinstance(x, np.ndarray):
        return np.array(x, copy=True)
    if type(x).__module__.startswith("cupy"):
        return np.asarray(x.get() if hasattr(x, "get") else x)
    return x


def _flatten(prefix: str, val: Any):
    """Yield (name, leaf) for arrays/scalars nested in dicts or tuples."""
    if isinstance(val, dict):
        for k, v in val.items():
            yield from _flatten(f"{prefix}.{k}", v)
    elif isinstance(val, (list, tuple)):
        for j, v in enumerate(val):
            yield from _flatten(f"{prefix}.{j}", v)
    else:
        yield prefix, val


def _jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return repr(x)


def clean_frame(ofdm: Ofdm, seed: int, cfo: float = 0.0):
    """One frame, guard-padded.

    cfo=0 gives a bit-perfect frame: that is the right default, because
    the first chain run is hunting plumbing bugs and any impairment makes
    a convention bug and a DSP bug hard to tell apart. But at cfo=0 the
    CFO blocks are a trivial test -- estimate is ~0 and correction is the
    identity -- so pass a real cfo to exercise them. AWGN stays off
    either way: CFO is deterministic, noise is not.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=PAYLOAD_BITS).astype(np.uint8)
    tx = np.asarray(ofdm.generate_frame(bits))
    pad = np.zeros((tx.shape[0], GUARD), dtype=tx.dtype)
    padded = np.concatenate([pad, tx, pad], axis=-1)
    if cfo:
        ch = Channel(cfo=cfo, cfo_fft_size=CFG["fft_size"], seed=seed)
        padded = np.asarray(ch.process(padded))
    return padded, bits


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfo", type=float, default=0.0,
                    help="CFO as a fraction of subcarrier spacing (no AWGN either way)")
    ap.add_argument("--out", default=None, help="output dir (default build/trace[_cfo])")
    args = ap.parse_args()

    global OUT
    OUT = args.out or (os.path.join(HERE, "build", "trace_cfo" if args.cfo else "trace"))
    os.makedirs(OUT, exist_ok=True)
    ofdm = Ofdm(**CFG)
    rx, tx_bits = clean_frame(ofdm, SEED, cfo=args.cfo)

    rec = Recorder()
    rec.hook(ofdm.sync, "process", "sync")
    rec.hook(ofdm.cfo, "process", "cfo_estimate")
    rec.hook(ofdm.cfo, "correct", "cfo_correct")
    rec.hook(ofdm.demod, "process", "fft")
    rec.hook(ofdm.channel_estimator, "process", "chanest")
    rec.hook(ofdm.equalizer, "process", "equalize")
    rec.hook(ofdm.grid, "extract_data", "grid_data")
    rec.hook(ofdm.grid, "extract_pilots", "grid_pilots")
    # Class-level: these objects are built from the decoded header.
    rec.hook_class(Modem, "demodulate", "demod_bits")
    rec.hook_class(Modem, "demodulate_stats", "demod_stats")
    rec.hook_class(FEC, "decode", "fec")
    rec.hook_class(_PermutationInterleaverBase, "decode", "deinterleave")
    try:
        res = ofdm.rx_process(rx)
    finally:
        rec.restore()

    # Self-check: the hooks must not have perturbed the pipeline.
    rx_bits = np.asarray(res["bits"]).ravel().astype(np.uint8)[:PAYLOAD_BITS]
    ok_bits = bool(np.array_equal(rx_bits, tx_bits))
    crc = res["crc_valid"]
    crc_ok = bool(np.asarray(crc).ravel()[0]) if crc is not None else None

    counts: Dict[str, int] = {}
    manifest: List[Dict[str, Any]] = []
    for e in rec.events:
        i = counts.get(e["label"], 0)
        counts[e["label"]] = i + 1
        stem = f"{e['label']}_{i}"
        entry: Dict[str, Any] = {"label": e["label"], "call": i, "files": {}}
        slots = [("out", e["out"])]
        slots += [(f"in{j}", a) for j, a in enumerate(e["args"])]
        # kwargs matter: Equalizer.process(data, channel_est=h) passes the
        # channel estimate by keyword, so dropping kwargs loses an input.
        slots += [(f"kw.{k}", v) for k, v in e["kwargs"].items()]
        for slot, val in slots:
            for key, arr in _flatten(slot, val):
                if isinstance(arr, np.ndarray):
                    fn = f"{stem}.{key}.npy"
                    np.save(os.path.join(OUT, fn), arr)
                    entry["files"][key] = {"file": fn, "shape": list(arr.shape),
                                           "dtype": str(arr.dtype)}
                else:
                    entry["files"][key] = {"value": _jsonable(arr)}
        manifest.append(entry)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    np.save(os.path.join(OUT, "rx_iq.npy"), rx[0])
    # Same dumb "I Q" per line format golden/sc_sync_rx.txt uses, so
    # gen_stimulus.py can consume it unmodified -- one conversion path,
    # so the RTL and Python cannot disagree about scaling by accident.
    with open(os.path.join(OUT, "rx_iq.txt"), "w") as f:
        for z in np.asarray(rx[0]).ravel():
            f.write(f"{float(np.real(z)):.9g} {float(np.imag(z)):.9g}\n")
    np.save(os.path.join(OUT, "tx_bits.npy"), tx_bits)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump({
            "config": CFG, "seed": SEED, "guard": GUARD,
            "n_samples": int(rx.shape[-1]),
            "cfo": args.cfo,
            "awgn": False,
            "frame_found": bool(res["frame_found"]),
            "crc_valid": crc_ok,
            "payload_bits_match_tx": ok_bits,
            "boundary_calls": counts,
            "header": {k: _jsonable(v) for k, v in (res.get("header") or {}).items()},
        }, f, indent=2)

    print(f"frame: {rx.shape[-1]} samples, guard={GUARD}, cfo={args.cfo}, awgn=off")
    print(f"  frame_found      = {res['frame_found']}")
    print(f"  crc_valid        = {crc_ok}")
    print(f"  payload == tx    = {ok_bits}")
    print("  boundary calls captured:")
    for k in sorted(counts):
        print(f"    {k:14s} {counts[k]}")
    if not (ok_bits and crc_ok):
        raise SystemExit("FAIL: clean frame did not decode -- fix before chaining RTL")
    print("\nOK: clean frame decodes in pure Python. Trace boundaries captured.")


if __name__ == "__main__":
    main()
