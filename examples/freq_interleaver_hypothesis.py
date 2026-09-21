#!/usr/bin/env python3
"""Does a FREQUENCY interleaver (between conv and the mapper) recover the
broad-fade case? Characterization only -- production PHY untouched.

The hypothesis, from docs/2026-09-21-multipath-severity-characterization.md:

  A 1-sample echo puts ONE wide null across the band (17% of subcarriers
  faded, widest run 36 bins). At 216 data subcarriers x 4 bits that is
  ~144 CONSECUTIVE damaged coded bits out of 864 per OFDM symbol. The
  chain is RS -> interleaver -> conv -> 16QAM, so on receive it is
  Viterbi -> deinterleave -> RS: the interleaver sits on the WRONG SIDE
  of the convolutional code and cannot touch its input. conv_v27 is K=7
  with a ~35-49 bit traceback, so a 144-bit burst leaves no reliable
  observation anywhere in the window and the trellis loses the path --
  measured as Viterbi AMPLIFYING, 0.052 raw BER in, 0.071 out.

  Interleaved, the same 144 bad bits spread over 864 give ~1 in 6, so a
  49-bit window holds ~8 bad bits: the same error COUNT, in the regime a
  rate-1/2 K=7 code handles easily.

This applies the permutation as a test-only wrapper around the existing
packetizer, symmetric on both sides, so nothing in the wire format or the
production chain changes. Two depths:

  symbol  -- permute within ONE OFDM symbol's coded bits (what 802.11a/g
             does; bounded latency, implementable)
  frame   -- permute across the whole payload (maximum dispersion, worse
             latency; an upper bound on what interleaving can buy)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel

import dmrs_doppler_study as H

OUT = Path("debug/freq_interleaver")
FS, PAYLOAD_BYTES, SNR_DB = 20e6, 5575, 15.0
TAIL, NOISE_LEN = 4096, 300_000


def perm_for(n_bits, depth, bits_per_symbol=4, n_data=216):
    """Block permutation. `depth` is the block size in coded bits; a
    classic write-rows / read-columns interleaver over that block, which
    sends adjacent coded bits to widely separated subcarriers."""
    if depth == "symbol":
        block = n_data * bits_per_symbol          # 864 for 216 x 16QAM
    else:
        block = n_bits
    rows = int(np.sqrt(block))
    while block % rows:
        rows -= 1
    cols = block // rows
    base = np.arange(block).reshape(rows, cols).T.reshape(-1)
    full = np.arange(n_bits)
    n_full = (n_bits // block) * block
    for start in range(0, n_full, block):
        full[start:start + block] = start + base
    return full


class _Wrapped:
    """Permutes the packetizer's encoded output and un-permutes decode's
    input. Symmetric, so it is a pure re-ordering of the SAME bits -- no
    redundancy added, no wire-format change beyond the ordering itself."""

    def __init__(self, pk, depth):
        self._pk = pk
        self._depth = depth
        self._p = None

    def __getattr__(self, name):
        return getattr(self._pk, name)

    def _perm(self, n):
        if self._p is None or self._p.shape[0] != n:
            self._p = perm_for(n, self._depth)
        return self._p

    def encode(self, bits):
        out = np.asarray(self._pk.encode(bits))
        p = self._perm(out.shape[-1])
        return out[:, p]

    def decode(self, bits, soft=None):
        bits = np.asarray(bits)
        p = self._perm(bits.shape[-1])
        inv = np.empty_like(p)
        inv[p] = np.arange(p.shape[0])
        un = bits[:, inv]
        un_soft = None if soft is None else np.asarray(soft)[:, inv]
        return self._pk.decode(un, soft=un_soft)


_DEPTH = {"mode": None}
_orig_hdr = Ofdm._decode_header_from_sync
_orig_gen = Ofdm.generate_frame


def _hdr(self, rx_iq, start_index, n_payload_symbols=None):
    h = _orig_hdr(self, rx_iq, start_index, n_payload_symbols)
    if _DEPTH["mode"] is not None:
        h["payload_packetizer"] = _Wrapped(h["payload_packetizer"], _DEPTH["mode"])
    return h


def _gen(self, bits, **kw):
    if _DEPTH["mode"] is None:
        return _orig_gen(self, bits, **kw)
    real = self.packetizer
    self.packetizer = _Wrapped(real, _DEPTH["mode"])
    try:
        return _orig_gen(self, bits, **kw)
    finally:
        self.packetizer = real


Ofdm._decode_header_from_sync = _hdr
Ofdm.generate_frame = _gen


def make(soft, iv=32):
    o = Ofdm(fft_size=256, n_pilot=8, n_data=216, cp_len=64, modem="qam16",
             fec="rs_m8", fec1="conv_v27",
             interleaver="block", interleaver_kwargs={"unit_bits": 8},
             crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
             n_training_symbols=2, dmrs_interval=iv, soft_decision=soft)
    o.MAX_PAYLOAD_SYMBOLS = 256
    return o


def run(a, delay_ns, soft, depth, trials=40, seed0=9000, iv=32):
    bits = np.random.default_rng(1).integers(
        0, 2, size=(1, PAYLOAD_BYTES * 8)).astype("uint8")
    ok = 0
    _DEPTH["mode"] = depth
    try:
        with H.interval_ctx(iv):
            for t in range(trials):
                rng = np.random.default_rng(seed0 + t)
                taps, dop, _ = Channel.paths_to_taps([
                    {"amplitude": 1.0, "delay_ns": 0},
                    {"amplitude": a, "delay_ns": delay_ns,
                     "phase_rad": float(rng.uniform(0, 2 * np.pi))}], FS)
                o = make(soft, iv)
                rx = Channel(snr_db=SNR_DB, multipath_taps=taps, tap_doppler_hz=dop,
                             sample_rate_hz=FS, tail_samples=TAIL,
                             noise_draw_len=NOISE_LEN, seed=seed0 + t,
                             backend="numpy").process(o.generate_frame(bits))
                try:
                    r = o.rx_process(rx)
                    ok += int(bool(np.asarray(r["crc_valid"])[0]))
                except ValueError:
                    pass
    finally:
        _DEPTH["mode"] = None
    return ok


CASES = [("a=0.6 d=50ns  (the residual)", 0.6, 50),
         ("a=0.6 d=200ns", 0.6, 200),
         ("a=0.6 d=1000ns (already scattered)", 0.6, 1000),
         ("a=0.8 d=50ns", 0.8, 50),
         ("a=0.2 d=100ns (control, clean)", 0.2, 100)]

def dispersion(n_bits=102012, n_data=216, bits_per_symbol=4, faded_bins=36):
    """What the de-interleaver actually hands Viterbi, measured rather than
    argued. Damage enters in MAPPED order as one contiguous run of faded
    subcarriers per OFDM symbol; the inverse permutation is what the
    decoder sees."""
    per_sym = n_data * bits_per_symbol
    bad = faded_bins * bits_per_symbol
    mapped = np.zeros(n_bits, dtype=bool)
    for start in range(0, n_bits - per_sym, per_sym):
        mapped[start:start + bad] = True

    def runs(m):
        m = np.asarray(m).astype(int)
        if m.sum() == 0:
            return 0, 0.0
        d = np.diff(np.concatenate([[0], m, [0]]))
        L = np.where(d == -1)[0] - np.where(d == 1)[0]
        return int(L.max()), float(L.mean())

    out = [dict(depth="none", geometry="-", max_run=runs(mapped)[0],
                mean_run=runs(mapped)[1])]
    for depth in ("symbol", "frame"):
        p = perm_for(n_bits, depth)
        conv = np.zeros(n_bits, dtype=bool)
        conv[p[mapped]] = True
        block = per_sym if depth == "symbol" else n_bits
        rows = int(np.sqrt(block))
        while block % rows:
            rows -= 1
        mx, mn = runs(conv)
        out.append(dict(depth=depth, geometry=f"{rows}x{block//rows}",
                        max_run=mx, mean_run=mn))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    args = ap.parse_args()
    print("Burst length Viterbi sees (conv_v27 is K=7, traceback ~35-49 bits):")
    for d in dispersion():
        print(f"  {d['depth']:>7} {d['geometry']:>10}  max run {d['max_run']:>4}  "
              f"mean {d['mean_run']:.1f}")
    print()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    print(f"{'case':>34} {'soft':>5} | {'none':>7} {'symbol':>7} {'frame':>7}")
    for label, a, dns in CASES:
        for soft in (False, True):
            got = {}
            for depth in (None, "symbol", "frame"):
                got[str(depth)] = run(a, dns, soft, depth, trials=args.trials)
            rows.append(dict(label=label, a=a, delay_ns=dns, soft=soft,
                             trials=args.trials, **got))
            print(f"{label:>34} {str(soft):>5} | "
                  f"{got['None']:>3}/{args.trials:<3} {got['symbol']:>3}/{args.trials:<3} "
                  f"{got['frame']:>3}/{args.trials:<3}", flush=True)
    (OUT / "results.json").write_text(json.dumps(rows, indent=1, default=float))
    print(f"\nwrote {OUT/'results.json'}  [{time.time()-t0:.0f}s]")
