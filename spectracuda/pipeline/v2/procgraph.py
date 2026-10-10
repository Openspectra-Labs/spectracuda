"""`RxProcessGraph` -- the same three blocks, but one OS PROCESS each.

WHY THIS EXISTS, measured rather than assumed. `RxFlowgraph` runs the
stages as threads and they really do overlap (2.0-2.5x by the
sum-of-busy/wall measure). It is still slower than one thread, because
under CPython's GIL the per-call cost INFLATES when three threads
contend: with the identical workload and the identical number of calls,
`sync.process` took 0.014 s single-threaded and 0.068 s with the three
stages running. The work does not grow; each call just costs more. No
amount of batching fixes that, and this machine has no free-threaded
build to escape it with.

Processes do not share a GIL. The blocks were already built for this --
each stage owns all of its state and builds its own DSP objects from a
`PhyConfig` (see `env.py`), so a stage can be constructed inside a fresh
process with nothing shared. That is what makes this a different
executor rather than a different design.

WHAT CROSSES A PROCESS BOUNDARY, and why it is affordable. Only the FIFO
payloads: an `FftBatch`'s bins (16 symbols x 256 complex64 = 8 KiB) and
an `LlrBatch`'s bits. Samples never cross -- TD keeps them, exactly as in
the threaded graph. Per 16-frame run that is of the order of a hundred
transfers of a few KiB, which is cheap next to the DSP in each one; if
that ever stops being true, the fix is shared memory, not a redesign.

`PhyConfig` is sent instead of `PhyEnv` deliberately: configuration is
picklable primitives, whereas the DSP objects are not worth pickling and
must not be shared anyway.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _queue
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .env import PhyConfig, PhyEnv
from .rx_bit_domain import RxBitDomain
from .rx_freq_domain import RxFreqDomain
from .rx_time_domain import RxTimeDomain, TdStream
from .stage_if import (FftBatch, FrameAbort, FrameDone, LlrBatch, SoftConfig,
                       SymbolType)

_STOP = "__stop__"


# ---------------------------------------------------------------- TD ----

def _td_worker(cfg: PhyConfig, chunk_symbols: int,
               q_in: Any, q_out: Any, q_done: Any, q_ctl: Any) -> None:
    """Samples in, `FftBatch`es out. Owns the sample buffer."""
    env = PhyEnv.build(cfg)
    td = RxTimeDomain(env)
    stream = TdStream(td)
    state: Dict[int, int] = {}            # frame_id -> symbols emitted
    finished: set = set()

    def region_end(offset: int) -> int:
        if offset < env.n_training_symbols:
            return env.n_training_symbols
        h = env.n_training_symbols + env.num_symbols_header
        return h if offset < h else -1

    def stype_for(offset: int) -> SymbolType:
        if offset < env.n_training_symbols:
            return SymbolType.TRAIN
        if offset < env.n_training_symbols + env.num_symbols_header:
            return SymbolType.HEADER
        return SymbolType.BODY

    def pump() -> None:
        while True:
            try:
                finished.add(q_done.get_nowait())
            except _queue.Empty:
                break
        for fid in list(state):
            if fid in finished or stream.capped(fid, state[fid]):
                stream.retire(fid)
                state.pop(fid, None)
                continue
            while True:
                off = state[fid]
                end = region_end(off)
                room = chunk_symbols if end < 0 else min(chunk_symbols, end - off)
                limit = stream.symbol_limit(fid)
                if limit >= 0:
                    room = min(room, max(0, limit - off))
                    if room <= 0:
                        stream.retire(fid)
                        state.pop(fid, None)
                        break
                n = stream.ready_count(fid, off, room)
                if n <= 0:
                    break
                try:
                    q_out.put(stream.symbols(fid, off, n, stype_for(off)))
                except (ValueError, KeyError):
                    q_ctl.put(FrameAbort(frame_id=fid, reason="td extract"))
                    state.pop(fid, None)
                    break
                state[fid] = off + n

    while True:
        try:
            item = q_in.get(timeout=0.01)
        except _queue.Empty:
            pump()
            continue
        if isinstance(item, str) and item == _STOP:
            pump()
            break
        for fid in stream.feed(item):
            state[fid] = 0
        for ev in td.drain_control():
            q_ctl.put(ev)
        pump()
    q_out.put(_STOP)


# ---------------------------------------------------------------- FD ----

def _fd_worker(cfg: PhyConfig, soft: SoftConfig, expected_fec: tuple,
               q_in: Any, q_out: Any, q_ctl_in: Any, q_ctl_out: Any) -> None:
    """Bins in, LLRs out. Decodes the header itself -- no backward path."""
    env = PhyEnv.build(cfg)
    fd = RxFreqDomain(env, soft)
    fd.configure_expected_fec(*expected_fec)
    pending: Dict[int, List[FftBatch]] = {}

    while True:
        try:
            while True:
                ev = q_ctl_in.get_nowait()
                if isinstance(ev, FrameAbort):
                    fd.abort(ev.frame_id)
        except _queue.Empty:
            pass
        try:
            batch = q_in.get(timeout=0.01)
        except _queue.Empty:
            continue
        if isinstance(batch, str) and batch == _STOP:
            break
        try:
            if batch.stype == SymbolType.TRAIN:
                fd.training(batch)
            elif batch.stype == SymbolType.HEADER:
                acc = pending.setdefault(batch.frame_id, [])
                acc.append(batch)
                if sum(b.n_sym for b in acc) >= env.num_symbols_header:
                    merged = acc[0] if len(acc) == 1 else FftBatch(
                        frame_id=batch.frame_id, symbol_offset=acc[0].symbol_offset,
                        sample_offset=acc[0].sample_offset,
                        bins=env.xp.concatenate([b.bins for b in acc], axis=0),
                        stype=SymbolType.HEADER)
                    pending.pop(batch.frame_id, None)
                    fd.decode_header(merged)
            else:
                for llr in fd.body(batch):
                    q_out.put(llr)
        except (ValueError, NotImplementedError, KeyError, RuntimeError) as exc:
            pending.pop(batch.frame_id, None)
            q_ctl_out.put(FrameAbort(frame_id=batch.frame_id, reason=f"fd: {exc}"))
    q_out.put(_STOP)


# --------------------------------------------------------------- BIT ----

def _bit_worker(cfg: PhyConfig, q_in: Any, q_out: Any, q_done: Any,
                q_ctl_in: Any) -> None:
    """LLRs in, decoded bytes out. Accumulates to frame granularity."""
    env = PhyEnv.build(cfg)
    bd = RxBitDomain(env)
    while True:
        try:
            while True:
                ev = q_ctl_in.get_nowait()
                if isinstance(ev, FrameAbort):
                    bd.abort(ev.frame_id)
        except _queue.Empty:
            pass
        try:
            llr = q_in.get(timeout=0.01)
        except _queue.Empty:
            continue
        if isinstance(llr, str) and llr == _STOP:
            break
        try:
            result = bd.push(llr)
        except (ValueError, NotImplementedError):
            q_done.put(llr.frame_id)
            continue
        if result is not None:
            result["frame_id"] = llr.frame_id
            q_out.put(result)
            q_done.put(llr.frame_id)       # flow control: TD may drop samples
    q_out.put(_STOP)


# ------------------------------------------------------------- graph ----

class RxProcessGraph:
    """TD | FD | BIT, one process each, connected by queues.

    `fork` is used explicitly: the stages re-import nothing and numba's
    compiled kernels are inherited, so a worker starts in milliseconds
    rather than re-JITting.
    """

    def __init__(self, cfg: PhyConfig, *, soft: Optional[SoftConfig] = None,
                 expected_fec: tuple = ("rs_m8", "conv_v27"),
                 chunk_symbols: int = 16, queue_depth: int = 256) -> None:
        self.cfg = cfg
        ctx = mp.get_context("fork")
        self.q_iq = ctx.Queue(maxsize=queue_depth)
        self.q_i1 = ctx.Queue(maxsize=queue_depth)
        self.q_i2 = ctx.Queue(maxsize=queue_depth)
        self.q_out = ctx.Queue()
        self.q_done = ctx.Queue()
        self.q_ctl_fd = ctx.Queue()
        self.q_ctl_bit = ctx.Queue()
        soft = soft or SoftConfig()
        self.procs = [
            ctx.Process(target=_td_worker, name="rx-td", daemon=True,
                        args=(cfg, chunk_symbols, self.q_iq, self.q_i1,
                              self.q_done, self.q_ctl_fd)),
            ctx.Process(target=_fd_worker, name="rx-fd", daemon=True,
                        args=(cfg, soft, expected_fec, self.q_i1, self.q_i2,
                              self.q_ctl_fd, self.q_ctl_bit)),
            ctx.Process(target=_bit_worker, name="rx-bit", daemon=True,
                        args=(cfg, self.q_i2, self.q_out, self.q_done,
                              self.q_ctl_bit)),
        ]

    def start(self) -> "RxProcessGraph":
        for p in self.procs:
            p.start()
        return self

    def feed(self, chunk: Any) -> None:
        self.q_iq.put(chunk)

    def collect(self, expect: int, timeout: float = 60.0) -> List[Dict[str, Any]]:
        """Wait for `expect` frames, sorted by frame_id.

        Sorted, never in completion order -- completion order is
        timing-dependent even though the output is not.
        """
        out: Dict[int, Dict[str, Any]] = {}
        deadline = time.monotonic() + timeout
        while len(out) < expect and time.monotonic() < deadline:
            try:
                item = self.q_out.get(timeout=0.05)
            except _queue.Empty:
                continue
            if isinstance(item, str):
                break
            out[item["frame_id"]] = item
        return [out[k] for k in sorted(out)]

    def stop(self) -> None:
        try:
            self.q_iq.put(_STOP)
        except Exception:
            pass
        for p in self.procs:
            p.join(timeout=3.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
