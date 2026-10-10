"""`RxFlowgraph` -- the three stages as independent workers behind bounded
FIFOs, one thread each.

This is the point of V2. Everything before it existed to make this safe:
the partition is bit-exact (step 1), the streaming FSM is robust (step 2),
and no stateful block is shared between stages (step 3). What is added
here is concurrency and nothing else, so a failure now is a concurrency
failure.

WHY THE STAGES DID NOT HAVE TO CHANGE. `RxPipeline` moved batches between
them with a plain loop; this moves them through queues on separate
threads. The stages are identical either way, which is what the forward-
only dataflow bought.

THE ONE THING THREADING REALLY CHANGES: TD MUST PUSH.

Under `RxPipeline`, FD knew `n_total_slots` from the header and the loop
asked TD for exactly that many symbols. A worker cannot be asked -- it
has to decide. And TD cannot know how long a frame is, because the length
is in the header, which FD reads. So TD pushes symbols until either:

* `TdStream.capped()` -- `max_payload_symbols` past the header, a bound
  that makes termination independent of every other stage; or
* a `FrameDone` arrives from BIT, which is FLOW CONTROL ("nobody needs
  these samples any more"), carries no decode information, and is purely
  an optimization -- without it TD simply runs to the cap.

FD already discards body symbols past `n_total_slots`, so the overshoot
is wasted work and never wrong output. This is where `capped()` finally
earns its place; under the single-threaded driver nothing ever reached it.

BOUNDED QUEUES, AND WHY OVERFLOW IS FRAME-SCOPED. An unbounded queue
turns a slow stage into a memory leak instead of visible backpressure. But
dropping an individual batch from the middle of a frame would hand a
silently corrupted partial frame downstream, which is a correctness
defect rather than a performance one. So overflow aborts the WHOLE frame
the batch belonged to.

CONTROL IS A SEPARATE CHANNEL, AND THEREFORE UNORDERED. If aborts shared
the data queue, a full queue would block the very abort the full queue
caused. Because they do not share it, an abort for frame N can overtake
some of frame N's data -- so every consumer keeps an invalidated-id set
and discards arrivals, rather than only dropping what it already holds.
FD and BIT both implement that; see their `abort()`.

DETERMINISM IS A REQUIREMENT, NOT A HOPE. Identical input must give
byte-identical output on every run, including under stalls. It holds
because each frame is processed independently, each stage keeps per-frame
state keyed by `frame_id`, and the queues preserve order within a stage.
Results are returned sorted by `frame_id` so a caller never sees
completion order, which is genuinely timing-dependent.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .env import PhyEnv
from .rx_bit_domain import RxBitDomain
from .rx_freq_domain import RxFreqDomain
from .rx_time_domain import RxTimeDomain, TdStream
from .stage_if import (FftBatch, FrameAbort, FrameDone, LlrBatch, SoftConfig,
                       StreamGap, SymbolType)

_SHUTDOWN = object()


class CountingQueue:
    """A bounded queue that records what a pipeline operator needs to see.

    Depth, high-water and drops are not optional extras: they are the
    only way to tell "the pipeline is slow" from "one stage is the
    bottleneck", and retrofitting them means the first measurements are
    the ones nobody can explain. So they ship with the queue.
    """

    def __init__(self, name: str, maxsize: int) -> None:
        self.name = name
        self.maxsize = maxsize
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self.stats = {"put": 0, "got": 0, "dropped": 0, "high_water": 0,
                      "blocked_puts": 0}

    def _note_depth(self) -> None:
        d = self._q.qsize()
        if d > self.stats["high_water"]:
            self.stats["high_water"] = d

    def put_blocking(self, item: Any) -> None:
        """For TX-style paths: never lose data, apply backpressure."""
        if self._q.full():
            with self._lock:
                self.stats["blocked_puts"] += 1
        self._q.put(item)
        with self._lock:
            self.stats["put"] += 1
            self._note_depth()

    def put_or_drop(self, item: Any) -> bool:
        """Drop rather than block -- correct ONLY where data would
        otherwise be lost, i.e. at the radio input, where nothing is
        holding the samples for us.

        Returns False when the item was refused, and the caller must then
        abort that item's whole frame rather than carry on with a hole
        in it.
        """
        try:
            self._q.put_nowait(item)
        except queue.Full:
            with self._lock:
                self.stats["dropped"] += 1
            return False
        with self._lock:
            self.stats["put"] += 1
            self._note_depth()
        return True

    def put_backpressure(self, item: Any, stop: threading.Event,
                         timeout: float = 0.05) -> bool:
        """Wait for room -- the right policy BETWEEN stages.

        A producing stage still holds its input (TD's samples are in its
        own buffer), so stalling loses nothing while dropping destroys a
        frame. An earlier version dropped here, and TD -- which runs to
        its cap and so overshoots every short frame -- filled the queue
        and aborted real frames with `i1 overflow`. Returns False only on
        shutdown, so a caller never mistakes "stopping" for "overflow".
        """
        while not stop.is_set():
            try:
                self._q.put(item, timeout=timeout)
            except queue.Full:
                with self._lock:
                    self.stats["blocked_puts"] += 1
                continue
            with self._lock:
                self.stats["put"] += 1
                self._note_depth()
            return True
        return False

    def get(self, timeout: Optional[float] = 0.1) -> Any:
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        with self._lock:
            self.stats["got"] += 1
        return item

    def qsize(self) -> int:
        return self._q.qsize()

    def join_empty(self, deadline: float) -> bool:
        while self._q.qsize() and time.monotonic() < deadline:
            time.sleep(0.001)
        return self._q.qsize() == 0


class ControlBus:
    """Always-deliverable, so it can never be blocked by a full data queue.

    Unbounded on purpose: control traffic is O(frames), not O(symbols),
    and the failure it guards against is precisely congestion.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: List[Callable[[Any], None]] = []
        self.log: List[Any] = []

    def subscribe(self, fn: Callable[[Any], None]) -> None:
        with self._lock:
            self._subs.append(fn)

    def publish(self, event: Any) -> None:
        with self._lock:
            self.log.append(event)
            subs = list(self._subs)
        for fn in subs:
            fn(event)


@dataclass
class _Worker:
    name: str
    thread: Optional[threading.Thread] = None
    stop: threading.Event = field(default_factory=threading.Event)
    busy_s: float = 0.0
    items: int = 0


class RxFlowgraph:
    """TD | FD | BIT as three threads, bounded queues between them."""

    def __init__(self, env: PhyEnv, *, soft: Optional[SoftConfig] = None,
                 expected_fec: tuple = ("rs_m8", "conv_v27"),
                 chunk_symbols: int = 1, queue_depth: int = 64) -> None:
        self.env = env
        self.chunk_symbols = chunk_symbols

        self.td = RxTimeDomain(env)
        self.stream = TdStream(self.td)
        self.fd = RxFreqDomain(env, soft or SoftConfig())
        self.fd.configure_expected_fec(*expected_fec)
        self.bd = RxBitDomain(env)

        self.q_iq = CountingQueue("iq", queue_depth)
        self.q_i1 = CountingQueue("i1", queue_depth)
        self.q_i2 = CountingQueue("i2", queue_depth)
        self.control = ControlBus()
        self.control.subscribe(self._on_control)

        self._out_lock = threading.Lock()
        self._out: Dict[int, Dict[str, Any]] = {}
        self._done: set = set()          # frame_ids BIT has finished with
        self._td_state: Dict[int, Dict[str, Any]] = {}

        self.workers = {n: _Worker(n) for n in ("td", "fd", "bit")}
        self.stats = {"completed": 0, "header_failed": 0, "payload_failed": 0,
                      "aborted": 0, "gaps": 0, "overflow_aborts": 0}

    # ---- control -----------------------------------------------------

    def _on_control(self, ev: Any) -> None:
        if isinstance(ev, FrameAbort):
            # Both stages must keep the id invalidated, not merely drop
            # what they hold: this event can arrive before some of that
            # frame's batches.
            self.fd.abort(ev.frame_id)
            self.bd.abort(ev.frame_id)
            self.stats["aborted"] += 1
        elif isinstance(ev, FrameDone):
            with self._out_lock:
                self._done.add(ev.frame_id)
        elif isinstance(ev, StreamGap):
            self.stats["gaps"] += 1

    # ---- lifecycle ---------------------------------------------------

    def start(self) -> "RxFlowgraph":
        for name, fn in (("td", self._run_td), ("fd", self._run_fd),
                         ("bit", self._run_bit)):
            w = self.workers[name]
            w.stop.clear()
            w.thread = threading.Thread(target=fn, name=f"rx-{name}", daemon=True)
            w.thread.start()
        return self

    def feed(self, chunk: Any) -> None:
        """Hand samples to TD. Blocks if TD is behind, which is the point
        of a bounded queue -- the alternative is an unbounded backlog."""
        self.q_iq.put_blocking(chunk)

    def drain(self, timeout: float = 30.0, *, settle_polls: int = 4,
              poll_s: float = 0.02) -> List[Dict[str, Any]]:
        """Wait for QUIESCENCE, then stop and return results.

        Testing each queue for "empty" in turn is NOT a quiescence test,
        and getting that wrong was an intermittent frame loss: TD refills
        q_i1 after it has been declared empty, `stop()` then fires with
        work still in flight, and frames in progress are lost. So this
        waits until the queues are empty AND no stage has made progress
        for `settle_polls` consecutive polls.

        Results come back sorted by `frame_id`, never in completion
        order -- completion order is genuinely timing-dependent, so a
        caller who observed it could not be deterministic even though the
        pipeline is.
        """
        deadline = time.monotonic() + timeout
        quiet = 0
        last = None
        while time.monotonic() < deadline:
            empty = all(q.qsize() == 0 for q in (self.q_iq, self.q_i1, self.q_i2))
            now = tuple(w.items for w in self.workers.values())
            # TD may still owe symbols for a frame it has not released.
            td_pending = len(self._td_state)
            if empty and now == last and td_pending == 0:
                quiet += 1
                if quiet >= settle_polls:
                    break
            else:
                quiet = 0
            last = now
            time.sleep(poll_s)
        self.stop()
        with self._out_lock:
            return [self._out[k] for k in sorted(self._out)]

    def stop(self) -> None:
        for w in self.workers.values():
            w.stop.set()
        for q in (self.q_iq, self.q_i1, self.q_i2):
            q.put_or_drop(_SHUTDOWN)
        for w in self.workers.values():
            if w.thread is not None:
                w.thread.join(timeout=5.0)
                w.thread = None

    # ---- workers -----------------------------------------------------

    def _run_td(self) -> None:
        """Detect frames, then PUSH their symbols downstream.

        Pushing is the real difference from the single-threaded driver:
        TD decides, rather than being asked, and so needs its own
        stopping rule -- the cap, optionally shortened by `FrameDone`.
        """
        w = self.workers["td"]
        env = self.env
        # A SHORT idle poll, deliberately. TD owes symbols to frames whose
        # samples have already arrived, and it emits them from _td_pump --
        # so the poll interval is this stage's output latency, not just an
        # idle cost. At the queue's 0.1 s default, the last frame of a
        # burst waited up to 100 ms for a pump, which on a 0.4 s run was
        # most of the measured gap against the serial driver.
        while not w.stop.is_set():
            chunk = self.q_iq.get(timeout=0.002)
            if chunk is None:
                self._td_pump(w)         # keep emitting even when input idles
                continue
            if chunk is _SHUTDOWN:
                break
            t0 = time.perf_counter()
            for fid in self.stream.feed(chunk):
                self._td_state[fid] = {"emitted": 0}
            for ev in self.td.drain_control():
                self.control.publish(ev)
            w.busy_s += time.perf_counter() - t0
            w.items += 1
            self._td_pump(w)

    def _td_pump(self, w: _Worker) -> None:
        """Emit whatever is ready, for every frame TD still owns."""
        env = self.env
        for fid in list(self._td_state):
            st = self._td_state[fid]
            with self._out_lock:
                finished = fid in self._done
            if finished or self.stream.capped(fid, st["emitted"]):
                # Either BIT is done with it (flow control) or TD has hit
                # its own bound. Both release the samples; neither is a
                # config path.
                self.stream.retire(fid)
                self._td_state.pop(fid, None)
                continue
            t0 = time.perf_counter()
            progressed = False
            while True:
                # Clamp to the current region. A batch must never straddle
                # TRAIN / HEADER / BODY: TD is the only stage that can
                # label symbols, FD dispatches on that label, and a mixed
                # batch has no correct label. With chunk_symbols=4 and
                # n_training_symbols = num_symbols_header = 2, the first
                # batch covered symbols 0..3 -- training AND header -- was
                # labelled BODY, and FD was handed body data before any
                # header existed: 3 of 3 frames lost to header_failed.
                n = self._clamp_to_region(st["emitted"], self.chunk_symbols)
                want = st["emitted"] + n
                if not self.stream.ready(fid, want):
                    break
                stype = self._stype_for(st["emitted"], n)
                try:
                    batch = self.stream.symbols(fid, st["emitted"], n, stype)
                except (ValueError, KeyError):
                    self.control.publish(FrameAbort(frame_id=fid, reason="td extract"))
                    self._td_state.pop(fid, None)
                    break
                if not self.q_i1.put_backpressure(batch, w.stop):
                    # Refused only on shutdown -- but the frame is now
                    # incomplete, and FD would still emit `last` later,
                    # letting BIT decode a frame with a HOLE in it. That
                    # was a real, intermittent RS failure. Abort instead.
                    self.control.publish(FrameAbort(frame_id=fid,
                                                    reason="shutdown mid-frame"))
                    self._td_state.pop(fid, None)
                    break
                st["emitted"] += n
                progressed = True
            if progressed:
                w.busy_s += time.perf_counter() - t0

    def _region_end(self, offset: int) -> int:
        """Where the region containing `offset` ends (exclusive).

        Three regions, in the order they arrive: training, header, then
        body -- which is all TD can distinguish, and exactly the split
        `rx_if.vh` gives it. Body has no end TD knows about, so it
        reports None.
        """
        env = self.env
        if offset < env.n_training_symbols:
            return env.n_training_symbols
        hdr_end = env.n_training_symbols + env.num_symbols_header
        if offset < hdr_end:
            return hdr_end
        return -1                      # body: unbounded as far as TD knows

    def _clamp_to_region(self, offset: int, n: int) -> int:
        """Shorten a batch so it stays inside one region."""
        end = self._region_end(offset)
        return n if end < 0 else min(n, end - offset)

    def _stype_for(self, offset: int, n: int) -> SymbolType:
        """Label a batch that `_clamp_to_region` has kept inside one
        region, so the label follows from `offset` alone."""
        env = self.env
        if offset < env.n_training_symbols:
            return SymbolType.TRAIN
        if offset < env.n_training_symbols + env.num_symbols_header:
            return SymbolType.HEADER
        return SymbolType.BODY

    def _run_fd(self) -> None:
        w = self.workers["fd"]
        env = self.env
        pending: Dict[int, List[FftBatch]] = {}
        while not w.stop.is_set():
            batch = self.q_i1.get()
            if batch is None:
                continue
            if batch is _SHUTDOWN:
                break
            t0 = time.perf_counter()
            try:
                self._fd_handle(batch, pending)
            except (ValueError, NotImplementedError, KeyError, RuntimeError) as exc:
                kind = "header_failed" if not self.fd._frames.get(batch.frame_id) \
                    else "payload_failed"
                self.stats[kind] += 1
                self.control.publish(FrameAbort(frame_id=batch.frame_id,
                                                reason=f"fd: {exc}"))
                pending.pop(batch.frame_id, None)
            w.busy_s += time.perf_counter() - t0
            w.items += 1

    def _fd_handle(self, batch: FftBatch, pending: Dict[int, List[FftBatch]]) -> None:
        """Route one batch by its symbol type.

        TRAIN and HEADER runs are accumulated until complete, because
        both are averaged/decoded over their whole run and a partial run
        is meaningless. BODY goes straight through.
        """
        env = self.env
        if batch.stype == SymbolType.TRAIN:
            self.fd.training(batch)
            return
        if batch.stype == SymbolType.HEADER:
            acc = pending.setdefault(batch.frame_id, [])
            acc.append(batch)
            have = sum(b.n_sym for b in acc)
            if have < env.num_symbols_header:
                return
            merged = acc[0] if len(acc) == 1 else FftBatch(
                frame_id=batch.frame_id, symbol_offset=acc[0].symbol_offset,
                sample_offset=acc[0].sample_offset,
                bins=self.env.xp.concatenate([b.bins for b in acc], axis=0),
                stype=SymbolType.HEADER, frame_start=False)
            pending.pop(batch.frame_id, None)
            self.fd.decode_header(merged)
            return
        for llr in self.fd.body(batch):
            if not self.q_i2.put_backpressure(llr, self.workers["fd"].stop):
                # Same hazard on this side: dropping one LlrBatch and
                # later delivering the frame's `last` would hand BIT a
                # frame missing a symbol, which decodes to garbage rather
                # than failing cleanly.
                self.control.publish(FrameAbort(frame_id=batch.frame_id,
                                                reason="shutdown mid-frame"))
                return

    def _run_bit(self) -> None:
        w = self.workers["bit"]
        while not w.stop.is_set():
            llr = self.q_i2.get()
            if llr is None:
                continue
            if llr is _SHUTDOWN:
                break
            t0 = time.perf_counter()
            try:
                result = self.bd.push(llr)
            except (ValueError, NotImplementedError) as exc:
                self.stats["payload_failed"] += 1
                self.control.publish(FrameAbort(frame_id=llr.frame_id,
                                                reason=f"fec: {exc}"))
                self.control.publish(FrameDone(frame_id=llr.frame_id, ok=False))
                result = None
            if result is not None:
                result["frame_id"] = llr.frame_id
                try:
                    result["evm"] = self.fd.evm(llr.frame_id)
                except KeyError:
                    result["evm"] = None
                with self._out_lock:
                    self._out[llr.frame_id] = result
                self.stats["completed"] += 1
                self.fd.retire(llr.frame_id)
                # Flow control back to TD: these samples are spent.
                self.control.publish(FrameDone(frame_id=llr.frame_id, ok=True))
            w.busy_s += time.perf_counter() - t0
            w.items += 1

    # ---- reporting ---------------------------------------------------

    def report(self) -> Dict[str, Any]:
        """Per-stage occupancy and per-queue pressure.

        `busy_s` is wall time inside the stage, so comparing the three
        says which one is the bottleneck -- the pipeline runs no faster
        than its slowest stage. High-water against `maxsize` says whether
        a queue was ever the constraint.
        """
        return {
            "stages": {n: {"busy_s": round(w.busy_s, 6), "items": w.items}
                       for n, w in self.workers.items()},
            "queues": {q.name: dict(q.stats, maxsize=q.maxsize)
                       for q in (self.q_iq, self.q_i1, self.q_i2)},
            "pipeline": dict(self.stats),
            "td": dict(self.td.stats),
            "stream": dict(self.stream.stats),
            "fd": dict(self.fd.stats),
            "bit": dict(self.bd.stats),
        }
