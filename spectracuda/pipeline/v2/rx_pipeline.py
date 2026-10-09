"""`RxPipeline` -- the single-threaded driver for TD -> FD -> BIT.

WHAT THIS IS FOR. The three stages are independent and know nothing about
each other; something has to move batches between them. Here that
something is a plain loop, which is deliberate: a plain loop makes the
partition provable. Any mismatch against `Ofdm` is a partition bug,
because there is no concurrency to blame it on.

Step 4 replaces this loop with bounded queues and one thread per stage.
The stages do not change when that happens -- only this file does. That
is the whole reason it exists as a separate object rather than as test
scaffolding.

PER-FRAME STATE LIVES HERE, NOT ON THE STAGES. A frame in flight has a
small amount of sequencing state (has its header been decoded? how many
body slots have been handed over?). That belongs to whoever is
sequencing, which is this object -- the stages keep only what their own
algorithm needs. `Ofdm` mixes the two on `self`, which is precisely what
makes it unthreadable.

WHAT IS AND IS NOT A BACKWARD PATH. This driver calls
`TdStream.retire()` when a frame finishes, which is flow control --
"those samples are no longer needed". It carries no decode information
and is strictly optional, because `TdStream.capped()` already bounds a
frame's emission. The thing V2 refuses is a backward CONFIG path, and
there is none: FD decodes the header itself and forwards everything BIT
needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .env import PhyEnv
from .rx_bit_domain import RxBitDomain
from .rx_freq_domain import RxFreqDomain
from .rx_time_domain import RxTimeDomain, TdStream
from .stage_if import FrameAbort, SoftConfig, StreamGap, SymbolType


@dataclass
class _InFlight:
    """Sequencing state for one frame. Owned by the driver."""

    frame_id: int
    header_done: bool = False
    cfg: Any = None
    slots_sent: int = 0
    failed: Optional[str] = None


class RxPipeline:
    """Feed IQ in, get decoded frames out.

    `feed()` returns the frames that COMPLETED during that call, so a
    caller can drive it with arbitrary chunk sizes and does not have to
    know anything about symbol or frame boundaries.
    """

    def __init__(self, env: PhyEnv, *, soft: Optional[SoftConfig] = None,
                 expected_fec: tuple = ("rs_m8", "conv_v27"),
                 chunk_symbols: int = 1) -> None:
        self.env = env
        self.td = RxTimeDomain(env)
        self.stream = TdStream(self.td)
        self.fd = RxFreqDomain(env, soft or SoftConfig())
        self.fd.configure_expected_fec(*expected_fec)
        self.bd = RxBitDomain(env)
        self.chunk_symbols = chunk_symbols
        self._inflight: Dict[int, _InFlight] = {}
        self.control: List[Any] = []
        self.stats = {"completed": 0, "header_failed": 0, "payload_failed": 0,
                      "aborted": 0, "gaps": 0}

    # ---- control -----------------------------------------------------

    def _publish(self, events: List[Any]) -> None:
        """Fan control events out to every stage that holds frame state.

        Both FD and BIT must see an abort, and both keep the id
        invalidated afterwards rather than merely dropping what they
        hold: control travels on its own channel so a full data queue
        cannot block it, which means it arrives BEFORE some of that
        frame's chunks.
        """
        for ev in events:
            self.control.append(ev)
            if isinstance(ev, FrameAbort):
                self.fd.abort(ev.frame_id)
                self.bd.abort(ev.frame_id)
                self._inflight.pop(ev.frame_id, None)
                self.stats["aborted"] += 1
            elif isinstance(ev, StreamGap):
                self.stats["gaps"] += 1

    def gap(self, n_samples: int = 0) -> None:
        """Declare the input discontinuous; see `TdStream.gap`."""
        self._publish(self.stream.gap(n_samples))

    def reset(self) -> None:
        self.stream.reset()
        self._inflight.clear()

    # ---- the loop ----------------------------------------------------

    def feed(self, chunk: Any) -> List[Dict[str, Any]]:
        """Push samples through as far as they will go."""
        for fid in self.stream.feed(chunk):
            self._inflight[fid] = _InFlight(frame_id=fid)
        self._publish(self.td.drain_control())
        done: List[Dict[str, Any]] = []
        # list() because a frame may retire itself mid-iteration.
        for fid in list(self._inflight):
            r = self._advance(fid)
            if r is not None:
                done.append(r)
        return done

    def _advance(self, fid: int) -> Optional[Dict[str, Any]]:
        env = self.env
        f = self._inflight.get(fid)
        if f is None:
            return None

        if not f.header_done:
            need = env.n_training_symbols + env.num_symbols_header
            if not self.stream.ready(fid, need):
                return None
            try:
                self.fd.training(self.stream.symbols(
                    fid, 0, env.n_training_symbols, SymbolType.TRAIN))
                f.cfg = self.fd.decode_header(self.stream.symbols(
                    fid, env.n_training_symbols, env.num_symbols_header,
                    SymbolType.HEADER))
            except (ValueError, NotImplementedError, KeyError) as exc:
                # A header that will not decode is the single most common
                # real outcome of a false sync detection, so it must cost
                # one frame and nothing more: drop it and keep streaming.
                # `Ofdm`'s own stream_debug_counts exists to tell this
                # case apart from "never triggered" and "payload failed".
                f.failed = f"header: {exc}"
                self.stats["header_failed"] += 1
                self._finish(fid)
                return None
            f.header_done = True

        cfg = f.cfg
        while f.slots_sent < cfg.n_total_slots:
            n = min(self.chunk_symbols, cfg.n_total_slots - f.slots_sent)
            base = env.n_training_symbols + env.num_symbols_header + f.slots_sent
            if not self.stream.ready(fid, base + n):
                return None
            try:
                batches = self.fd.body(self.stream.symbols(fid, base, n))
            except (ValueError, KeyError) as exc:
                f.failed = f"body: {exc}"
                self.stats["payload_failed"] += 1
                self._finish(fid)
                return None
            f.slots_sent += n
            for llr in batches:
                try:
                    result = self.bd.push(llr)
                except (ValueError, NotImplementedError) as exc:
                    # An uncorrectable codeword is a delivery failure, not
                    # a pipeline error.
                    f.failed = f"fec: {exc}"
                    self.stats["payload_failed"] += 1
                    self._finish(fid)
                    return None
                if result is not None:
                    result["frame_id"] = fid
                    result["evm"] = self.fd.evm(fid)
                    self.stats["completed"] += 1
                    self._finish(fid)
                    return result
        return None

    def _finish(self, fid: int) -> None:
        """Release a frame everywhere it is held."""
        self.fd.retire(fid)
        self.bd.retire(fid)
        self.stream.retire(fid)
        self._inflight.pop(fid, None)
