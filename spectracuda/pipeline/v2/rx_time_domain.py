"""TD -- the time domain stage: ADC quantization, RSSI, preamble sync,
CFO estimate and correction, slot extraction and the FFT.

TD'S CONTRACT, AND THE ONE THING IT DOES NOT KNOW. TD produces
frequency-domain symbols and nothing else. It needs OFDM *timing* only --
not the payload length, not the modulation, not the FEC. All three live in
the header, which FD decodes (see `rx_freq_domain.py`), so TD can never
learn them without a backward path, and V2 has none by design.

The consequence is that TD cannot know where a frame ENDS. It therefore
does not try: it emits symbols as it produces them, and FD stops consuming
a frame once it has had the `body_syms` the header announced. An earlier
draft of this design had TD collect a MAX_PAYLOAD_SYMBOLS window for FD to
trim, which was wrong -- it buys nothing and costs latency and memory
exactly where it hurts most, in continuous streaming.

CANDIDATE VS ACTIVE SYNCHRONIZATION -- the sharp edge in this file. TD
keeps its sliding preamble search running while a frame is in flight,
because back-to-back frames of differing lengths have to be caught. But a
correlation peak found INSIDE frame N's payload may be a false one, and
re-aligning on it would corrupt a frame that was decoding fine. This is
not hypothetical here: `Ofdm.reset_stream`'s own comment describes
`stream_debug_counts` partly as a way to separate real preamble misses
from "the signature of the sync-collision mechanism".

So there is one hard invariant:

    Candidate evaluation is NON-DESTRUCTIVE. A candidate detection's
    timing lives in its own state, never in the active frame's. The
    active frame's synchronization is replaced only by an explicit
    promotion decision, never as a side effect of evaluating a peak.

When a candidate is promoted and the active frame has to be given up, TD
emits an explicit `FrameAbort` so FD and BIT drop their state for that
`frame_id`. No feedback from BIT is involved in any of this.

WHY SAMPLES NEVER LEAVE THIS STAGE. `rx_corrected` is TD-private and
keyed by `frame_id`. FD asks for symbols by (frame_id, offset, count) and
receives `FftBatch`es; it never holds time-domain samples. That is what
makes "TD owns the time domain" a structural fact rather than a comment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ...block import Block
from ...framing.stats import compute_rssi_db
from .env import PhyEnv
from .stage_if import FftBatch, FrameAbort, SymbolType


@dataclass
class FrameSync:
    """TD's per-frame state. Private to TD -- FD only ever holds the id.

    `pos0` is where the first TRAINING symbol's window starts, already
    carrying the `- timing_advance` nudge that places every window
    slightly early inside its CP (see `Ofdm`'s constructor comment on
    `timing_advance`): the preamble has no CP, so this one offset is what
    every later window inherits as it walks forward by `slot_len`.
    """

    frame_id: int
    start_index: Any
    sync_metric: Any
    cfo_estimate: Any
    rx_corrected: Any
    pos0: Any
    rssi_db: Any
    n_emitted: int = 0
    aborted: bool = False


@dataclass
class _Candidate:
    """A preamble peak seen while a frame is already active.

    Held HERE, apart from the active `FrameSync`, which is the whole
    point: evaluating this can never perturb the frame currently
    decoding. Promotion is an explicit act, not a side effect.
    """

    start_index: int
    metric: float
    sample_offset: int


class RxTimeDomain(Block):
    """Samples in, `FftBatch`es out.

    Pull-shaped for now: FD calls `symbols()` for the training and header
    symbols, then -- once it has decoded the length -- for the body. That
    is demand-driven and therefore trivially bounded, and it becomes a
    push into a bounded queue when the flowgraph arrives (plan step 4)
    without the stage's logic changing.
    """

    batch_shape_doc = (
        "(1, n_samples) complex64 in -> FftBatch(bins=(n_sym, fft_size) complex64) out; "
        "one frame per detect() call, symbols pulled by (frame_id, offset, count)"
    )

    def __init__(self, env: PhyEnv) -> None:
        super().__init__(backend=env.backend)
        self.env = env
        self._frames: Dict[int, FrameSync] = {}
        self._next_frame_id = 0
        self._candidate: Optional[_Candidate] = None
        self.pending_control: List[Any] = []
        # Counters, present from the first commit rather than retrofitted:
        # a pipeline's useful diagnostics are per-stage occupancy and
        # drops, and bolting them on later means the first measurements
        # are the ones you cannot explain.
        self.stats = {"frames_started": 0, "frames_aborted": 0, "symbols_emitted": 0,
                      "candidates_seen": 0, "candidates_promoted": 0}

    # ---- helpers -----------------------------------------------------

    def _quantize(self, iq: Any) -> Any:
        """Boundary ADC quantization, byte-for-byte as `Ofdm._quantize`.

        Reproduced rather than imported because it is four lines and
        importing it would mean V2 holding an `Ofdm`; see `env.py`.
        """
        if self.env.iq_dtype == "float32":
            return iq
        xp = self.env.xp
        real_q = xp.real(iq).astype("float16").astype("float32")
        imag_q = xp.imag(iq).astype("float16").astype("float32")
        return (real_q + 1j * imag_q).astype("complex64")

    # ---- detection ---------------------------------------------------

    def detect(self, rx_iq: Any, *, sample_offset: int = 0) -> Dict[str, Any]:
        """Quantize, measure RSSI, run sync, estimate and apply CFO.

        Returns the same four always-present observables `Ofdm.rx_process`
        reports regardless of whether a frame was found -- `frame_found`,
        `start_index`, `sync_metric`, `rssi_db` -- plus `frame_id` when
        one was. RSSI is computed unconditionally because it describes
        received energy, not frame content.
        """
        env = self.env
        xp = env.xp
        rx_iq = xp.asarray(rx_iq)
        if rx_iq.ndim == 1:
            rx_iq = rx_iq[None, :]
        rx_iq = self._quantize(rx_iq)

        rssi_db = compute_rssi_db(xp, rx_iq)
        sync_result = env.sync.process(rx_iq)
        start_index = sync_result["start_index"]
        metric = sync_result["metric"]

        # Gated on item 0's metric only, matching `Ofdm`'s "one call = one
        # frame" convention. `sync.process()` always returns SOME
        # candidate -- it is a best-window search, not a detector with a
        # built-in null hypothesis -- so without this gate pure noise
        # decodes into garbage.
        item0 = float(np.asarray(self._to_host(metric))[0])
        if item0 < env.sync_threshold:
            return {"frame_found": False, "start_index": start_index,
                    "sync_metric": metric, "rssi_db": rssi_db, "frame_id": None}

        cfo_estimate = env.cfo.process(rx_iq, start_index=start_index)
        rx_corrected = env.cfo.correct(rx_iq, cfo_estimate)
        pos0 = start_index + env.fft_size - env.timing_advance

        frame_id = self._next_frame_id
        self._next_frame_id += 1
        self._frames[frame_id] = FrameSync(
            frame_id=frame_id, start_index=start_index, sync_metric=metric,
            cfo_estimate=cfo_estimate, rx_corrected=rx_corrected, pos0=pos0,
            rssi_db=rssi_db,
        )
        self.stats["frames_started"] += 1
        return {"frame_found": True, "start_index": start_index,
                "sync_metric": metric, "rssi_db": rssi_db, "frame_id": frame_id}

    def _to_host(self, arr: Any) -> np.ndarray:
        get = getattr(arr, "get", None)
        return np.asarray(get() if callable(get) else arr)

    # ---- candidate handling -----------------------------------------

    def offer_candidate(self, start_index: int, metric: float, sample_offset: int) -> None:
        """Record a preamble peak seen while a frame is active.

        Recording only. It cannot disturb the active frame, which is the
        invariant this file exists to protect.
        """
        self.stats["candidates_seen"] += 1
        if self._candidate is None or metric > self._candidate.metric:
            self._candidate = _Candidate(start_index=int(start_index),
                                         metric=float(metric),
                                         sample_offset=int(sample_offset))

    def promote_candidate(self, active_frame_id: Optional[int], reason: str = "") -> Optional[_Candidate]:
        """Explicitly give up the active frame in favour of the candidate.

        The only path by which an active frame's synchronization is ever
        replaced. Emits a `FrameAbort` for the abandoned frame so FD and
        BIT can drop its state -- they may still be holding chunks for it,
        and because control overtakes data they must keep the id
        invalidated rather than merely discarding what they hold.
        """
        cand = self._candidate
        if cand is None:
            return None
        if active_frame_id is not None:
            self.abort(active_frame_id, reason=reason or "candidate promoted")
        self._candidate = None
        self.stats["candidates_promoted"] += 1
        return cand

    def abort(self, frame_id: int, reason: str = "") -> None:
        """Drop TD's state for a frame and publish the abort."""
        fs = self._frames.pop(frame_id, None)
        if fs is not None:
            fs.aborted = True
        self.stats["frames_aborted"] += 1
        self.pending_control.append(FrameAbort(frame_id=frame_id, reason=reason))

    def drain_control(self) -> List[Any]:
        out, self.pending_control = self.pending_control, []
        return out

    # ---- symbol production -------------------------------------------

    def symbols(self, frame_id: int, symbol_offset: int, n_sym: int,
                stype: SymbolType = SymbolType.BODY) -> FftBatch:
        """Extract `n_sym` consecutive slots and FFT them.

        BIT-EXACTNESS NOTE. The FFT runs here for EVERY symbol, DMRS
        included, where `Ofdm` instead FFTs DMRS slots inside
        `_estimate_channel_from_dmrs`. The bins are identical because it
        is the same `demod.process` on the same samples -- that method's
        docstring is explicit that a DMRS is the training symbol
        re-transmitted and goes through the same demod path -- so moving
        the transform earlier changes scheduling, not arithmetic.

        Gathered with one fancy-index call rather than `n_sym` separate
        slices, matching `Ofdm`'s own reason for doing it that way: the
        spacing between symbols is a constant `slot_len`, so every slot
        start is computable by broadcasting with no data dependency.
        """
        env = self.env
        xp = env.xp
        fs = self._frames.get(frame_id)
        if fs is None:
            raise KeyError(f"frame_id={frame_id} is not active in TD "
                           f"(aborted, retired, or never detected)")

        starts = fs.pos0[:, None] + (symbol_offset + xp.arange(n_sym))[None, :] * env.slot_len
        idx = starts[:, :, None] + xp.arange(env.slot_len)[None, None, :]
        # Explicit bounds check with ValueError, not IndexError: fancy
        # indexing raises IndexError on a truncated buffer, which
        # `mac/session.py`'s `except (ValueError, NotImplementedError)`
        # does NOT catch -- so a corrupted frame would stop being a
        # handled failure and start being a crash. `Ofdm` guards the same
        # way for the same reason.
        if idx.size and int(idx.max()) >= fs.rx_corrected.shape[1]:
            raise ValueError(
                f"frame {frame_id}: symbols {symbol_offset}..{symbol_offset + n_sym - 1} "
                f"need samples up to index {int(idx.max())} but only "
                f"{fs.rx_corrected.shape[1]} are present -- truncated frame"
            )
        batch_idx = xp.arange(fs.rx_corrected.shape[0])[:, None, None]
        slots = fs.rx_corrected[batch_idx, idx]                 # (1, n_sym, slot_len)
        flat = slots.reshape(slots.shape[0] * n_sym, env.slot_len)
        bins = env.demod.process(flat)                          # (n_sym, fft_size)

        fs.n_emitted += n_sym
        self.stats["symbols_emitted"] += n_sym
        return FftBatch(
            frame_id=frame_id,
            symbol_offset=symbol_offset,
            sample_offset=int(self._to_host(starts)[0, 0]),
            bins=bins,
            stype=stype,
            frame_start=(symbol_offset == 0),
        )

    def frame(self, frame_id: int) -> FrameSync:
        return self._frames[frame_id]

    def retire(self, frame_id: int) -> None:
        """Release a completed frame's samples.

        Explicit rather than refcounted so the memory high-water is a
        property of the pipeline's depth and not of the garbage
        collector's mood.
        """
        self._frames.pop(frame_id, None)

    def process(self, batch: Any, **kwargs: Any) -> Any:
        """`Block`'s entry point -- `detect()` under its required name."""
        return self.detect(batch, **kwargs)
