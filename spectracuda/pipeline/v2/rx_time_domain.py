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
from .stage_if import FftBatch, FrameAbort, StreamGap, SymbolType


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
    # Streaming only. `abs_start` is the preamble's ABSOLUTE index, which
    # is the fixed reference every later correction is measured from --
    # `cfo.correct()` de-rotates from index 0 of whatever array it is
    # given, so correcting a span that began somewhere else would apply
    # the wrong phase. Always correcting from the preamble keeps the
    # phase consistent across the frame; the resulting constant offset
    # relative to a one-shot correction from the buffer's own origin is a
    # COMMON phase and is absorbed by h_hat.
    abs_start: Optional[int] = None
    corrected_len: int = 0


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

    def begin_stream_frame(self, buffer: Any, rel: int, metric: float,
                           *, base_offset: int) -> int:
        """Open a frame from a streaming detection.

        Only the preamble (plus the trailing-edge guard's L samples) is
        buffered at this point, so the CFO is estimated now -- from the
        preamble, which is all the estimator needs -- and APPLIED later,
        as the rest of the frame arrives. That split is what makes
        streaming different from `detect()`, where everything is present
        at once.
        """
        env, xp = self.env, self.env.xp
        start_arr = xp.asarray([rel])
        cfo_estimate = env.cfo.process(buffer, start_index=start_arr)
        frame_id = self._next_frame_id
        self._next_frame_id += 1
        self._frames[frame_id] = FrameSync(
            frame_id=frame_id, start_index=start_arr,
            sync_metric=xp.asarray([metric]), cfo_estimate=cfo_estimate,
            rx_corrected=None,
            # Relative to the frame's OWN corrected array, which begins at
            # the preamble -- hence `fft_size` (skip the preamble) less the
            # timing advance, and no `start_index` term.
            pos0=xp.asarray([env.fft_size - env.timing_advance]),
            rssi_db=compute_rssi_db(xp, buffer),
            abs_start=base_offset + rel,
        )
        self.stats["frames_started"] += 1
        return frame_id

    def extend_stream_frame(self, frame_id: int, buffer: Any, base_offset: int) -> None:
        """Re-correct the frame's span now that more samples have arrived.

        Corrected from `abs_start` every time, for the phase reason given
        on `FrameSync.abs_start`. Only grows, and only when it has to.
        """
        env = self.env
        fs = self._frames[frame_id]
        begin = fs.abs_start - base_offset
        if begin < 0:
            raise ValueError(
                f"frame {frame_id}: its preamble at {fs.abs_start} has been trimmed "
                f"from the buffer (origin now {base_offset}) -- the frame outlived "
                f"the samples it needs"
            )
        span = buffer[:, begin:]
        if int(span.shape[-1]) <= fs.corrected_len:
            return
        fs.rx_corrected = env.cfo.correct(span, fs.cfo_estimate)
        fs.corrected_len = int(span.shape[-1])

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


# ======================================================================
# Streaming
# ======================================================================
#
# `detect()` above is "one buffer, one frame" -- enough to prove the
# partition, not enough to run a radio. `TdStream` is the continuous-feed
# form, and it is where TD's real complexity lives.
#
# TWO BUGS INHERITED AS REQUIREMENTS, both measured in V1 rather than
# reasoned about, and both easy to reintroduce:
#
# 1. BOUND THE BUFFER AS HISTORY + CHUNK, never as a fixed total. V1
#    originally capped with `buffer[-cap:]` AFTER concatenating, which
#    discarded the entire previous buffer whenever one chunk was as long
#    as the cap -- and 2048 samples is exactly what the Pluto example
#    scripts feed. A preamble that had started in the previous chunk's
#    tail lost its head and was never found: ~9% of all alignments
#    silently lost on a CLEAN channel.
#
# 2. TRAILING-EDGE GUARD. A partially-arrived preamble already scores
#    (2(k-L)/k)^2 at the last candidate offset -- 0.44 with three
#    quarters present, 0.73 with seven eighths, both far above the 0.3
#    default threshold -- with a `start_index` that is (fft_size - k)
#    samples EARLY. Early by more than the CP means ISI and a failed
#    decode: up to ~30% of alignments lost at chunk=64. The fix is to
#    refuse a detection until L = fft_size/2 further samples are
#    buffered, by which point the argmax is the true start. L is the
#    exact bound and is independent of the threshold, because with <= L
#    preamble samples present the two halves do not overlap at all.
#
# HOW A FRAME ENDS, WITHOUT ANYTHING FLOWING BACKWARD. TD cannot know a
# frame's length. So it does not decide: each active frame emits until it
# hits `max_payload_symbols` (a bound, so termination never depends on
# another stage) or until the orchestrator retires it. Retirement is
# FLOW CONTROL -- "these samples are no longer needed" -- and carries no
# decode information, so it is not the backward config path V2 rejects;
# it is also strictly optional, because the cap alone guarantees the
# frame stops.
#
# Several frames may therefore be active at once, which is what makes
# back-to-back frames of DIFFERING lengths work: a new preamble starts a
# new frame immediately instead of waiting for the previous one to be
# declared finished by a stage downstream.


class TdStream:
    """Continuous-feed front end around `RxTimeDomain`.

    Owns the sample buffer and the SEEKING/ACTIVE decision. This is the
    state `Ofdm` keeps as `_stream_buffer` / `_stream_state` /
    `_stream_frame_start` on `self`; here it belongs to TD, which is what
    lets a second stage run on another thread without racing it.
    """

    def __init__(self, td: RxTimeDomain, *, search_window_symbols: int = 8) -> None:
        self.td = td
        self.env = td.env
        self.search_window_symbols = search_window_symbols
        self.reset()

    def reset(self) -> None:
        """Abandon whatever is in flight and start over."""
        xp = self.env.xp
        self.buffer = xp.zeros((1, 0), dtype="complex64")
        self.base_offset = 0          # absolute index of buffer[0]
        self.active: List[int] = []   # frame_ids still emitting, oldest first
        self._starts: Dict[int, int] = {}   # frame_id -> absolute preamble start
        # Absolute starts already detected, INCLUDING retired frames.
        # `_starts` alone is not enough: it loses a frame when the frame
        # retires, but that frame's samples are still in the buffer, so
        # the same preamble gets detected and decoded a SECOND time.
        # Measured: three transmitted frames came out as four decodes,
        # the last one a byte-identical duplicate. Pruned in `_trim` once
        # the samples are gone, since a trimmed preamble can never be
        # re-detected.
        self._seen: set = set()
        self._min_extent = (self.env.n_training_symbols + self.env.num_symbols_header)
        self.stats = {"chunks": 0, "detections": 0, "suppressed_in_frame": 0,
                      "gaps": 0, "capped": 0}

    # ---- input -------------------------------------------------------

    def gap(self, n_samples: int = 0) -> List[Any]:
        """Declare the input discontinuous.

        Every active frame's symbol alignment is now meaningless -- a gap
        is not one frame's problem, it invalidates the notion of "the next
        slot_len samples are the next symbol" for all of them. So every
        active frame is aborted and the buffer dropped, and the aborts are
        returned for the orchestrator to publish on the control channel.
        """
        xp = self.env.xp
        events: List[Any] = []
        for fid in list(self.active):
            self.td.abort(fid, reason="stream gap")
        events.extend(self.td.drain_control())
        events.append(StreamGap(sample_offset=self.base_offset + int(self.buffer.shape[-1]),
                                reason="declared by caller"))
        self.active.clear()
        self._starts.clear()
        self.base_offset += int(self.buffer.shape[-1]) + int(n_samples)
        self.buffer = xp.zeros((1, 0), dtype="complex64")
        self.stats["gaps"] += 1
        return events

    def feed(self, chunk: Any) -> List[int]:
        """Append samples and return the frame_ids newly detected.

        Detection only. Symbols are produced by `ready()` so the caller
        controls batch size, which is the knob step 5 sweeps.
        """
        xp = self.env.xp
        chunk = xp.asarray(chunk)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        chunk = self.td._quantize(chunk)
        self.buffer = xp.concatenate([self.buffer, chunk], axis=-1)
        self.stats["chunks"] += 1
        self._trim()
        return self._search()

    def _trim(self) -> None:
        """Keep history + the newest chunk, never a fixed total.

        See requirement 1 above: the fixed-total form is what lost ~9% of
        alignments at chunk=2048. Samples still needed by an active frame
        are never trimmed, because TD owns those until the frame is
        retired or capped.
        """
        keep_from = self.base_offset
        if self.active:
            keep_from = min(self._starts[f] for f in self.active)
        history = self.search_window_symbols * self.env.fft_size
        want_from = self.base_offset + int(self.buffer.shape[-1]) - history
        cut_at = min(keep_from, want_from) - self.base_offset
        if cut_at > 0:
            self.buffer = self.buffer[:, cut_at:]
            self.base_offset += int(cut_at)
        self._seen = {s for s in self._seen if s >= self.base_offset}

    def _search(self) -> List[int]:
        """Find every preamble in the buffer, IN TIME ORDER.

        Two things make this harder than one `sync.process()` call:

        1. `sync.process()` returns ONE argmax for the array it is given,
           so a single call finds a single preamble. Fine with small
           chunks, wrong as soon as one chunk carries several frames: at
           8192-sample chunks three transmitted frames came out as two
           decodes, both of the wrong payload.
        2. That argmax is the GLOBAL maximum, which need not be the
           EARLIEST preamble. Accepting it and advancing past it skipped
           every frame before it -- whole-buffer feeds decoded only the
           last frame.

        So the buffer is scanned in overlapping windows instead, and peaks
        are taken in the order they occur. The window is 2*fft_size + L
        with a stride of fft_size, which guarantees any preamble is wholly
        inside at least one window along with the L samples its
        trailing-edge guard needs.
        """
        env = self.env
        found: List[int] = []
        L = env.fft_size // 2
        buf_len = int(self.buffer.shape[-1])
        if buf_len < env.fft_size:
            return found                   # not even one preamble's worth

        win = 2 * env.fft_size + L
        stride = env.fft_size
        min_frame = env.fft_size + self._min_extent * env.slot_len
        cursor = 0
        while cursor < buf_len:
            span = self.buffer[:, cursor:cursor + win]
            if int(span.shape[-1]) < env.fft_size:
                break
            res = env.sync.process(span)
            metric = float(np.asarray(self.td._to_host(res["metric"]))[0])
            rel = int(np.asarray(self.td._to_host(res["start_index"]))[0])
            absolute = self.base_offset + cursor + rel

            accept = metric >= env.sync_threshold
            # Trailing-edge guard, measured against the WHOLE buffer: a
            # partially-arrived preamble scores 0.73 with seven eighths
            # present, with a start up to fft_size too early, and early
            # by more than the CP means ISI and a failed decode.
            if accept and (cursor + rel + env.fft_size + L > buf_len):
                accept = False
            # Already taken, within a tolerance -- the argmax shifts a
            # sample or two between calls because the window differs even
            # though the samples do not. L is safe: two genuine frames
            # cannot start that close.
            if accept and any(abs(absolute - seen) <= L for seen in self._seen):
                accept = False
            if accept:
                # A peak inside an active frame's MINIMUM extent is
                # recorded as a candidate, not promoted: there the frame
                # is certainly still running, so the peak is a false
                # correlation and acting on it would corrupt a frame that
                # is decoding fine. Past that span TD cannot tell, so the
                # peak is taken at face value -- more likely the next
                # frame than a phantom, and a wrong guess costs one frame
                # rather than every frame after it.
                for fid in self.active:
                    floor = self._starts[fid] + env.fft_size
                    if floor <= absolute < floor + self._min_extent * env.slot_len:
                        self.td.offer_candidate(absolute, metric, absolute)
                        self.stats["suppressed_in_frame"] += 1
                        accept = False
                        break

            if accept:
                fid = self.td.begin_stream_frame(
                    self.buffer, absolute - self.base_offset, metric,
                    base_offset=self.base_offset)
                self._starts[fid] = absolute
                self._seen.add(absolute)
                self.active.append(fid)
                self.stats["detections"] += 1
                found.append(fid)
                cursor = (absolute - self.base_offset) + min_frame
            else:
                cursor += stride
        return found

    # ---- output ------------------------------------------------------

    def ready(self, frame_id: int, n_sym: int) -> bool:
        """Are `n_sym` symbols from this frame's start fully buffered?"""
        env = self.env
        need = (self._starts[frame_id] + env.fft_size - env.timing_advance
                + n_sym * env.slot_len)
        return need <= self.base_offset + int(self.buffer.shape[-1])

    def capped(self, frame_id: int, emitted: int) -> bool:
        """Has this frame emitted everything TD is willing to give it?

        The bound that makes termination independent of any other stage.
        """
        return emitted >= (self.env.n_training_symbols + self.env.num_symbols_header
                           + self.env.max_payload_symbols)

    def symbols(self, frame_id: int, symbol_offset: int, n_sym: int,
                stype: SymbolType = SymbolType.BODY) -> FftBatch:
        """Produce symbols for a streaming frame.

        Brings the frame's CFO correction up to date first, because in
        streaming the samples these symbols need may have arrived long
        after the frame was opened.
        """
        self.td.extend_stream_frame(frame_id, self.buffer, self.base_offset)
        batch = self.td.symbols(frame_id, symbol_offset, n_sym, stype)
        # `sample_offset` must be ABSOLUTE on the wire, and `td.symbols`
        # computes it against the frame's own corrected array. Re-base it
        # so a gap or re-sync stays unambiguous downstream.
        return FftBatch(
            frame_id=batch.frame_id, symbol_offset=batch.symbol_offset,
            sample_offset=self._starts[frame_id] + batch.sample_offset,
            bins=batch.bins, stype=batch.stype, frame_start=batch.frame_start,
        )

    def retire(self, frame_id: int) -> None:
        """Flow control, not configuration: release a frame's samples.

        Carries no decode information, so it is not the backward path V2
        rejects, and it is optional -- `capped()` already guarantees a
        frame stops.
        """
        self.td.retire(frame_id)
        self._starts.pop(frame_id, None)
        if frame_id in self.active:
            self.active.remove(frame_id)
        self._trim()
