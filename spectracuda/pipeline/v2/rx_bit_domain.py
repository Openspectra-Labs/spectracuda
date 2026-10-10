"""BIT -- the bit domain stage: inner de-interleave, FEC decode, CRC.

WHERE INCREMENTAL STOPS, AND WHY THAT IS NOT A SHORTCOMING. BIT accepts
`LlrBatch`es as they arrive, so FD never stalls waiting for it. But it
cannot DECODE incrementally, for two hard reasons:

1. The outer block interleaver is sized by the whole frame's bit count.
   `framing/packetizer.py` says so directly: "n_bits varies per call ...
   interleaver instances are built lazily and cached by n_bits". A
   frame-wide permutation cannot be inverted from a partial frame.
2. `Ofdm` calls `decode_soft` ONCE per frame. Chunking the Viterbi shifts
   results at chunk boundaries through the sliding-window traceback, so a
   chunked decode would silently stop being bit-exact against the oracle.

So this stage is incremental TRANSPORT and frame-granular DECODE. That is
a constraint to respect rather than optimize away: anyone who "fixes" it
by chunking the Viterbi changes decoded output without changing any test
that only looks at clean channels.

The pipeline still parallelizes, because the skew is across frames:

    TD   frame 102   FFT
    FD   frame 101   equalize + demap
    BIT  frame 100   whole-frame FEC
    MAC  frame  99   packets

WHY BIT NEEDS NOTHING FROM FD BEYOND THE STREAM. Everything it would
otherwise have to ask for -- payload length, encoded bit count, the
crc/fec0/fec1 codes, the interleaver geometry -- arrives on the frame's
first `LlrBatch` as a `HeaderConfig`, because FD decoded the header
itself. That is what makes the dataflow strictly forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ...block import Block
from ...framing.packetizer import Packetizer
from ...registry import resolve
from .env import PhyEnv
from .stage_if import HeaderConfig, LlrBatch


@dataclass
class _Accum:
    """One frame's coded bits, accumulated until it is complete."""

    frame_id: int
    header: Optional[HeaderConfig] = None
    hard: List[Any] = field(default_factory=list)
    soft: List[Any] = field(default_factory=list)
    n_symbols: int = 0


class RxBitDomain(Block):
    """`LlrBatch` in, decoded payload bytes out."""

    batch_shape_doc = (
        "LlrBatch(hard_bits=(n*bits_per_ofdm_symbol,) uint8, llrs=same or None) in -> "
        "{bits: (1, payload_len_bits) uint8, crc_valid: (1,) bool} out on the frame's last batch"
    )

    def __init__(self, env: PhyEnv) -> None:
        super().__init__(backend=env.backend)
        self.env = env
        # Per-stage codecs, warmed on first use and never shared with FD
        # or with a TX stage -- `ConvolutionalCode` holds a libcorrect
        # handle with internal decoder state and builds `_native_soft`
        # lazily with no lock (`fec/viterbi.py:174,329`).
        self._pkt_cache: Dict[tuple, Any] = {}
        self._il2_cache: Dict[int, Any] = {}
        self._frames: Dict[int, _Accum] = {}
        self.invalidated: set = set()
        self.finished: set = set()
        self._finished_cap = 256
        self.stats = {"frames_decoded": 0, "decode_failures": 0,
                      "dropped_invalid": 0, "partial_frames_discarded": 0,
                      "dropped_surplus": 0}

    # ---- lifecycle ---------------------------------------------------

    def abort(self, frame_id: int) -> None:
        """Honour a `FrameAbort`, and keep honouring it.

        The id stays in `invalidated` rather than this just dropping what
        is held, because control overtakes data: the abort arrives before
        some of that frame's chunks, and those have to be discarded on
        arrival. Retiring the id early is exactly how a partial frame
        reaches MAC.
        """
        if self._frames.pop(frame_id, None) is not None:
            self.stats["partial_frames_discarded"] += 1
        self.invalidated.add(frame_id)

    def retire(self, frame_id: int) -> None:
        """Mark the frame finished; see `RxFreqDomain.retire` for why
        finished and forgotten must differ."""
        self._frames.pop(frame_id, None)
        self.invalidated.discard(frame_id)
        self.finished.add(frame_id)
        if len(self.finished) > self._finished_cap:
            self.finished = set(sorted(self.finished)[-self._finished_cap:])

    # ---- the stream --------------------------------------------------

    def push(self, batch: LlrBatch) -> Optional[Dict[str, Any]]:
        """Accumulate a batch; decode and return a result on `last`."""
        if batch.frame_id in self.invalidated:
            self.stats["dropped_invalid"] += 1
            return None
        if batch.frame_id in self.finished:
            self.stats["dropped_surplus"] += 1
            return None

        acc = self._frames.setdefault(batch.frame_id, _Accum(batch.frame_id))
        if batch.header is not None:
            acc.header = batch.header
        acc.hard.append(batch.hard_bits)
        if batch.llrs is not None:
            acc.soft.append(batch.llrs)
        acc.n_symbols += 1

        if not batch.last:
            return None
        try:
            return self._decode(acc)
        finally:
            self.retire(batch.frame_id)

    def _decode(self, acc: _Accum) -> Dict[str, Any]:
        """Un-permute, truncate, FEC-decode, CRC-check."""
        env, xp = self.env, self.env.xp
        cfg = acc.header
        if cfg is None:
            raise RuntimeError(f"frame {acc.frame_id}: complete but no header ever arrived")

        hard = xp.concatenate([xp.asarray(h).reshape(-1) for h in acc.hard])[None, :]
        # From the header, NOT inferred from how many batches arrived:
        # FD chunks differently in frame-scaling mode (one batch for the
        # whole frame), and an inferred block size silently ran the
        # inverse permutation on the wrong geometry.
        block = cfg.bits_per_symbol_payload

        # Un-permute BEFORE truncating. The inner interleaver ran over
        # whole PADDED OFDM symbols on transmit, so its inverse has to see
        # the same whole symbols -- truncating first hands it a partial
        # block, which still "works" and still decodes something.
        encoded = self._il2(hard, block)[:, : cfg.encoded_bit_count]

        soft = None
        if acc.soft:
            soft_cat = xp.concatenate([xp.asarray(s).reshape(-1) for s in acc.soft])[None, :]
            # The SAME inverse permutation as the hard bits: these are
            # per-coded-bit quantities in the same order, so a soft path
            # that skipped this would hand Viterbi confidences belonging
            # to different bits.
            soft = self._il2(soft_cat, block)[:, : cfg.encoded_bit_count]

        pkt = self._packetizer(cfg)
        try:
            result = pkt.decode(encoded, soft=soft)
        except (ValueError, NotImplementedError):
            self.stats["decode_failures"] += 1
            raise
        self.stats["frames_decoded"] += 1
        return {
            "bits": result["bits"],
            "crc_valid": result["crc_valid"],
            "header": dict(cfg.fields),
        }

    # ---- codecs ------------------------------------------------------

    def _il2(self, flat: Any, block: int) -> Any:
        """Inverse of the inner (frequency) interleaver, per OFDM symbol.

        A no-op when `interleaver2="none"`. Block size is the DECODED
        modem's bits-per-symbol, not this receiver's own
        `bits_per_ofdm_symbol` -- getting that wrong is a real bug this
        project already hit once ("cannot reshape array of size 80 into
        shape (240)"), because the receiver's configured modem need not
        match the frame's.
        """
        env, xp = self.env, self.env.xp
        if env.interleaver2 == "none":
            return flat
        rows = flat.reshape(-1, block)
        if block not in self._il2_cache:
            self._il2_cache[block] = resolve(
                "interleaver", env.interleaver2, n_bits=block,
                backend=env.backend, **env.interleaver2_kwargs)
        out = self._il2_cache[block].decode(rows)
        return xp.asarray(out).reshape(flat.shape[0], -1)

    def _packetizer(self, cfg: HeaderConfig) -> Any:
        """Build (and cache) a decoder for the codes the WIRE announced.

        Resolved from the decoded header, never from this object's own
        configuration -- a real receiver is a separate device that never
        saw the transmitter's constructor call. The interleaver is the one
        exception: it is never signalled over the air, so it comes from
        configuration (`framing/packetizer.py`'s module docstring).
        """
        env = self.env
        key = (cfg.crc, cfg.fec0, cfg.fec1)
        if key not in self._pkt_cache:
            self._pkt_cache[key] = Packetizer(
                crc=cfg.crc, fec=cfg.fec0, fec1=cfg.fec1,
                interleaver=env.interleaver,
                interleaver_kwargs=env.interleaver_kwargs,
                backend=env.backend)
        return self._pkt_cache[key]

    def process(self, batch: Any, **kwargs: Any) -> Any:
        return self.push(batch)
