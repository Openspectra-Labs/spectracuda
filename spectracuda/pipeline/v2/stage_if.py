"""Stage interfaces for `OfdmV2` -- the FIFO contracts between the time,
frequency and bit domains.

WHY V2 EXISTS. `pipeline/ofdm.py`'s `Ofdm` runs sync -> CFO -> FFT ->
equalize -> demap -> Viterbi -> RS -> CRC on ONE thread, so one core
carries the whole chain. There is no seam to put a thread boundary on:
its two big methods (`_decode_header_from_sync`,
`_decode_payload_from_header`) are straight-line and cut ACROSS the three
domains rather than along them, and per-frame state lives on `self` where
a second thread would race it. V2 re-partitions the same algorithms into
blocks separated by bounded FIFOs, one thread per block, so the load
spreads over cores.

`Ofdm` IS NOT TOUCHED. It stays the shipping path and is V2's
bit-exactness oracle. V2 runs the same transforms it does -- sync, CFO,
FFT, channel estimation, equalization, demapping, Viterbi, RS, CRC -- so
what changes is WHEN they are called and WHO owns the state between
calls, never the arithmetic. Each stage constructs its OWN instances of
those blocks (see `env.py`); only immutable geometry is shared.

THE DATAFLOW IS STRICTLY FORWARD:

    TD --FftBatch--> FD --LlrBatch--> BIT --> MAC

There is deliberately NO backward configuration path. The RTL does it the
other way: `fpga/rtl/rx/rx_bit_domain.v:86` decodes the header and drives
`cfg_mod`/`cfg_dmrs_period` BACKWARD into `rx_freq_domain.v:100`, with FD
buffering body symbols in B1 until that config arrives. V2 instead puts
the header decoder inside FD, so FD is self-sufficient. Adaptive MCS and
DMRS both stay fully per-frame; what changes is only where the header is
read. The consequence to remember: V2's FD/BD boundary NO LONGER matches
the RTL's, so V2 stage dumps cannot be used as RTL stage fixtures. The
forward boundaries still carry the same content as the RTL's I1 (FFT
bins) and I2 (LLR groups).

WHAT IS NOT MIRRORED FROM THE RTL. valid/ready backpressure, the 2-bit
`fseq` wrap rule and B1 sizing are fabric timing concerns with no Python
analogue. V2 stays batched per `spectracuda/block.py`'s batch-shape
contract (arrays keep a leading axis of 1 so `Ofdm`'s blocks run
unchanged). The mirror is of content and ownership, not of timing.

BUFFER OWNERSHIP, AND WHY IT IS A RULE AND NOT A STYLE NOTE. Once a stage
publishes a batch it must not mutate or reuse that numpy backing memory.
Per-stage Python state is NOT sufficient for thread safety if the arrays
themselves are shared and recycled -- the consumer may still be reading.
Published arrays are immutable and owned by the consumer. Any buffer-pool
scheme needs explicit handback and is deferred until measurements justify
it.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class SymbolType(enum.IntEnum):
    """One enum for the whole receiver, named after the RTL's
    `fpga/rtl/rx/rx_if.vh` so the two cannot drift in vocabulary even
    though V2 draws the FD/BD line in a different place.

    Ownership of the classification is split exactly as the RTL splits
    it: TD assigns TRAIN / HEADER / BODY, because that is all it can know
    from timing alone. FD refines BODY into DATA / DMRS / C2, because
    that needs `dmrs_period` from the header. DMRS NEVER LEAVES FD -- it
    is consumed there to refresh H[k] and is not payload.
    """

    TRAIN = 0
    HEADER = 1
    BODY = 2     # TD only: not yet classified
    DATA = 3
    DMRS = 4     # never crosses FD -> BIT
    C2 = 5


@dataclass(frozen=True)
class FftBatch:
    """I1: TD -> FD. One or more consecutive OFDM symbols' FFT bins.

    `bins` is (n_sym, fft_size) complex, natural bin order 0..fft_size-1,
    exactly what `Ofdm`'s `self.demod.process()` returns for the same
    time-domain slots -- which is what makes the split bit-exact: TD
    running the FFT up front over every symbol (DMRS included) produces
    the same bins `Ofdm` gets from `_estimate_channel_from_dmrs`'s own
    `demod.process` call on those slots.

    `symbol_offset` is the symbol's index within its frame, 0 = the first
    TRAINING symbol, counting the way `Ofdm`'s `pos` walks forward.

    `sample_offset` is ABSOLUTE in the incoming sample stream, not
    frame-relative. It exists so a stream gap or a re-synchronization is
    unambiguous downstream: two batches whose `sample_offset`s are not
    `slot_len` apart are not adjacent symbols, whatever their
    `symbol_offset`s say.

    `n_sym` is a tuning knob, not a design decision -- see
    `OfdmV2`'s `chunk_symbols`. One symbol per FIFO message is a guess,
    and `ofdm.py`'s own comments reject per-symbol Python calls as too
    slow in a batched design, so the batch size is swept (1/4/8/16)
    rather than assumed.
    """

    frame_id: int
    symbol_offset: int
    sample_offset: int
    bins: Any                      # (n_sym, fft_size) complex
    stype: SymbolType = SymbolType.BODY
    frame_start: bool = False

    @property
    def n_sym(self) -> int:
        return int(self.bins.shape[0])


@dataclass(frozen=True)
class HeaderConfig:
    """C1's payload, but travelling FORWARD on the frame's first
    `LlrBatch` instead of backward on its own wires.

    This is the whole reason V2 needs no feedback path. FD decodes the
    header itself, keeps the two fields it needs (`mod_scheme` to demap,
    `dmrs_interval` to classify and refresh H[k]), and forwards the rest
    so BIT never has to ask for anything: `payload_len_bits`,
    `encoded_bit_count` and the crc/fec0/fec1 codes are all BIT's
    business, not FD's.

    `fields` keeps `HeaderCodec.decode_bits()`'s own dict verbatim so the
    façade can hand callers exactly what `Ofdm.rx_process` puts under
    its "header" key -- the result contract is documented as a fully
    enumerated key set (`ofdm.py:1215`) and V2 must not narrow it.
    """

    mod_scheme: str
    crc: str
    fec0: str
    fec1: str
    payload_len_bits: int
    encoded_bit_count: int
    dmrs_interval: int
    c2_len_bytes: int
    n_payload_symbols: int
    n_c2_symbols: int
    n_data_total: int
    n_total_slots: int
    # Coded bits in ONE payload OFDM symbol, from the DECODED modem --
    # the inner interleaver's block size. It is carried here rather than
    # inferred downstream from "total bits / number of batches", which
    # was a real bug: in frame-scaling mode FD emits the whole frame as a
    # single batch, so the inferred block became the whole frame and the
    # inverse permutation silently ran on the wrong geometry. It also must
    # not come from the receiver's own `bits_per_ofdm_symbol`, which need
    # not match the frame's modulation.
    bits_per_symbol_payload: int = 0
    fields: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LlrBatch:
    """I2: FD -> BIT. Demapped soft values for one or more DATA/C2
    symbols, in transmission bit order.

    `llrs` is (n_sym * bits_per_ofdm_symbol,) in whatever representation
    the frame's soft configuration asks for, and that representation is
    NOT implied by this class -- see `soft` below. `hard_bits` carries the
    hard decision alongside, because `Ofdm` needs both (the hard bits are
    what the packetizer decodes when soft is off, and they are fused with
    the EVM power sums).

    `header` is present ONLY on the frame's first batch, which is what
    lets BIT configure itself without a backward channel.

    `last` marks the frame's final batch. BIT uses it to know the frame is
    complete, because -- see `OfdmV2`'s docstring -- BIT cannot decode
    incrementally: the outer block interleaver is sized by the whole
    frame's bit count, and `Ofdm` calls `decode_soft` once per frame, so
    chunking the Viterbi would shift results at chunk boundaries through
    the sliding-window traceback. Incremental TRANSPORT, frame-granular
    DECODE.
    """

    frame_id: int
    symbol_offset: int
    llrs: Optional[Any]            # (n_sym * bits_per_ofdm_symbol,) or None when soft is off
    hard_bits: Any                 # (n_sym * bits_per_ofdm_symbol,) uint8
    stype: SymbolType = SymbolType.DATA
    header: Optional[HeaderConfig] = None
    last: bool = False


@dataclass(frozen=True)
class SoftConfig:
    """The soft-LLR representation, stated explicitly because it is four
    independent knobs and `Ofdm`'s defaults differ from the FPGA profile
    on EVERY one of them:

        parameter          Ofdm default      FPGA profile
        soft_llr_metric    "maxlog"          "thresh_wq"
        soft_llr_bits      None (8-bit)      4
        soft_llr_clip      6.0               2-3
        soft_llr_scale     "frame"           "stream"

    An equivalence test that does not pin all four fails for reasons that
    have nothing to do with the refactor, which is why this is a type and
    not four loose kwargs.

    `scale` is the one that constrains the ARCHITECTURE, not just the
    numbers:

    * "frame" is NON-CAUSAL, twice over (`ofdm.py:1916-1930`):
      `w = w / xp.mean(w)` averages |H|^2 over the WHOLE payload, and
      `sigma2=None` makes `demodulate_soft` self-calibrate from the
      payload's own min-distances. FD cannot finalize symbol 0's LLRs
      until it has seen every symbol, so in this mode FD must buffer the
      frame before emitting. Inter-frame pipelining survives; intra-frame
      overlap does not. Supported for `Ofdm`-default equivalence.
    * "stream" is CAUSAL and was built for exactly this. `ofdm.py:1375`:
      "the noise power and the |H|^2 normalizer a streaming receiver has
      BEFORE its first payload symbol" -- the noise from the header
      symbols, the |H|^2 mean from the training estimate, with
      sigma2 := 1. This is V2's default, and the ONLY mode in which FD
      emits LLRs incrementally.

    Numerical LLRs differ between the two modes, so equivalence is tested
    per-mode and never across modes.
    """

    enabled: bool = True
    metric: str = "maxlog"
    bits: Optional[int] = None
    clip: float = 6.0
    scale: str = "stream"

    @property
    def is_causal(self) -> bool:
        """True when FD may emit a symbol's LLRs without having seen the
        rest of the frame. Drives whether FD streams or buffers."""
        return self.scale == "stream"


class ControlEvent:
    """Base for the out-of-band messages.

    These travel on a channel SEPARATE from the data FIFO. If they shared
    it, a full data queue would block the very abort that the full queue
    caused -- control must always be deliverable.

    The consequence, which is easy to get wrong: because the control
    channel is unordered with respect to the data queue, an abort for
    frame N arrives BEFORE some of frame N's queued chunks. So an abort
    cannot simply drop what is currently buffered. Each consumer keeps an
    invalidated set of frame_ids, discards chunks for them ON ARRIVAL,
    and retires an id only once that frame's data is known drained. The
    invariant that matters: no incomplete frame ever reaches MAC.
    """

    __slots__ = ()


@dataclass(frozen=True)
class FrameAbort(ControlEvent):
    """Abandon everything for `frame_id`.

    Emitted by TD when a candidate preamble wins and the active frame has
    to be given up, and by any stage on queue overflow. Overflow is
    frame-scoped on purpose: dropping an individual FFT or LLR chunk from
    mid-frame would hand a silently corrupted partial frame downstream,
    which is a correctness defect rather than a performance detail.
    """

    frame_id: int
    reason: str = ""


@dataclass(frozen=True)
class StreamGap(ControlEvent):
    """The sample stream is discontinuous at `sample_offset`.

    Distinct from `FrameAbort`: a gap says the input itself lost
    continuity (a dropped USB/PCIe buffer, a retune), so every stage's
    notion of symbol alignment is stale, not just one frame's.
    """

    sample_offset: int
    reason: str = ""


@dataclass(frozen=True)
class FrameDone(ControlEvent):
    """`frame_id` is finished downstream; its samples may be released.

    FLOW CONTROL, NOT CONFIGURATION. This travels from BIT back toward
    TD, which looks like the backward path V2 rejects and is not: it
    carries no decode information, only "nobody needs these samples any
    more". It is also strictly optional -- TD bounds every frame's
    emission at `max_payload_symbols` on its own, so a lost or never-sent
    FrameDone costs wasted work and never correctness.

    It exists because in the threaded flowgraph TD PUSHES symbols instead
    of being asked for them, and TD cannot know how long a frame is (that
    is in the header, which FD reads). Without this it would always run
    to the cap.
    """

    frame_id: int
    ok: bool = True
