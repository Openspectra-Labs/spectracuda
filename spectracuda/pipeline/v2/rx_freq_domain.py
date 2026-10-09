"""FD -- the frequency domain stage: channel estimation, THE HEADER
DECODE, DMRS refresh, equalization, pilot-CPE correction and demapping.

WHY THE HEADER DECODER LIVES HERE. This is the design decision the whole
of V2 turns on. FD cannot demap a payload symbol without knowing the
constellation -- 216 data subcarriers are 432 LLRs at QPSK, 864 at 16-QAM,
1296 at 64-QAM, and the modulation is chosen per frame by the transmitter
and announced in the header. It cannot even EQUALIZE the body without
`dmrs_period`, because DMRS symbols are what refresh H[k] and FD has to
know which arriving symbols those are.

The RTL answers this by sending the config backward: `rx_bit_domain.v:86`
decodes the header and drives `cfg_mod`/`cfg_dmrs_period` into
`rx_freq_domain.v:100`, with FD holding body symbols in B1 until it
arrives. V2 answers it by decoding the header HERE instead, so the
pipeline is strictly forward and no stage ever waits on a stage
downstream of it.

That costs almost nothing: the header is 2 BPSK symbols through one
`Packetizer(crc="crc16", fec="conv_v27")` with NO interleaver
(`framing/header.py:69`). It is not a second copy of the payload FEC
chain, and the RTL already keeps a separate header Viterbi
(`header_decode_v3.v`) distinct from `viterbi_dec` -- so this is a
placement difference, not duplicated logic.

What FD forwards instead of asking for: everything BIT needs
(`payload_len_bits`, `encoded_bit_count`, the crc/fec codes) rides on the
frame's first `LlrBatch` as a `HeaderConfig`.

STREAMING VS BUFFERING, AND WHY IT IS NOT FD'S CHOICE. FD emits a
symbol's LLRs the moment it has them -- but only when the soft scaling is
causal. `soft_llr_scale="frame"` normalizes by the mean |H|^2 over the
WHOLE payload and lets `demodulate_soft` self-calibrate sigma^2 from the
payload's own min-distances (`ofdm.py:1916-1930`), so symbol 0's LLRs
genuinely depend on the last symbol and FD must hold the frame. "stream"
takes the noise from the header symbols and the |H|^2 mean from the
training estimate -- "what a streaming receiver has BEFORE its first
payload symbol" (`ofdm.py:1375`) -- and is V2's default for exactly that
reason. See `SoftConfig.is_causal`.

PER-SYMBOL IS BIT-EXACT, AND THAT IS NOT AN ACCIDENT. `Ofdm` equalizes,
CPE-corrects and demaps all payload symbols in one batched call, but every
one of those steps is per-row: equalization is per subcarrier per row, the
CPE rotation is one scalar per row, and `demodulate_stats` reduces within
a row. So walking symbols one at a time changes the loop and not the
arithmetic. The two exceptions are both scaling terms, and both are
handled above: the frame-mode mean, and the per-frame scalars that
"stream" mode deliberately derives from the training and header symbols.

SCOPE: the C2 control region is NOT implemented here. Nothing in the
current configurations uses it, and the RTL has none either
(`rx_freq_domain.v` requires `cfg_c2_syms == 0`), so a frame that
advertises C2 raises rather than being silently mis-split.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ...block import Block
from ...framing import c2 as _c2
from ...framing import dmrs as _dmrs
from ...framing.header import HeaderCodec
from ...framing.packetizer import Packetizer
from ...modem import Modem
from .env import PhyEnv
from .stage_if import FftBatch, HeaderConfig, LlrBatch, SoftConfig, SymbolType


@dataclass
class _FrameState:
    """FD's private per-frame state.

    This is what `Ofdm` parks on `self` -- and the reason it cannot be
    threaded. Here it is keyed by frame_id and owned by FD alone, so two
    frames can be in flight without either seeing the other's channel
    estimate or noise reference.
    """

    frame_id: int
    h_data: Any = None
    h_pilots: Any = None
    train_h2_sum: Any = None
    stream_noise: Any = None
    stream_h2_mean: Any = None
    header: Optional[HeaderConfig] = None
    payload_modem: Any = None
    bits_per_symbol_payload: int = 0
    slot_map: Any = None
    slot_cursor: int = 0          # index into slot_map
    data_emitted: int = 0
    evm_err: float = 0.0
    evm_ref: float = 0.0
    # frame-mode only: held because the scale is non-causal
    buffered_eq: List[Any] = field(default_factory=list)
    buffered_h: List[Any] = field(default_factory=list)


class RxFreqDomain(Block):
    """`FftBatch` in, `LlrBatch` out, plus the header decoded locally."""

    batch_shape_doc = (
        "FftBatch(bins=(n_sym, fft_size) complex64) in -> "
        "LlrBatch(hard_bits=(n_sym*bits_per_ofdm_symbol,) uint8, llrs=same or None) out; "
        "header decoded in-stage, forwarded on the frame's first LlrBatch"
    )

    def __init__(self, env: PhyEnv, soft: Optional[SoftConfig] = None) -> None:
        super().__init__(backend=env.backend)
        self.env = env
        self.soft = soft or SoftConfig()
        # Per-stage instances, never shared. `framing/header.py:69` keeps
        # `_HEADER_PACKETIZER` as a module-level singleton and
        # `ConvolutionalCode._native_soft` is lazily built with no lock
        # (`fec/viterbi.py:174,329`), so one object touched by RX-FD and
        # RX-BIT concurrently is a genuine race.
        self._header_codec = HeaderCodec(scramble_seed=env.header_scramble_seed)
        self._pilot_values = env.pilot_values
        self._cfg_fec0: Optional[str] = None
        self._cfg_fec1: Optional[str] = None
        self._modem_cache: Dict[str, Any] = {}
        self._frames: Dict[int, _FrameState] = {}
        self.invalidated: set = set()
        self.stats = {"headers_decoded": 0, "header_failures": 0,
                      "dmrs_refreshes": 0, "data_symbols": 0, "dropped_invalid": 0}

    # ---- lifecycle ---------------------------------------------------

    def abort(self, frame_id: int) -> None:
        """Honour a `FrameAbort`.

        The id goes into `invalidated` and STAYS there, rather than only
        dropping what is currently held: the control channel is unordered
        with respect to the data queue, so an abort arrives before some of
        that frame's chunks and those must be discarded on arrival too.
        """
        self._frames.pop(frame_id, None)
        self.invalidated.add(frame_id)

    def retire(self, frame_id: int) -> None:
        """Release a completed frame and stop invalidating its id."""
        self._frames.pop(frame_id, None)
        self.invalidated.discard(frame_id)

    def _live(self, frame_id: int) -> bool:
        if frame_id in self.invalidated:
            self.stats["dropped_invalid"] += 1
            return False
        return True

    # ---- training ----------------------------------------------------

    def training(self, batch: FftBatch) -> None:
        """Estimate H[k] from the TRAINING symbol(s).

        Averaged over `n_training_symbols` exactly as `Ofdm` does, and
        `h_pilots` is accumulated alongside `h_data` because the CPE
        correction below needs the channel at the PILOT subcarriers --
        not `h_data` re-indexed, since the estimate genuinely differs per
        subcarrier.
        """
        env, xp = self.env, self.env.xp
        if not self._live(batch.frame_id):
            return
        fs = self._frames.setdefault(batch.frame_id, _FrameState(batch.frame_id))

        h_data_sum = h_pilots_sum = None
        for i in range(batch.n_sym):
            known = batch.bins[i:i + 1, :][:, env.train_known_indices]
            h_full = env.channel_estimator.process(known)
            h_d = h_full[:, env.grid.data_indices]
            h_p = h_full[:, env.grid.pilot_indices]
            h_data_sum = h_d if h_data_sum is None else h_data_sum + h_d
            h_pilots_sum = h_p if h_pilots_sum is None else h_pilots_sum + h_p
        fs.h_data = h_data_sum / env.n_training_symbols
        fs.h_pilots = h_pilots_sum / env.n_training_symbols
        # Needed by the thresh_w/thresh_wq metrics, which compare
        # n_data*|H[k]|^2 against this sum -- the RTL's own comparison
        # (llr_weight.v), so it must come from the TRAINING estimate and
        # not from the payload.
        fs.train_h2_sum = xp.sum(xp.abs(fs.h_data) ** 2, axis=-1)

    # ---- the header --------------------------------------------------

    def decode_header(self, batch: FftBatch) -> HeaderConfig:
        """Equalize, BPSK-demap and DECODE the header, in this stage.

        Also derives the two per-frame scalars `"stream"` scaling needs,
        from the header's own equalized symbols: this is the point at
        which a streaming receiver has them, and deriving them anywhere
        later would make the scale non-causal.
        """
        env, xp = self.env, self.env.xp
        fs = self._frames[batch.frame_id]

        bits_chunks, eq_chunks = [], []
        for i in range(batch.n_sym):
            rx_data = env.grid.extract_data(xp, batch.bins[i:i + 1, :])
            eq = env.equalizer.process(rx_data, channel_est=fs.h_data)
            bits_chunks.append(env.header_modem.demodulate(eq))
            eq_chunks.append(eq)

        if self.soft.scale == "stream":
            heq = xp.concatenate(eq_chunks, axis=-1)
            hp, _ = env.header_modem._point_table()
            hp = xp.asarray(hp)
            hd = xp.min(xp.abs(heq[..., None] - hp[None, None, :]) ** 2, axis=-1)
            fs.stream_noise = xp.maximum(xp.mean(hd, axis=-1), 1e-12)
            fs.stream_h2_mean = xp.maximum(xp.mean(xp.abs(fs.h_data) ** 2, axis=-1), 1e-12)

        wire = xp.concatenate(bits_chunks, axis=-1)
        fields = self._decode_header_bits(wire)
        cfg = self._build_config(fields)
        fs.header = cfg
        fs.payload_modem = self._modem_for(cfg.mod_scheme)
        fs.bits_per_symbol_payload = env.grid.n_data * fs.payload_modem.bits_per_symbol
        fs.slot_map = _dmrs.dmrs_slot_map(cfg.n_data_total, cfg.dmrs_interval)
        self.stats["headers_decoded"] += 1
        return cfg

    def _decode_header_bits(self, wire: Any) -> Dict[str, Any]:
        """Gather the content bits, then decode.

        The gather is NOT a slice. The header's bits are spread across the
        header symbols' subcarriers by `header_positions_flat` (a PAPR fix
        -- see `Ofdm`'s class docstring), so plain-slicing the first
        `wire_len` positions would hand the codec filler and get a
        plausible-looking garbage header. Item 0 carries the shared header
        content, matching `Ofdm._decode_header_symbols`'s own
        "one call = one frame" convention.
        """
        host = np.asarray(self._to_host(wire))[0]
        content = host[self.env.header_positions_flat]
        try:
            return self._header_codec.decode_bits(content)
        except Exception:
            self.stats["header_failures"] += 1
            raise

    def _build_config(self, fields: Dict[str, Any]) -> HeaderConfig:
        """Turn decoded header fields into the forward-travelling config.

        `strict_fec_check` is applied BEFORE any codec is constructed, for
        the reason `Ofdm` gives: a false sync detection on noise draws a
        random-but-valid fec0 code often enough, and most such draws land
        on an LDPC variant whose construction costs up to ~1.6 s on a
        Pi 5. Rejecting first is what stops a frame that was never there
        from stalling the receiver.
        """
        env = self.env
        if env.strict_fec_check and (fields["fec0"] != self._cfg_fec0 or
                                     fields["fec1"] != self._cfg_fec1):
            raise ValueError(
                f"decoded header fec0={fields['fec0']!r}/fec1={fields['fec1']!r} does not "
                f"match this receiver's fec={self._cfg_fec0!r}/fec1={self._cfg_fec1!r} "
                f"-- rejecting before constructing a codec (likely a false sync detection)"
            )

        c2_len = fields["c2_len_bytes"]
        if c2_len:
            raise NotImplementedError(
                f"c2_len_bytes={c2_len}: the C2 control region is not implemented in "
                f"OfdmV2's frequency domain yet (the RTL has none either -- "
                f"rx_freq_domain.v requires cfg_c2_syms == 0). Raising rather than "
                f"mis-splitting the payload region."
            )

        modem = self._modem_for(fields["mod_scheme"])
        bits_per_sym = env.grid.n_data * modem.bits_per_symbol
        pkt = self._packetizer_for(fields)
        raw_len = fields["payload_len_bits"]
        try:
            encoded = pkt.encoded_length(raw_len)
        except ValueError as exc:
            raise ValueError(
                f"decoded payload_len_bits={raw_len} is incompatible with decoded "
                f"crc={fields['crc']!r}/fec0={fields['fec0']!r}/fec1={fields['fec1']!r} "
                f"-- likely header corruption ({exc})"
            ) from exc

        # Ceiling, not exact division: the transmitter pads a partial last
        # payload symbol, and `encoded_bit_count` already says where the
        # real data ends, so no extra wire field is needed to tell filler
        # from payload.
        n_pay = math.ceil(encoded / bits_per_sym)
        n_c2 = _c2.n_c2_symbols(c2_len, env.grid.n_data)
        n_data_total = n_c2 + n_pay
        n_total = _dmrs.total_slots(n_data_total, fields["dmrs_interval"])
        if n_total > env.max_payload_symbols:
            # A bad header decode returning a huge garbage count used to
            # crash deep inside slot extraction; fail clearly here.
            raise ValueError(
                f"decoded header claims {n_pay} payload + {n_total - n_data_total} DMRS "
                f"= {n_total} slots, exceeding MAX_PAYLOAD_SYMBOLS="
                f"{env.max_payload_symbols} -- likely header corruption"
            )
        return HeaderConfig(
            mod_scheme=fields["mod_scheme"], crc=fields["crc"],
            fec0=fields["fec0"], fec1=fields["fec1"],
            payload_len_bits=raw_len, encoded_bit_count=encoded,
            dmrs_interval=fields["dmrs_interval"], c2_len_bytes=c2_len,
            n_payload_symbols=n_pay, n_c2_symbols=n_c2,
            n_data_total=n_data_total, n_total_slots=n_total,
            bits_per_symbol_payload=bits_per_sym,
            fields=dict(fields),
        )

    # ---- body --------------------------------------------------------

    def body(self, batch: FftBatch) -> List[LlrBatch]:
        """Classify, refresh H[k] on DMRS, equalize, CPE-correct, demap.

        Returns zero or more `LlrBatch`es: zero when every symbol in the
        batch was DMRS (consumed here -- DMRS never leaves FD) or when the
        frame is already complete, and zero in frame-scaling mode until
        the frame's last symbol arrives.
        """
        env, xp = self.env, self.env.xp
        if not self._live(batch.frame_id):
            return []
        fs = self._frames[batch.frame_id]
        cfg = fs.header
        if cfg is None:
            raise RuntimeError(f"frame {batch.frame_id}: body before header")

        out: List[LlrBatch] = []
        for i in range(batch.n_sym):
            if fs.slot_cursor >= cfg.n_total_slots:
                break                       # past the frame; TD over-emits by design
            kind = int(np.asarray(fs.slot_map)[fs.slot_cursor])
            fs.slot_cursor += 1
            bins = batch.bins[i:i + 1, :]

            if kind == _dmrs.DMRS_SLOT:
                # A DMRS *is* the training symbol re-transmitted, so it
                # goes through the identical estimator path -- if the two
                # ever diverged, segment 0 and segments 1+ would be
                # estimates of subtly different things while looking
                # equally plausible.
                h_full = env.channel_estimator.process(bins[:, env.train_known_indices])
                fs.h_data = h_full[:, env.grid.data_indices]
                fs.h_pilots = h_full[:, env.grid.pilot_indices]
                self.stats["dmrs_refreshes"] += 1
                continue

            eq = self._equalize_and_correct(fs, bins)
            self.stats["data_symbols"] += 1
            is_last = (fs.data_emitted + 1 == cfg.n_data_total)

            if self.soft.is_causal or not self.soft.enabled:
                out.append(self._emit(fs, eq, fs.h_data, is_last))
            else:
                # Frame-mode scaling is non-causal, so hold the symbol.
                fs.buffered_eq.append(eq)
                fs.buffered_h.append(fs.h_data)
                fs.data_emitted += 1
                if is_last:
                    out.extend(self._flush_frame_mode(fs))
                continue
            fs.data_emitted += 1
        return out

    def _equalize_and_correct(self, fs: _FrameState, bins: Any) -> Any:
        """Equalize one symbol and remove its common phase error.

        The CPE term is measured from the symbol's OWN pilots, equalized
        through the same equalizer as the data: the static channel gain at
        the pilot subcarriers is not 1.0 in general, so comparing raw
        pilots would fold channel response into what is meant to be a
        pure drift measurement.
        """
        env, xp = self.env, self.env.xp
        rx_data = env.grid.extract_data(xp, bins)
        eq = env.equalizer.process(rx_data, channel_est=fs.h_data)

        pilots_rx = env.grid.extract_pilots(xp, bins)
        eq_pilots = env.equalizer.process(pilots_rx, channel_est=fs.h_pilots)
        ratio = eq_pilots / xp.tile(self._pilot_values, (eq_pilots.shape[0], 1))
        ratio_mean = xp.mean(ratio, axis=-1)
        cpe = xp.angle(ratio_mean)
        # Reliability gate: |ratio_mean| is near 1 when the pilot phases
        # agree and collapses toward 0 when they are scattered (a deep
        # fade). Leave such a symbol uncorrected rather than rotating it
        # by a phase averaged out of noise.
        reliable = xp.abs(ratio_mean) > 0.05
        cpe = xp.where(reliable, cpe, xp.zeros_like(cpe))
        return eq * xp.exp(-1j * cpe)[:, None]

    def _emit(self, fs: _FrameState, eq: Any, h: Any, is_last: bool) -> LlrBatch:
        """Demap one symbol, accumulate its EVM power sums, package it."""
        env = self.env
        hard, err, ref = fs.payload_modem.demodulate_stats(eq)
        fs.evm_err += float(self._to_host(err).sum())
        fs.evm_ref += float(self._to_host(ref).sum())
        llrs = self._soft_for(fs, eq, h) if (self.soft.enabled and fs.payload_modem) else None
        header = fs.header if fs.data_emitted == 0 else None
        return LlrBatch(
            frame_id=fs.frame_id, symbol_offset=fs.data_emitted,
            llrs=llrs, hard_bits=hard.reshape(-1),
            stype=SymbolType.DATA, header=header, last=is_last,
        )

    def _flush_frame_mode(self, fs: _FrameState) -> List[LlrBatch]:
        """Demap a held frame once the non-causal scale can be computed.

        Only reachable with `soft_llr_scale="frame"`, which exists for
        `Ofdm`-default equivalence. Inter-frame pipelining survives;
        intra-frame overlap is given up, by construction.
        """
        xp = self.env.xp
        eq_all = xp.concatenate(fs.buffered_eq, axis=0)
        h_all = xp.concatenate(fs.buffered_h, axis=0)
        n = eq_all.shape[0]
        hard, err, ref = fs.payload_modem.demodulate_stats(eq_all)
        fs.evm_err = float(self._to_host(err).sum())
        fs.evm_ref = float(self._to_host(ref).sum())
        w = xp.abs(h_all) ** 2
        w = w / xp.mean(w)
        llrs = fs.payload_modem.demodulate_soft(
            eq_all, weight=w, llr_clip=self.soft.clip,
            llr_bits=self.soft.bits, sigma2=None)
        fs.buffered_eq, fs.buffered_h = [], []
        return [LlrBatch(
            frame_id=fs.frame_id, symbol_offset=0,
            llrs=llrs.reshape(-1), hard_bits=hard.reshape(-1),
            stype=SymbolType.DATA, header=fs.header, last=True,
        )]

    def _soft_for(self, fs: _FrameState, eq: Any, h: Any) -> Any:
        """One symbol's soft values under a CAUSAL scale.

        Every term here is known before the payload starts: the noise from
        the header symbols, the |H|^2 mean from the training estimate, the
        thresh_* reference from `train_h2_sum`. That is what makes
        emitting a symbol immediately legitimate rather than merely
        convenient.
        """
        env, xp = self.env, self.env.xp
        metric = self.soft.metric
        if metric in ("thresh_w", "thresh_wq"):
            # |H[k]|^2 / mean rounded to a power of two and clamped to
            # [1/8, 2] -- exactly the RTL's comparisons in llr_weight.v.
            h2 = xp.abs(h) ** 2
            ref = fs.train_h2_sum[:, None] * float(np.sqrt(2.0))
            kexp = xp.full(h2.shape, -3, dtype="int64")
            for mm in (-3, -2, -1, 0):
                kexp = kexp + (h2 * env.grid.n_data >= ref * (2.0 ** mm))
            if metric == "thresh_wq":
                return fs.payload_modem.demodulate_soft_tableq(eq, kexp).reshape(-1)
            return fs.payload_modem.demodulate_soft_thresh(
                eq, kexp, llr_bits=self.soft.bits or 4).reshape(-1)
        w = xp.abs(h) ** 2 / (fs.stream_h2_mean * fs.stream_noise)[:, None]
        return fs.payload_modem.demodulate_soft(
            eq, weight=w, llr_clip=self.soft.clip,
            llr_bits=self.soft.bits, sigma2=1.0).reshape(-1)

    def evm(self, frame_id: int) -> Any:
        """Normalized RMS EVM over the frame's UNTRUNCATED symbols.

        Untruncated on purpose, matching `Ofdm`: the last symbol's filler
        bits are still real EVM data, they are just not real codeword
        bits. float32 to match `compute_evm`'s dtype for complex64 input.
        """
        fs = self._frames[frame_id]
        return self.env.xp.asarray([math.sqrt(fs.evm_err / fs.evm_ref)], dtype="float32")

    # ---- plumbing ----------------------------------------------------

    def configure_expected_fec(self, fec0: str, fec1: str) -> None:
        """Supply what `strict_fec_check` compares against.

        Out-of-band because it is a property of this receiver, not of the
        wire -- the same reason the interleaver is never signalled.
        """
        self._cfg_fec0, self._cfg_fec1 = fec0, fec1

    def set_pilot_values(self, pilot_values: Any) -> None:
        self._pilot_values = pilot_values

    def _modem_for(self, scheme: str) -> Any:
        if scheme not in self._modem_cache:
            self._modem_cache[scheme] = Modem(scheme, backend=self.env.backend)
        return self._modem_cache[scheme]

    def _packetizer_for(self, fields: Dict[str, Any]) -> Any:
        """A throwaway packetizer, used ONLY for `encoded_length()`.

        FD needs the encoded bit count to size the frame; it does not
        decode anything. BIT builds its own decoding packetizer from the
        same forwarded fields, so neither stage shares a codec with the
        other.
        """
        env = self.env
        return Packetizer(
            crc=fields["crc"], fec=fields["fec0"], fec1=fields["fec1"],
            interleaver=env.interleaver, interleaver_kwargs=env.interleaver_kwargs,
            backend=env.backend,
        )

    def _to_host(self, arr: Any) -> np.ndarray:
        get = getattr(arr, "get", None)
        return np.asarray(get() if callable(get) else arr)

    def process(self, batch: Any, **kwargs: Any) -> Any:
        """`Block`'s entry point. The stage is driven by the three
        explicit calls (`training`, `decode_header`, `body`) because the
        symbol's role decides what FD does with it, and that role is
        carried on the batch rather than inferred."""
        if not isinstance(batch, FftBatch):
            raise TypeError("RxFreqDomain.process expects an FftBatch")
        if batch.stype == SymbolType.TRAIN:
            self.training(batch)
            return []
        if batch.stype == SymbolType.HEADER:
            self.decode_header(batch)
            return []
        return self.body(batch)
