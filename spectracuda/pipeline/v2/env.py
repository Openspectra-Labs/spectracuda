"""`PhyEnv` -- the geometry and DSP blocks the three V2 stages share.

WHY THIS TYPE EXISTS RATHER THAN THE STAGES TAKING AN `Ofdm`. Step 1's
whole job is to prove the TD/FD/BIT partition is bit-exact, and the
cheapest way to make that proof honest is for V2 to run the SAME block
instances `Ofdm` runs -- identical arithmetic on identical data, so any
difference is the partition's fault and nothing else. `PhyEnv.from_ofdm`
does that.

But if the stages took an `Ofdm` directly, V2 would never be able to
stand on its own, and the shortcut would quietly become the design. So
the dependency is named, narrow, and constructible two ways: borrowed
from a V1 object today, built directly once the stages own their blocks
(plan step 3). The stages only ever see `PhyEnv`.

WHAT IS NOT HERE, DELIBERATELY: anything per-frame, and anything mutable
that two stages would both touch. `Ofdm` parks per-frame state on `self`
(`_stream_noise`/`_stream_h2_mean` are written by its header decode at
`ofdm.py:1385` and read by its payload decode at `:1937`), which is
exactly what makes it unthreadable. `PhyEnv` is read-only configuration;
per-frame state belongs to whichever stage owns that step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhyEnv:
    """Read-only PHY geometry plus the stateless DSP blocks.

    `sync`, `cfo`, `demod`, `grid`, `equalizer`, `channel_estimator` and
    `header_modem` are shared because they carry no per-frame state --
    they are transforms. The things that DO carry state (the payload
    `Modem`, the `Packetizer` and its FEC codecs) are deliberately absent:
    `framing/header.py:69` keeps `_HEADER_PACKETIZER` as a module-level
    singleton and `ConvolutionalCode._native_soft` is lazily built with no
    lock (`fec/viterbi.py:174,329`), so a single instance touched by RX-FD,
    RX-BIT and TX-BIT concurrently is a genuine race. Each stage builds
    its own and warms it at startup.
    """

    # --- geometry -----------------------------------------------------
    fft_size: int
    cp_len: int
    slot_len: int
    timing_advance: int
    n_training_symbols: int
    num_symbols_header: int
    max_payload_symbols: int
    sync_threshold: float
    strict_fec_check: bool
    backend: str
    xp: Any

    # --- stateless transforms (shared) --------------------------------
    sync: Any
    cfo: Any
    demod: Any
    grid: Any
    equalizer: Any
    channel_estimator: Any
    header_modem: Any
    train_known_indices: Any

    # --- out-of-band receiver configuration ---------------------------
    # The interleaver is NEVER signalled over the air (see
    # `framing/packetizer.py`'s module docstring): the receiver must
    # already be configured with the matching one. So unlike mod/crc/fec,
    # these come from configuration and not from the decoded header.
    interleaver: str
    interleaver_kwargs: dict
    interleaver2: str
    interleaver2_kwargs: dict
    # "float32" (the default) means no quantization; anything else makes
    # TD round real/imag through float16 and back, simulating a
    # finite-resolution ADC at the boundary rather than in compute.
    iq_dtype: str

    @classmethod
    def from_ofdm(cls, ofdm: Any) -> "PhyEnv":
        """Borrow geometry and blocks from a configured `Ofdm`.

        Step-1 scaffolding, and the reason the equivalence gate is
        meaningful: V2 and V1 then run the same transforms on the same
        samples, so a mismatch can only come from the partition.
        """
        return cls(
            fft_size=ofdm.fft_size,
            cp_len=ofdm.cp_len,
            slot_len=ofdm.slot_len,
            timing_advance=ofdm.timing_advance,
            n_training_symbols=ofdm.n_training_symbols,
            num_symbols_header=ofdm.num_symbols_header,
            max_payload_symbols=ofdm.MAX_PAYLOAD_SYMBOLS,
            sync_threshold=ofdm.sync_threshold,
            strict_fec_check=ofdm.strict_fec_check,
            backend=ofdm.backend,
            xp=ofdm.xp,
            sync=ofdm.sync,
            cfo=ofdm.cfo,
            demod=ofdm.demod,
            grid=ofdm.grid,
            equalizer=ofdm.equalizer,
            channel_estimator=ofdm.channel_estimator,
            header_modem=ofdm.header_modem,
            train_known_indices=ofdm._train_known_indices,
            interleaver=ofdm.interleaver,
            interleaver_kwargs=dict(ofdm.interleaver_kwargs),
            interleaver2=ofdm.interleaver2,
            interleaver2_kwargs=dict(ofdm.interleaver2_kwargs),
            iq_dtype=ofdm.iq_dtype,
        )
