"""`PhyConfig` / `PhyEnv` -- what the three V2 stages are built from.

TWO JOBS, KEPT APART ON PURPOSE.

`PhyConfig` is pure configuration: numbers and strategy names, nothing
constructed. `PhyEnv` is that config plus the geometry derived from it --
the resource grid, the preamble, the pilot values, the training symbol's
known content, the header's bit positions. All of that is IMMUTABLE and
therefore safe for several stages on several threads to read at once.

WHAT `PhyEnv` DELIBERATELY DOES NOT HOLD: the DSP blocks. It hands out
FACTORIES instead (`make_sync`, `make_cfo`, `make_demod`,
`make_equalizer`, `make_channel_estimator`), so every stage constructs
its own. That is the step-3 change, and the reason is not tidiness:

* Anything that can carry state must not be shared across threads, and
  "this block looks stateless today" is not a property anyone can rely
  on through a future optimization. A block that grows a cache -- exactly
  how this project accelerated the interleaver, the FEC codecs and the
  LDPC construction -- silently becomes a race.
* `framing/header.py:69` already keeps `_HEADER_PACKETIZER` as a
  module-level singleton and `ConvolutionalCode._native_soft` is built
  lazily with no lock (`fec/viterbi.py:174,329`). That is the shape of
  the hazard, in this codebase, today.

TWO WAYS TO BUILD, AND WHY BOTH EXIST.

`PhyEnv.build(config)` constructs everything from configuration, so V2
stands on its own. `PhyEnv.from_ofdm(ofdm)` borrows a configured `Ofdm`'s
geometry instead, which is what keeps the bit-exactness proof honest: V2
then runs against the same derived constants as the oracle, so a mismatch
can only be the partition's fault.

They must agree, or V2-standalone silently differs from the V2 that was
proved correct. `tests/test_ofdm_v2_equivalence.py` asserts that
agreement element by element rather than trusting that two copies of one
derivation stayed in step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ...backend import default_backend, get_xp
from ...framing.header import header_wire_len_bits
from ...ofdm import OfdmDemodulator, ResourceGrid
from ...registry import resolve

# Import for registration side effects, exactly as `Ofdm` does: the
# registry is populated by these modules being imported, so resolving
# "schmidl_cox" / "mmse" / "ls" by name fails without them.
from ...sync import SchmidlCoxSync  # noqa: F401
from ...cfo import SchmidlCoxCFO  # noqa: F401
from ...equalizer import MMSEEqualizer, ZFEqualizer  # noqa: F401
from ...channel import LSChannelEstimator, MMSEChannelEstimator  # noqa: F401


def _dc_is_nulled(grid: Any) -> bool:
    """Is the DC (centre) subcarrier among the nulls?

    `ResourceGrid` keeps the resulting index arrays but not the flag that
    produced them, so the flag is recovered from the geometry.
    """
    nulls = grid.null_indices
    host = nulls.get() if callable(getattr(nulls, "get", None)) else nulls
    return bool(int(grid.fft_size // 2) in set(int(i) for i in np.asarray(host)))


@dataclass(frozen=True)
class PhyConfig:
    """Everything needed to build a receiver, and nothing constructed.

    Defaults match `Ofdm`'s so that a `PhyConfig()` and a default-ish
    `Ofdm` describe the same radio. `timing_advance=None` resolves to
    `min(2, cp_len)`, as `Ofdm` does -- the preamble has no CP, so every
    window that follows inherits this one nudge into the cyclic prefix.
    """

    fft_size: int = 256
    cp_len: int = 32
    n_data: int = 216
    n_pilot: int = 8
    dc_null: bool = True
    n_training_symbols: int = 2

    sync: str = "schmidl_cox"
    cfo: str = "schmidl_cox"
    equalizer: str = "mmse"
    channel_estimator: str = "ls"

    # These MUST match `Ofdm`'s defaults, and the values are not
    # guessable. A wrong preamble_seed means the receiver never syncs; a
    # wrong training_seed means it syncs and then decodes nothing. Neither
    # produces an error -- the link is simply dead. An earlier version of
    # this file defaulted both to 0 and did exactly that, which is why
    # `test_standalone_env_matches_the_borrowed_one` compares the derived
    # arrays element by element instead of trusting the construction.
    preamble_seed: int = 123
    training_seed: int = 999
    header_scramble_seed: int = 42

    timing_advance: Optional[int] = None
    sync_threshold: float = 0.3
    max_payload_symbols: int = 128
    strict_fec_check: bool = False

    # Never signalled over the air (`framing/packetizer.py`'s module
    # docstring): the receiver must already be configured to match, so
    # these come from here and not from the decoded header.
    interleaver: str = "block"
    interleaver_kwargs: Dict[str, Any] = field(default_factory=lambda: {"unit_bits": 8})
    interleaver2: str = "block"
    interleaver2_kwargs: Dict[str, Any] = field(default_factory=dict)

    iq_dtype: str = "float32"
    backend: Optional[str] = None
    pilot_values: Optional[Any] = None


@dataclass(frozen=True)
class PhyEnv:
    """Immutable geometry plus factories for per-stage blocks."""

    cfg: PhyConfig
    backend: str
    xp: Any

    # --- derived geometry (immutable; shared is safe) -----------------
    slot_len: int
    timing_advance: int
    num_symbols_header: int
    header_wire_len: int
    grid: Any
    preamble_time: Any
    pilot_values: Any
    train_grid_freq: Any
    train_known_indices: Any
    train_known_values: Any
    # The header's bits are SPREAD across the header symbols' subcarriers
    # (plus scrambled) rather than packed densely into the first slots --
    # a PAPR fix, see `Ofdm`'s class docstring. The demapped wire bits
    # must be gathered through these positions, never plain-sliced.
    header_positions_flat: Any

    # --- convenience mirrors of cfg ----------------------------------
    @property
    def fft_size(self) -> int:
        return self.cfg.fft_size

    @property
    def cp_len(self) -> int:
        return self.cfg.cp_len

    @property
    def n_training_symbols(self) -> int:
        return self.cfg.n_training_symbols

    @property
    def sync_threshold(self) -> float:
        return self.cfg.sync_threshold

    @property
    def max_payload_symbols(self) -> int:
        return self.cfg.max_payload_symbols

    @property
    def strict_fec_check(self) -> bool:
        return self.cfg.strict_fec_check

    @property
    def interleaver(self) -> str:
        return self.cfg.interleaver

    @property
    def interleaver_kwargs(self) -> Dict[str, Any]:
        return self.cfg.interleaver_kwargs

    @property
    def interleaver2(self) -> str:
        return self.cfg.interleaver2

    @property
    def interleaver2_kwargs(self) -> Dict[str, Any]:
        return self.cfg.interleaver2_kwargs

    @property
    def iq_dtype(self) -> str:
        return self.cfg.iq_dtype

    @property
    def header_scramble_seed(self) -> int:
        return self.cfg.header_scramble_seed

    # --- per-stage block factories -----------------------------------
    #
    # One instance per stage, never shared. See the module docstring.

    def make_sync(self) -> Any:
        return resolve("sync", self.cfg.sync, fft_size=self.cfg.fft_size,
                       backend=self.backend)

    def make_cfo(self) -> Any:
        # Resolved with the pilot arguments `PilotBasedCFO` needs;
        # `SchmidlCoxCFO` ignores them through its **kwargs sink, so one
        # call site serves either strategy -- the same registry pattern
        # `Ofdm` uses.
        return resolve("cfo", self.cfg.cfo, fft_size=self.cfg.fft_size,
                       cp_len=self.cfg.cp_len,
                       pilot_indices=self.grid.pilot_indices,
                       tx_pilots=self.pilot_values,
                       n_repeats=self.cfg.n_training_symbols,
                       backend=self.backend)

    def make_demod(self) -> Any:
        return OfdmDemodulator(self.cfg.fft_size, self.cfg.cp_len, backend=self.backend)

    def make_equalizer(self) -> Any:
        return resolve("equalizer", self.cfg.equalizer, backend=self.backend)

    def make_channel_estimator(self) -> Any:
        return resolve("channel_estimator", self.cfg.channel_estimator,
                       pilot_indices=self.train_known_indices,
                       fft_size=self.cfg.fft_size,
                       tx_pilots=self.train_known_values,
                       cp_len=self.cfg.cp_len,
                       backend=self.backend)

    # --- construction -------------------------------------------------

    @classmethod
    def build(cls, cfg: Optional[PhyConfig] = None, **overrides: Any) -> "PhyEnv":
        """Derive the geometry from configuration alone.

        Every step here mirrors `Ofdm.__init__`'s derivation, including
        the seeds, because the two must produce identical constants -- a
        receiver whose training reference differs by one subcarrier
        decodes nothing, and would do so without any error.
        """
        from dataclasses import replace as _replace
        cfg = cfg or PhyConfig()
        if overrides:
            cfg = _replace(cfg, **overrides)

        backend = cfg.backend or default_backend()
        xp = get_xp(backend)

        if cfg.timing_advance is None:
            timing_advance = min(2, cfg.cp_len)
        elif not (0 <= cfg.timing_advance <= cfg.cp_len):
            raise ValueError(
                f"timing_advance={cfg.timing_advance!r}; expected 0..cp_len "
                f"({cfg.cp_len}). The advance moves every FFT window earlier into "
                f"the cyclic prefix and must leave room for the channel's delay spread"
            )
        else:
            timing_advance = int(cfg.timing_advance)

        grid = ResourceGrid(fft_size=cfg.fft_size, n_data=cfg.n_data,
                            n_pilot=cfg.n_pilot, dc_null=cfg.dc_null)

        wire_len = header_wire_len_bits()
        num_symbols_header = math.ceil(wire_len / grid.n_data)
        total_header_slots = num_symbols_header * grid.n_data
        positions = np.unique(
            np.linspace(0, total_header_slots - 1, wire_len).round().astype(int))
        if len(positions) != wire_len:
            raise ValueError(
                f"n_data={cfg.n_data} gives only {len(positions)} distinct spread "
                f"positions for the header's {wire_len} bits across "
                f"{num_symbols_header} symbol(s) (rounding collision) -- use a larger n_data"
            )

        sync_block = resolve("sync", cfg.sync, fft_size=cfg.fft_size, backend=backend)
        if not hasattr(sync_block, "generate_preamble"):
            raise TypeError(
                f"sync={cfg.sync!r} has no generate_preamble(); the receiver needs a "
                f"sync strategy that can generate its own preamble"
            )
        preamble_time = sync_block.generate_preamble(seed=cfg.preamble_seed)

        pilot_values = (xp.ones((grid.n_pilot,), dtype="complex64")
                        if cfg.pilot_values is None
                        else xp.asarray(cfg.pilot_values, dtype="complex64"))

        # The SAME training symbol repeated n_training_symbols times (as
        # 802.11's Long Training Field is sent twice), so the receiver can
        # average across repetitions to cut noise.
        rng = np.random.default_rng(cfg.training_seed)
        alphabet = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j],
                            dtype="complex64") / np.sqrt(2)
        train_data = xp.asarray(alphabet[rng.integers(0, 4, size=grid.n_data)])
        train_grid_freq = grid.scatter(xp, pilot_values[None, :], train_data[None, :])[0]
        train_known_indices = xp.asarray(
            np.sort(np.concatenate([grid.pilot_indices, grid.data_indices])))
        train_known_values = train_grid_freq[train_known_indices]

        return cls(
            cfg=cfg, backend=backend, xp=xp,
            slot_len=cfg.cp_len + cfg.fft_size,
            timing_advance=timing_advance,
            num_symbols_header=num_symbols_header,
            header_wire_len=wire_len,
            grid=grid, preamble_time=preamble_time, pilot_values=pilot_values,
            train_grid_freq=train_grid_freq,
            train_known_indices=train_known_indices,
            train_known_values=train_known_values,
            header_positions_flat=positions,
        )

    @classmethod
    def from_ofdm(cls, ofdm: Any) -> "PhyEnv":
        """Borrow a configured `Ofdm`'s geometry.

        The honest form of the equivalence proof: V2 then runs against
        the oracle's own derived constants, so any mismatch is the
        partition's fault rather than a construction difference. Also the
        only way to mirror an `Ofdm` whose attributes were adjusted after
        construction -- the tests raise `MAX_PAYLOAD_SYMBOLS` that way.
        """
        cfg = PhyConfig(
            fft_size=ofdm.fft_size, cp_len=ofdm.cp_len,
            n_data=ofdm.grid.n_data, n_pilot=ofdm.grid.n_pilot,
            n_training_symbols=ofdm.n_training_symbols,
            # ResourceGrid does not keep the flag, so it is read back
            # from the geometry: dc_null means the centre bin is nulled.
            dc_null=_dc_is_nulled(ofdm.grid),
            preamble_seed=ofdm.preamble_seed, training_seed=ofdm.training_seed,
            timing_advance=ofdm.timing_advance,
            sync_threshold=ofdm.sync_threshold,
            max_payload_symbols=ofdm.MAX_PAYLOAD_SYMBOLS,
            strict_fec_check=ofdm.strict_fec_check,
            interleaver=ofdm.interleaver,
            interleaver_kwargs=dict(ofdm.interleaver_kwargs),
            interleaver2=ofdm.interleaver2,
            interleaver2_kwargs=dict(ofdm.interleaver2_kwargs),
            iq_dtype=ofdm.iq_dtype, backend=ofdm.backend,
        )
        return cls(
            cfg=cfg, backend=ofdm.backend, xp=ofdm.xp,
            slot_len=ofdm.slot_len, timing_advance=ofdm.timing_advance,
            num_symbols_header=ofdm.num_symbols_header,
            header_wire_len=ofdm.header_wire_len_bits,
            grid=ofdm.grid, preamble_time=ofdm._preamble_time,
            pilot_values=ofdm.pilot_values,
            train_grid_freq=ofdm._train_grid_freq,
            train_known_indices=ofdm._train_known_indices,
            train_known_values=ofdm._train_grid_freq[ofdm._train_known_indices],
            header_positions_flat=ofdm._header_positions_flat,
        )
