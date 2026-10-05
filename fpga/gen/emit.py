"""Emit HLS configuration headers from a constructed `Ofdm` object.

This is the generator half of the architecture in
`docs/vitis-hls-ofdm-ip-plan.md` section 2: **Python emits parameters,
tables and ROM contents; it never emits DSP source code.** The blocks in
`hls/src/` are hand-written once and `#include` what this produces.

## Why it takes an `Ofdm` OBJECT, not a dict of parameters

Construction is already the validation. `Ofdm.__init__` refuses unknown
FEC/CRC schemes, `n_training_symbols < 1`, `cp_len >= fft_size`, bad
`iq_dtype`, and (via `SchmidlCoxCFO`/`PilotBasedCFO`) incompatible
cfo/training combinations. An incoherent waveform cannot reach this
module, because Python declined to build one.

So `validate()` below only carries the **HLS-specific delta** -- things
Python accepts that fabric cannot express. That set is small and
enumerable, and catching it here is the difference between a clear error
now and a confusing synthesis failure three hours from now.

## Emitted files

    ofdm_params.h      geometry constants + strategy selection macros
    subcarrier_map.h   data / pilot / null index tables
    preamble_rom.h     Schmidl-Cox preamble, time domain
    training_rom.h     known training symbol + pilot values

All tables are `static const` at file scope, matching the convention
already used by `~/work/ofdm-hls/src/ofdm_lut.h`: each translation unit
that includes the header gets its own ROM, which is what HLS wants for
per-module BRAM inference.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, List

import numpy as np

from spectracuda.pipeline import Ofdm

# Three types, not one. The distinction is not cosmetic -- getting it
# wrong silently saturates a constant, which is the worst kind of bug
# because nothing reports it.
#
#   sample_t  the signal datapath. Normalized well below 1.0 (measured
#             peak 0.9 at the ADC after AGC), so ap_fixed<16,1> fits and
#             matches ofdm-hls's existing sample_t.
#   coef_t    ROM constants. BPSK pilots are EXACTLY +1.0 and some
#             training values reach 1.0 -- and ap_fixed<16,1> spans
#             [-1, +1), so 1.0 is NOT representable in it and would
#             quietly become 0.99997. Needs the extra integer bit.
#   scale_t   derived scalars like TRAIN_SCALE = 1/N, which is exactly
#             1.0 when N=1. Same trap.
#
# Section 5 of the plan doc covers why 16 bits rather than fewer: the
# Artix DSP48E1 is 25x18, so narrowing buys no multiplier area.
SAMPLE_W, SAMPLE_I = 16, 1
COEF_W,   COEF_I   = 16, 2
SCALE_W,  SCALE_I  = 18, 2


def _ap_fixed_range(width: int, int_bits: int):
    """Representable range of ap_fixed<W,I>: [-2^(I-1), 2^(I-1) - 2^-(W-I)].
    I includes the sign bit (Vitis convention)."""
    frac = width - int_bits
    lo = -(2.0 ** (int_bits - 1))
    hi = (2.0 ** (int_bits - 1)) - (2.0 ** -frac)
    return lo, hi


class NotRepresentable(ValueError):
    """A generated constant does not fit the type it would be stored in.

    Raised at GENERATE time on purpose. The alternative is ap_fixed
    silently saturating the value and the radio being subtly wrong with
    nothing in any log to say so."""


def _check_representable(name: str, values, width: int, int_bits: int) -> None:
    import numpy as _np
    lo, hi = _ap_fixed_range(width, int_bits)
    arr = _np.asarray(list(values), dtype=_np.float64)
    if arr.size == 0:
        return
    bad = arr[(arr < lo) | (arr > hi)]
    if bad.size:
        raise NotRepresentable(
            f"{name}: {bad.size} of {arr.size} values do not fit "
            f"ap_fixed<{width},{int_bits}> (range [{lo:g}, {hi:g}]). "
            f"Worst offender {bad[_np.argmax(_np.abs(bad))]:.9g}. "
            f"Widen the integer part -- do NOT let it saturate."
        )

# Strategies with a hand-written HLS block today. Anything else is a
# valid spectracuda waveform with no fabric implementation yet -- which
# is a generate-time error, not a synthesis-time one.
IMPLEMENTED = {
    "sync": {"schmidl_cox"},
    "cfo": {"schmidl_cox"},
    "channel_estimator": {"ls"},
    "equalizer": {"mmse"},
    "modem": {"qpsk", "qam16", "qam64"},
    "fec": {"none"},          # inner stage; rs_m8 needs a Reed-Solomon decoder
    "fec1": {"none"},         # outer stage; conv_v27 needs Viterbi
    "crc": {"none", "crc16", "crc32"},
    "interleaver": {"none", "block"},
}


class UnsupportedWaveform(ValueError):
    """A waveform Python can build but fabric cannot express."""


def _strategy_name(obj: Any, fallback: str) -> str:
    """Registry-resolved blocks are instances, not strings. Recover the
    registered name from the class rather than requiring the caller to
    remember what they passed."""
    if isinstance(obj, str):
        return obj
    cls = type(obj).__name__
    return {
        "SchmidlCoxSync": "schmidl_cox",
        "ZadoffChuSync": "zadoff_chu",
        "SchmidlCoxCFO": "schmidl_cox",
        "PilotBasedCFO": "pilot_based",
        "LSChannelEstimator": "ls",
        "MMSEEqualizer": "mmse",
        "ZFEqualizer": "zf",
    }.get(cls, fallback)


def validate(ofdm: Ofdm, partial: bool = False) -> None:
    """Raise `UnsupportedWaveform` for anything fabric cannot build.

    `partial=True` downgrades the refusal to a printed warning. That is
    for INCREMENTAL development only -- emitting the tables for blocks
    that do exist while others are still being written. The gate itself
    stays exact: a partial emit prints what is missing every time, so a
    half-built chain can never be mistaken for a complete one.

    Deliberately checks only what Python does NOT already check -- see
    this module's docstring. Each rule cites why it exists, because a
    rule whose reason is forgotten is a rule nobody dares delete."""
    problems: List[str] = []

    n = ofdm.fft_size
    if n < 4 or (n & (n - 1)) != 0:
        # numpy.fft handles any N, so spectracuda never needed this check
        # (verified: there is no power-of-2 validation anywhere in
        # pipeline/ofdm.py or ofdm/fft.py). The Xilinx FFT LogiCORE and
        # any radix-2 implementation require a power of two.
        problems.append(
            f"fft_size={n} is not a power of 2. Valid in Python (numpy.fft "
            f"handles any N); no FFT core can be built for it."
        )

    if ofdm.cp_len <= 0:
        # Python allows cp_len=0. Without a CP there is no multipath
        # guard and the RX slot arithmetic degenerates; not a fabric
        # limitation as such, but not a waveform worth taping out.
        problems.append(f"cp_len={ofdm.cp_len}; fabric build requires cp_len >= 1")

    checks = [
        ("sync", _strategy_name(ofdm.sync, "?")),
        ("cfo", _strategy_name(ofdm.cfo, "?")),
        ("channel_estimator", _strategy_name(ofdm.channel_estimator, "?")),
        ("equalizer", _strategy_name(ofdm.equalizer, "?")),
        ("modem", getattr(ofdm.modem, "scheme", "?")),
        ("fec", getattr(ofdm, "fec", "none")),
        ("crc", getattr(ofdm, "crc", "none")),
        # The interleaver was in IMPLEMENTED but missing from this list,
        # so a waveform using one was silently accepted. A rule that is
        # declared but never evaluated is worse than no rule -- it reads
        # as coverage that is not there.
        ("interleaver", str(getattr(ofdm, "interleaver", "none"))),
    ]
    for kind, name in checks:
        allowed = IMPLEMENTED[kind]
        if name not in allowed:
            problems.append(
                f"{kind}={name!r} has no HLS block yet "
                f"(implemented: {sorted(allowed)})"
            )

    # fec is the INNER stage, fec1 the OUTER one (Ofdm's class docstring;
    # Packetizer encodes inner-then-outer and decodes outer-then-inner).
    # So the RF-validated config decodes conv_v27 (outer) with Viterbi and
    # THEN rs_m8 (inner) with Reed-Solomon.
    fec1 = getattr(ofdm, "fec1", "none")
    if fec1 not in IMPLEMENTED["fec1"]:
        problems.append(
            f"fec1={fec1!r} (outer stage) has no HLS block yet "
            f"(implemented: {sorted(IMPLEMENTED['fec1'])})"
        )

    if problems:
        msg = ("this waveform is valid in Python but cannot be built for "
               "fabric:\n" + "\n".join(f"  - {p}" for p in problems))
        if partial:
            print("PARTIAL EMIT -- still missing:\n" + msg)
            return
        raise UnsupportedWaveform(msg)


# ---------------------------------------------------------------- emit

def _q(x: float) -> str:
    """Format a real as a C literal for ap_fixed initialisation."""
    return f"{float(x):.9g}"


def _int_table(name: str, values: Iterable[int], count: int) -> str:
    vals = list(int(v) for v in values)
    body = ",".join(str(v) for v in vals)
    wrapped = []
    while body:
        wrapped.append(body[:72])
        body = body[72:]
    # re-wrap on commas so no number is split
    joined = ",".join(str(v) for v in vals)
    lines, cur = [], ""
    for tok in joined.split(","):
        if len(cur) + len(tok) + 1 > 70:
            lines.append(cur)
            cur = tok
        else:
            cur = tok if not cur else cur + "," + tok
    if cur:
        lines.append(cur)
    inner = ",\n    ".join(lines)
    return f"static const int {name}[{count}] = {{\n    {inner}\n}};\n"


def _sample_table(name: str, values: Iterable[float], count: int,
                  ctype: str = "coef_t",
                  width: int = COEF_W, int_bits: int = COEF_I) -> str:
    values = list(values)
    _check_representable(name, values, width, int_bits)
    toks = [_q(v) for v in values]
    lines, cur = [], ""
    for tok in toks:
        if len(cur) + len(tok) + 1 > 68:
            lines.append(cur)
            cur = tok
        else:
            cur = tok if not cur else cur + "," + tok
    if cur:
        lines.append(cur)
    inner = ",\n    ".join(lines)
    return f"static const {ctype} {name}[{count}] = {{\n    {inner}\n}};\n"


_BANNER = """// ============================================================
// {title}
//
// GENERATED by fpga/gen/emit.py from a spectracuda Ofdm object.
// Do not edit -- regenerate with `make headers`.
//
// Source waveform:
{cfg}// ============================================================
"""


def _banner(title: str, ofdm: Ofdm) -> str:
    cfg = (
        f"//   fft_size={ofdm.fft_size} cp_len={ofdm.cp_len} "
        f"n_data={ofdm.grid.n_data} n_pilot={ofdm.grid.n_pilot}\n"
        f"//   modem={getattr(ofdm.modem, 'scheme', '?')} "
        f"fec={getattr(ofdm, 'fec', 'none')} crc={getattr(ofdm, 'crc', 'none')}\n"
        f"//   sync={_strategy_name(ofdm.sync, '?')} "
        f"cfo={_strategy_name(ofdm.cfo, '?')} "
        f"chest={_strategy_name(ofdm.channel_estimator, '?')} "
        f"eq={_strategy_name(ofdm.equalizer, '?')}\n"
        f"//   n_training_symbols={ofdm.n_training_symbols} "
        f"preamble_seed={ofdm.preamble_seed} "
        f"training_seed={ofdm.training_seed}\n"
    )
    return _BANNER.format(title=title, cfg=cfg)


def emit(ofdm: Ofdm, out_dir: str) -> List[str]:
    """Validate, then write the headers. Returns the paths written."""
    validate(ofdm)
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []

    def write(name: str, text: str) -> None:
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            f.write(text)
        written.append(path)

    n_train = int(ofdm.n_training_symbols)

    # ---- ofdm_params.h ----
    params = _banner("ofdm_params.h -- geometry + strategy selection", ofdm)
    params += "#pragma once\n\n#include \"ap_fixed.h\"\n\n"
    params += (
        "// sample_t: signal datapath. coef_t: ROM constants, which reach\n"
        "// EXACTLY +-1.0 (BPSK pilots) and so need the extra integer bit --\n"
        "// ap_fixed<16,1> spans [-1,+1) and would saturate 1.0 silently.\n"
        f"typedef ap_fixed<{SAMPLE_W}, {SAMPLE_I}> sample_t;\n"
        f"typedef ap_fixed<{COEF_W}, {COEF_I}> coef_t;\n"
        f"typedef ap_fixed<{SCALE_W}, {SCALE_I}> scale_t;\n\n"
    )
    params += "// ---- geometry ----\n"
    for k, v in [
        ("FFT_SIZE", ofdm.fft_size),
        ("CP_LEN", ofdm.cp_len),
        ("SLOT_LEN", ofdm.slot_len),
        ("N_DATA", ofdm.grid.n_data),
        ("N_PILOT", ofdm.grid.n_pilot),
        ("N_NULL", int(ofdm.grid.null_indices.size)),
        ("N_TRAINING", n_train),
        ("N_TRAIN_KNOWN", int(np.asarray(ofdm._train_known_indices).size)),
        ("NUM_HEADER_SYMS", ofdm.num_symbols_header),
        ("HEADER_LEN_BITS", int(np.asarray(ofdm._header_scramble_mask).size)),
        ("SC_HALF_L", ofdm.fft_size // 2),
    ]:
        params += f"static const int {k} = {int(v)};\n"
    params += (
        "\n// RX averages the channel estimate over N_TRAINING repetitions\n"
        "// (ofdm.py:948-959). Emitting the reciprocal turns that divide\n"
        "// into a multiply -- free for N=2 (a shift) and not free for N=3.\n"
        f"static const scale_t TRAIN_SCALE = {_q(1.0 / n_train)};\n"
    )
    _check_representable("TRAIN_SCALE", [1.0 / n_train], SCALE_W, SCALE_I)
    params += "\n// ---- strategy selection ----\n"
    params += (
        "// Which hand-written block enters the build. These are different\n"
        "// ALGORITHMS, not different constants -- see plan doc section 2.1.\n"
    )
    for kind, obj, fallback in [
        ("SYNC", ofdm.sync, "?"),
        ("CFO", ofdm.cfo, "?"),
        ("CHEST", ofdm.channel_estimator, "?"),
        ("EQ", ofdm.equalizer, "?"),
    ]:
        params += f"#define {kind}_{_strategy_name(obj, fallback).upper()} 1\n"
    params += f"#define MODEM_{getattr(ofdm.modem, 'scheme', 'none').upper()} 1\n"
    params += f"#define FEC_{str(getattr(ofdm, 'fec', 'none')).upper()} 1\n"
    params += f"#define CRC_{str(getattr(ofdm, 'crc', 'none')).upper()} 1\n"
    write("ofdm_params.h", params)

    # ---- subcarrier_map.h ----
    sc = _banner("subcarrier_map.h -- subcarrier allocation", ofdm)
    sc += "#pragma once\n\n#include \"ofdm_params.h\"\n\n"
    sc += (
        "// Index 0 is DC in unshifted FFT bin order. The guard band sits\n"
        "// around fft_size/2 (the Nyquist edge), NOT at the array ends --\n"
        "// see spectracuda/ofdm/resource_grid.py's module docstring for\n"
        "// the bug that came from getting this backwards.\n"
        "// Precomputed index ROMs rather than runtime is_null/is_pilot\n"
        "// tests: the address is then known each cycle, which is what lets\n"
        "// the mapping loop reach II=1.\n\n"
    )
    sc += _int_table("DATA_IDX", ofdm.grid.data_indices, ofdm.grid.n_data) + "\n"
    sc += _int_table("PILOT_IDX", ofdm.grid.pilot_indices, ofdm.grid.n_pilot) + "\n"
    sc += _int_table("NULL_IDX", ofdm.grid.null_indices,
                     int(ofdm.grid.null_indices.size)) + "\n"
    sc += _int_table("TRAIN_KNOWN_IDX", ofdm._train_known_indices,
                     int(np.asarray(ofdm._train_known_indices).size))
    write("subcarrier_map.h", sc)

    # ---- preamble_rom.h ----
    pre = np.asarray(ofdm._preamble_time).ravel()
    p = _banner("preamble_rom.h -- Schmidl-Cox preamble, time domain", ofdm)
    p += "#pragma once\n\n#include \"ofdm_params.h\"\n\n"
    p += (
        "// One OFDM symbol, NO cyclic prefix, built from energy on every\n"
        "// other subcarrier so its two FFT_SIZE/2 halves are identical.\n"
        "// The RX detector does NOT read this table -- Schmidl-Cox\n"
        "// correlates the signal against a delayed copy of itself and uses\n"
        "// only the halves' structure. It is here for the TRANSMITTER.\n"
        "// (A zadoff_chu sync would need it on RX as a matched-filter\n"
        "// template; that is the difference that removes a ROM from the\n"
        "// receiver, see plan doc section 2.1.)\n\n"
    )
    p += _sample_table("PREAMBLE_I", np.real(pre), pre.size,
                       "sample_t", SAMPLE_W, SAMPLE_I) + "\n"
    p += _sample_table("PREAMBLE_Q", np.imag(pre), pre.size,
                       "sample_t", SAMPLE_W, SAMPLE_I)
    write("preamble_rom.h", p)

    # ---- training_rom.h ----
    tg = np.asarray(ofdm._train_grid_freq).ravel()
    known = np.asarray(ofdm._train_known_indices).ravel()
    tk = tg[known]
    pv = np.asarray(ofdm.pilot_values).ravel()
    t = _banner("training_rom.h -- known training symbol + pilots", ofdm)
    t += "#pragma once\n\n#include \"ofdm_params.h\"\n\n"
    t += (
        "// ONE training symbol regardless of N_TRAINING: the same symbol is\n"
        "// repeated N times (802.11 LTF style, ofdm.py:512), so the ROM does\n"
        "// not scale with N -- only the counters do.\n"
        "// Frequency domain, at TRAIN_KNOWN_IDX only. The LS estimator\n"
        "// divides the received symbol by these to get h_hat.\n\n"
    )
    t += _sample_table("TRAIN_KNOWN_I", np.real(tk), tk.size) + "\n"
    t += _sample_table("TRAIN_KNOWN_Q", np.imag(tk), tk.size) + "\n"
    t += _sample_table("PILOT_I", np.real(pv), pv.size) + "\n"
    t += _sample_table("PILOT_Q", np.imag(pv), pv.size)
    write("training_rom.h", t)

    return written


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))          # fpga/gen
    # C headers for the archived Vitis-HLS flow
    out = os.path.normpath(os.path.join(here, "..", "..", "archive", "hls-cpp", "src", "generated"))
    from .golden import CFG
    ofdm = Ofdm(**CFG)
    for path in emit(ofdm, out):
        print(f"  wrote {os.path.relpath(path, os.path.join(here, '..'))}")


if __name__ == "__main__":
    main()
