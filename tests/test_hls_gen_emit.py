"""The HLS config generator (hls/gen/emit.py).

Two things are being pinned here. First, that the generator REFUSES
waveforms fabric cannot build, at generate time -- the whole point of
treating `Ofdm(...)` as a specification rather than a parameter bag is
that errors surface now, not three hours into synthesis. Second, that a
constant which does not fit its ap_fixed type is an error rather than a
silent saturation; that bug was real (BPSK pilots are exactly +1.0 and
ap_fixed<16,1> spans [-1,+1)), and nothing would have reported it.
"""
import os

import numpy as np
import pytest

from hls.gen.emit import (
    NotRepresentable,
    UnsupportedWaveform,
    _ap_fixed_range,
    _check_representable,
    emit,
    validate,
)
from spectracuda.pipeline import Ofdm

BASE = dict(fft_size=256, n_pilot=6, n_data=200, cp_len=32,
            modem="qpsk", fec="conv_v27", crc="crc32",
            sync="schmidl_cox", cfo="schmidl_cox",
            channel_estimator="ls", equalizer="mmse")


def test_ap_fixed_range_matches_vitis_convention():
    # I includes the sign bit, so <16,1> is [-1, +1), NOT [-2, +2).
    lo, hi = _ap_fixed_range(16, 1)
    assert lo == pytest.approx(-1.0)
    assert hi == pytest.approx(1.0 - 2.0 ** -15)
    lo2, hi2 = _ap_fixed_range(16, 2)
    assert lo2 == pytest.approx(-2.0)
    assert hi2 == pytest.approx(2.0 - 2.0 ** -14)


def test_exactly_one_does_not_fit_sample_t_but_does_fit_coef_t():
    """The actual bug: BPSK pilot value +1.0."""
    with pytest.raises(NotRepresentable, match="do not fit"):
        _check_representable("PILOT_I", [1.0], 16, 1)
    _check_representable("PILOT_I", [1.0], 16, 2)   # must not raise


def test_representability_error_names_the_worst_offender():
    with pytest.raises(NotRepresentable, match="1.75"):
        _check_representable("T", [0.1, -1.75, 0.2], 16, 1)


def test_valid_waveform_passes_validation():
    validate(Ofdm(**BASE))


def test_non_power_of_two_fft_is_rejected():
    """Valid in Python -- numpy.fft handles any N, and spectracuda has no
    power-of-2 check anywhere. No FFT core can be built for it."""
    cfg = dict(BASE, fft_size=200, n_data=150, n_pilot=6)
    ofdm = Ofdm(**cfg)          # Python is happy to build this
    with pytest.raises(UnsupportedWaveform, match="not a power of 2"):
        validate(ofdm)


def test_unimplemented_strategy_is_rejected():
    ofdm = Ofdm(**dict(BASE, equalizer="zf"))
    with pytest.raises(UnsupportedWaveform, match="equalizer='zf'"):
        validate(ofdm)


def test_outer_fec_is_rejected_while_rs_is_not_in_fabric():
    ofdm = Ofdm(**dict(BASE, fec1="rs_m8"))
    with pytest.raises(UnsupportedWaveform, match="fec1"):
        validate(ofdm)


def test_all_problems_are_reported_together_not_one_at_a_time():
    """A generate-time check that surfaces one error per run wastes the
    round trip it exists to save."""
    ofdm = Ofdm(**dict(BASE, fft_size=200, n_data=150, equalizer="zf"))
    with pytest.raises(UnsupportedWaveform) as exc:
        validate(ofdm)
    msg = str(exc.value)
    assert "power of 2" in msg and "equalizer" in msg


def test_emit_writes_the_four_headers(tmp_path):
    paths = emit(Ofdm(**BASE), str(tmp_path))
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ["ofdm_params.h", "preamble_rom.h",
                     "subcarrier_map.h", "training_rom.h"]


def test_emitted_geometry_matches_the_object(tmp_path):
    ofdm = Ofdm(**BASE)
    emit(ofdm, str(tmp_path))
    txt = (tmp_path / "ofdm_params.h").read_text()
    assert f"static const int FFT_SIZE = {ofdm.fft_size};" in txt
    assert f"static const int SLOT_LEN = {ofdm.slot_len};" in txt
    assert f"static const int N_DATA = {ofdm.grid.n_data};" in txt
    assert f"static const int SC_HALF_L = {ofdm.fft_size // 2};" in txt


def test_strategy_selection_macros_are_emitted(tmp_path):
    emit(Ofdm(**BASE), str(tmp_path))
    txt = (tmp_path / "ofdm_params.h").read_text()
    for macro in ("SYNC_SCHMIDL_COX", "CFO_SCHMIDL_COX", "CHEST_LS",
                  "EQ_MMSE", "MODEM_QPSK", "FEC_CONV_V27", "CRC_CRC32"):
        assert f"#define {macro} 1" in txt


def test_subcarrier_tables_match_the_grid_exactly(tmp_path):
    """The table is the allocation -- a transcription slip here puts every
    symbol on the wrong subcarriers."""
    ofdm = Ofdm(**BASE)
    emit(ofdm, str(tmp_path))
    txt = (tmp_path / "subcarrier_map.h").read_text()
    body = txt.split("DATA_IDX[200] = {", 1)[1].split("}", 1)[0]
    got = [int(t) for t in body.replace("\n", "").split(",") if t.strip()]
    assert got == list(np.asarray(ofdm.grid.data_indices))


def test_pilot_rom_holds_the_real_pilot_values(tmp_path):
    ofdm = Ofdm(**BASE)
    emit(ofdm, str(tmp_path))
    txt = (tmp_path / "training_rom.h").read_text()
    assert "static const coef_t PILOT_I[6]" in txt   # coef_t, not sample_t
    body = txt.split("PILOT_I[6] = {", 1)[1].split("}", 1)[0]
    got = [float(t) for t in body.replace("\n", "").split(",") if t.strip()]
    assert got == pytest.approx(list(np.real(np.asarray(ofdm.pilot_values))))


def test_training_rom_holds_one_symbol_regardless_of_n_training(tmp_path):
    """The same symbol is repeated N times (802.11 LTF style), so the ROM
    must not scale with N -- only the counters do."""
    a = tmp_path / "n1"
    b = tmp_path / "n4"
    emit(Ofdm(**dict(BASE, n_training_symbols=1)), str(a))
    emit(Ofdm(**dict(BASE, n_training_symbols=4)), str(b))
    ta = (a / "training_rom.h").read_text().split("TRAIN_KNOWN_I")[1]
    tb = (b / "training_rom.h").read_text().split("TRAIN_KNOWN_I")[1]
    assert ta == tb
    assert "static const int N_TRAINING = 4;" in (b / "ofdm_params.h").read_text()


def test_train_scale_is_the_reciprocal_so_averaging_is_a_multiply(tmp_path):
    emit(Ofdm(**dict(BASE, n_training_symbols=4)), str(tmp_path))
    txt = (tmp_path / "ofdm_params.h").read_text()
    assert "static const scale_t TRAIN_SCALE = 0.25;" in txt


def test_train_scale_of_one_needs_the_wider_type(tmp_path):
    """N=1 gives TRAIN_SCALE = 1.0, which ap_fixed<16,1> cannot hold."""
    emit(Ofdm(**dict(BASE, n_training_symbols=1)), str(tmp_path))
    txt = (tmp_path / "ofdm_params.h").read_text()
    assert "static const scale_t TRAIN_SCALE = 1;" in txt
    assert "typedef ap_fixed<18, 2> scale_t;" in txt


def test_generation_is_deterministic(tmp_path):
    """Same waveform in, byte-identical headers out -- otherwise the
    build is not reproducible and diffs are noise."""
    a, b = tmp_path / "a", tmp_path / "b"
    emit(Ofdm(**BASE), str(a))
    emit(Ofdm(**BASE), str(b))
    for name in ("ofdm_params.h", "subcarrier_map.h",
                 "preamble_rom.h", "training_rom.h"):
        assert (a / name).read_text() == (b / name).read_text()
