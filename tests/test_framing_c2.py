"""C2 region profile constants and the `c2_len_bytes` header field
(critical-region plan step 1).

Nothing transmits a C2 region yet -- that is steps 3 and 4. This covers
the wire format and the fixed profile, which everything after depends
on.

See docs/2026-09-20-critical-c2-region-plan.md.
"""
import numpy as np
import pytest

from spectracuda.framing import HeaderCodec, Packetizer
from spectracuda.framing.c2 import C2_MAX_BYTES, C2_MODEM, C2_PROFILE
from spectracuda.modem import Modem


def craft_header_bits(codec, overrides):
    """Build header bits with raw byte overrides, bypassing encode_bits'
    validation -- the only way to produce the corrupted values a decoder
    has to survive."""
    bits = codec.encode_bits(1000, "qpsk", "none", None, "crc32", "none", 0, 0)
    raw = bytearray(np.packbits(np.asarray(bits, dtype="uint8") ^ codec._scramble_mask).tobytes())
    for idx, val in overrides.items():
        raw[idx] = val
    return np.unpackbits(np.frombuffer(bytes(raw), dtype=np.uint8)) ^ codec._scramble_mask


# -- the fixed profile ------------------------------------------------


def test_profile_is_the_hardware_validated_concatenated_pairing():
    """QPSK + RS(outer-decoded-last) + Viterbi(facing the channel), the
    combination already measured at ~3-4dB coding gain -- not a new,
    untested scheme invented for this feature."""
    assert C2_MODEM == "qpsk"
    assert C2_PROFILE["crc"] == "crc32"
    assert C2_PROFILE["fec"] == "rs_m8"       # fec0, inner
    assert C2_PROFILE["fec1"] == "conv_v27"   # fec1, outer, faces the channel
    assert C2_PROFILE["interleaver"] == "block"


def test_profile_actually_builds_a_packetizer():
    """Catches a typo in a scheme name here, at test time, rather than
    the first time anyone transmits a C2 region."""
    pk = Packetizer(**C2_PROFILE)
    assert pk.encoded_length(C2_MAX_BYTES * 8) > C2_MAX_BYTES * 8  # FEC expands
    Modem(C2_MODEM)


def test_max_bytes_fits_the_slot_budget_with_room_to_spare():
    """320 bytes must not consume the whole payload region -- the main
    payload still has to fit alongside it."""
    pk = Packetizer(**C2_PROFILE)
    bits_per_symbol = 216 * Modem(C2_MODEM).bits_per_symbol
    n_sym = -(-pk.encoded_length(C2_MAX_BYTES * 8) // bits_per_symbol)
    assert n_sym == 15
    assert n_sym < 128 / 4, "C2 at max should stay well under a quarter of the budget"


# -- the header field -------------------------------------------------


@pytest.mark.parametrize("c2_len", [0, 1, 72, 255, 256, 319, C2_MAX_BYTES])
def test_c2_len_round_trips(c2_len):
    codec = HeaderCodec()
    bits = codec.encode_bits(1000, "qpsk", "conv_v27", b"ABCDEF", "crc32", "rs_m8", 32, c2_len)
    assert codec.decode_bits(bits)["c2_len_bytes"] == c2_len


def test_zero_means_no_c2_region():
    codec = HeaderCodec()
    assert codec.decode_bits(codec.encode_bits(1000, "qpsk", "none", None))["c2_len_bytes"] == 0


@pytest.mark.parametrize("c2_len", [0, 72, C2_MAX_BYTES])
@pytest.mark.parametrize("dmrs", [0, 16, 32, 64])
def test_c2_len_is_orthogonal_to_every_other_field(c2_len, dmrs):
    """c2_len_bytes took bytes 6-7 from user_data and sits next to the
    dmrs_period bits in byte 5. A packing error would corrupt one field
    only for certain values of another."""
    codec = HeaderCodec()
    bits = codec.encode_bits(54321, "qam64", "rs_m8", b"\x01\x02\x03\x04\x05\x06",
                             "crc16", "conv_v27", dmrs, c2_len)
    d = codec.decode_bits(bits)
    assert d["c2_len_bytes"] == c2_len
    assert d["dmrs_interval"] == dmrs
    assert d["payload_len_bits"] == 54321
    assert d["mod_scheme"] == "qam64"
    assert d["fec0"] == "rs_m8"
    assert d["fec1"] == "conv_v27"
    assert d["crc"] == "crc16"
    assert d["user_data"] == b"\x01\x02\x03\x04\x05\x06"


# -- the cap is policy, not field width -------------------------------


@pytest.mark.parametrize("bad", [C2_MAX_BYTES + 1, 1000, 65535])
def test_encode_rejects_over_the_cap(bad):
    codec = HeaderCodec()
    with pytest.raises(ValueError, match="c2_len_bytes"):
        codec.encode_bits(1000, "qpsk", "none", None, "crc32", "none", 0, bad)


def test_encode_rejects_negative():
    codec = HeaderCodec()
    with pytest.raises(ValueError, match="c2_len_bytes"):
        codec.encode_bits(1000, "qpsk", "none", None, "crc32", "none", 0, -1)


@pytest.mark.parametrize("corrupt_len", [C2_MAX_BYTES + 1, 9000, 65535])
def test_decode_rejects_over_the_cap_as_corruption(corrupt_len):
    """The field is 16 bits while the cap is 320, deliberately: a value
    the profile does not permit is recognisable as header corruption
    rather than looking like a legal request. If the field had been
    sized to the cap, this class of corruption would be undetectable."""
    codec = HeaderCodec()
    bits = craft_header_bits(codec, {6: (corrupt_len >> 8) & 0xFF, 7: corrupt_len & 0xFF})
    with pytest.raises(ValueError, match="c2_len_bytes"):
        codec.decode_bits(bits)


def test_a_value_at_the_cap_is_accepted_not_rejected():
    """Boundary: 320 is legal, 321 is not."""
    codec = HeaderCodec()
    bits = craft_header_bits(codec, {6: 0x01, 7: 0x40})  # 320
    assert codec.decode_bits(bits)["c2_len_bytes"] == 320


# -- version bump -----------------------------------------------------


def test_protocol_version_bumped_for_the_layout_change():
    assert HeaderCodec.PROTOCOL_VERSION == 2
    codec = HeaderCodec()
    d = codec.decode_bits(codec.encode_bits(1000, "qpsk", "none", None))
    assert d["protocol_version"] == 2


def test_user_data_shrank_to_six_bytes():
    assert HeaderCodec.USER_DATA_LEN_BYTES == 6
    codec = HeaderCodec()
    assert len(codec.decode_bits(codec.encode_bits(1000, "qpsk", "none", None))["user_data"]) == 6


def test_header_is_still_112_bits():
    """The field came out of user_data, so nothing grew."""
    codec = HeaderCodec()
    assert len(codec.encode_bits(1000, "qpsk", "none", None, "crc32", "none", 32, 320)) == 112
    assert HeaderCodec.HEADER_LEN_BITS == 112
