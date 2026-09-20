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
    """Build header WIRE bits with raw information-byte overrides,
    bypassing encode_bits' validation -- the only way to produce the
    out-of-range values a decoder has to survive.

    The overrides go in BEFORE the header's own CRC+FEC, so the crafted
    header is internally consistent and passes the CRC. That is the
    point: it isolates the c2_len_bytes range check from the header CRC,
    which would otherwise reject any byte-level tampering first."""
    info = bytearray(14)
    info[0] = codec.PROTOCOL_VERSION
    info[1], info[2] = 0x03, 0xE8      # payload_len_bits = 1000
    info[3] = 1                        # qpsk
    info[4] = (6 << 5) | 0             # crc32, fec0=none
    for idx, val in overrides.items():
        info[idx] = val
    info_bits = np.unpackbits(np.frombuffer(bytes(info), dtype=np.uint8))
    wire = np.asarray(codec.packetizer.encode(info_bits[None, :]))[0]
    return wire ^ codec._scramble_mask


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
    # 2 added c2_len_bytes; 3 added the header's own CRC+FEC.
    assert HeaderCodec.PROTOCOL_VERSION == 3
    codec = HeaderCodec()
    d = codec.decode_bits(codec.encode_bits(1000, "qpsk", "none", None))
    assert d["protocol_version"] == 3


def test_user_data_shrank_to_six_bytes():
    assert HeaderCodec.USER_DATA_LEN_BYTES == 6
    codec = HeaderCodec()
    assert len(codec.decode_bits(codec.encode_bits(1000, "qpsk", "none", None))["user_data"]) == 6


def test_header_information_is_still_112_bits():
    """c2_len_bytes came out of user_data, so the INFORMATION content
    did not grow. The wire form is larger because the header carries its
    own CRC+FEC -- that is a separate change, see test_framing_header."""
    codec = HeaderCodec()
    assert HeaderCodec.HEADER_LEN_BITS == 112
    assert len(codec.encode_bits(1000, "qpsk", "none", None, "crc32", "none", 32, 320)) == (
        codec.wire_len_bits
    )


# -- derived sizing (step 2) ------------------------------------------
#
# n_c2_symbols() is THE number the transmitter and receiver must agree
# on. It is not signalled: both sides derive it from c2_len_bytes plus
# the fixed profile. If they ever disagree, both regions decode as
# garbage, so these pin the arithmetic exactly rather than just
# sanity-checking it.

from spectracuda.framing.c2 import (  # noqa: E402
    bits_per_c2_symbol,
    c2_encoded_bits,
    check_c2_len,
    n_c2_symbols,
)

N_DATA = 216


def test_bits_per_symbol_is_the_qpsk_rate_not_the_payload_rate():
    """The C2 region never inherits the header-selected modulation."""
    assert bits_per_c2_symbol(N_DATA) == N_DATA * 2
    assert bits_per_c2_symbol(64) == 128


@pytest.mark.parametrize("bad", [0, -1])
def test_bits_per_symbol_rejects_nonsense_n_data(bad):
    with pytest.raises(ValueError, match="n_data"):
        bits_per_c2_symbol(bad)


def test_zero_length_means_no_region_at_all():
    """Not an empty-but-present region: zero symbols, zero bits, so a
    c2_len_bytes=0 frame is byte-identical to one with no C2 support."""
    assert c2_encoded_bits(0) == 0
    assert n_c2_symbols(0, N_DATA) == 0


@pytest.mark.parametrize("c2_len,expected_bits,expected_symbols", [
    (1, 604, 2),
    (16, 844, 2),
    (64, 1612, 4),
    (72, 1740, 5),
    (128, 2636, 7),
    (256, 5196, 13),
    (320, 6220, 15),
])
def test_sizing_is_pinned_exactly(c2_len, expected_bits, expected_symbols):
    """Exact values, not bounds. A change to the profile or to the
    ceiling convention must break this test loudly -- it would
    otherwise be a silent wire-format change that only shows up as a
    decode failure between two versions."""
    assert c2_encoded_bits(c2_len) == expected_bits
    assert n_c2_symbols(c2_len, N_DATA) == expected_symbols


def test_symbols_are_a_ceiling_not_a_floor():
    """A partial last symbol is padded, never truncated -- truncating
    would silently drop the tail of the C2 payload."""
    for c2_len in range(1, 80):
        encoded = c2_encoded_bits(c2_len)
        n_sym = n_c2_symbols(c2_len, N_DATA)
        assert n_sym * bits_per_c2_symbol(N_DATA) >= encoded
        assert (n_sym - 1) * bits_per_c2_symbol(N_DATA) < encoded


def test_sizing_is_monotonic_in_length():
    """More C2 bytes must never need fewer symbols."""
    prev = 0
    for c2_len in range(0, C2_MAX_BYTES + 1, 8):
        n = n_c2_symbols(c2_len, N_DATA)
        assert n >= prev
        prev = n


def test_sizing_scales_with_the_grid():
    """A narrower grid carries fewer bits per symbol, so the same C2
    payload needs more symbols."""
    assert n_c2_symbols(72, 108) > n_c2_symbols(72, 216)


@pytest.mark.parametrize("bad", [-1, C2_MAX_BYTES + 1, 9000])
def test_sizing_rejects_out_of_range_lengths(bad):
    with pytest.raises(ValueError, match="c2_len_bytes"):
        n_c2_symbols(bad, N_DATA)
    with pytest.raises(ValueError, match="c2_len_bytes"):
        c2_encoded_bits(bad)


def test_header_and_sizing_share_one_validator():
    """A second copy of the range check would drift from this one."""
    with pytest.raises(ValueError, match="c2_len_bytes"):
        check_c2_len(C2_MAX_BYTES + 1)
    codec = HeaderCodec()
    with pytest.raises(ValueError, match="c2_len_bytes"):
        codec.encode_bits(100, "qpsk", "none", None, "crc32", "none", 0, C2_MAX_BYTES + 1)


def test_max_c2_still_leaves_most_of_the_budget_for_payload():
    """The operational constraint: C2 at its cap must not crowd out the
    main payload. 15 of 128 slots leaves 113."""
    assert n_c2_symbols(C2_MAX_BYTES, N_DATA) == 15
    assert 128 - n_c2_symbols(C2_MAX_BYTES, N_DATA) == 113
