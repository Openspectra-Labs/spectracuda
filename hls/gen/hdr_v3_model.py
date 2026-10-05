"""Plain-integer model of the protected (v3) frame header, as the RTL does it.

    info  = 14 bytes (version 3, len, mod, crc|fec0, dmrs|fec1, c2_len, 6 user)
    key   = crc16(info bytes)  (liquid crc: 32-bit reflected register, poly
            0xA001, init 0xFFFFFFFF, complemented, masked to 16), appended
            as 2 bytes big-endian -> 128 bits MSB-first
    code  = conv_v27 (K=7, 0o171 / 0o133, reg=(state<<1)|b) over 128 bits
            + 6 zero tail -> 268 bits
    wire  = code ^ mask268 (numpy default_rng(42))
    slots = 432 BPSK slots over 2 header symbols: wire bits at the select
            positions, filler bits (default_rng(2024)) elsewhere.

`python3 hdr_v3_model.py` checks every step against the live HeaderCodec /
Ofdm and writes the ROMs both RTL blocks read.
"""
import sys, os
import numpy as np
from spectracuda.pipeline import Ofdm
from spectracuda.framing.header import HeaderCodec


def crc16(data):
    c = 0xFFFFFFFF
    for d in data:
        c ^= d
        for _ in range(8):
            c = (c >> 1) ^ 0xA001 if c & 1 else c >> 1
    return (~c) & 0xFFFF


def conv(bits):
    st, out = 0, []
    for b in list(bits) + [0] * 6:
        reg = ((st << 1) | int(b)) & 0x7F
        out += [bin(reg & 0o171).count("1") & 1, bin(reg & 0o133).count("1") & 1]
        st = reg & 63
    return out


def info_bytes(length, mod, crc, fec0, fec1, dmrs_code, c2_len, user6):
    return [3, length >> 8, length & 255, mod, ((crc & 7) << 5) | (fec0 & 31),
            ((dmrs_code & 3) << 5) | (fec1 & 31), c2_len >> 8, c2_len & 255] + list(user6)


def wire_bits(ib, mask):
    k = crc16(ib)
    allb = ib + [k >> 8, k & 255]
    bits = [(B >> (7 - i)) & 1 for B in allb for i in range(8)]
    return [a ^ m for a, m in zip(conv(bits), mask)]


def main():
    o = Ofdm(fft_size=256, cp_len=32, n_data=216, n_pilot=8, modem="qam64", fec1="conv_v27",
             interleaver="block", interleaver_kwargs={"unit_bits": 8}, soft_decision=False,
             dmrs_interval=16)
    codec = o.header_codec
    mask = [int(x) for x in codec._scramble_mask]
    assert len(mask) == 268 and o.num_symbols_header == 2
    rng = np.random.default_rng(5)
    for t in range(200):
        L = int(rng.integers(8, 65535)); mod = ["qpsk", "qam16", "qam64"][t % 3]
        dm = [0, 16, 32, 64][t % 4]; user = bytes(rng.integers(0, 256, 6).tolist())
        want = [int(x) for x in codec.encode_bits(L, mod, "none", user, "none", "conv_v27", dm, 0)]
        got = wire_bits(info_bytes(L, {"qpsk": 1, "qam16": 2, "qam64": 3}[mod], 1, 0, 1,
                                   {0: 0, 16: 1, 32: 2, 64: 3}[dm], 0, user), mask)
        assert got == want, t
    pos = np.asarray(o._header_positions_flat).astype(int).ravel()
    fill_pos = np.asarray(o._header_filler_positions).astype(int).ravel()
    fill = [int(x) for x in o._header_filler_bits]
    assert len(pos) == 268 and len(fill_pos) == 164 and len(fill) == 164
    sel = np.zeros(432, int); sel[pos] = 1
    assert (np.sort(np.concatenate([pos, fill_pos])) == np.arange(432)).all()
    assert (np.diff(pos) > 0).all() and (np.diff(fill_pos) > 0).all()
    print("hdr_v3 model == HeaderCodec on 200 random headers; 268 wire bits over 432 slots")
    if len(sys.argv) > 1:
        out = sys.argv[1]; os.makedirs(out, exist_ok=True)
        open(os.path.join(out, "hdr3_mask.mem"), "w").write("".join(f"{b}\n" for b in mask))
        open(os.path.join(out, "hdr3_sel.mem"), "w").write("".join(f"{b}\n" for b in sel))
        open(os.path.join(out, "hdr3_fill.mem"), "w").write("".join(f"{b}\n" for b in fill))
        print("wrote", out)


if __name__ == "__main__":
    main()
