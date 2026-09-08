"""Correctness check for bridge_ldpc: encode a real random payload with
spectracuda's own LDPCCode.encode(), corrupt it through a simulated BSC(p)
channel (spectracuda's own decode() input contract -- hard bits + a
crossover probability, NOT raw soft LLRs, confirmed by reading
fec/ldpc.py's decode() directly), derive the exact same channel LLR array
decode() computes internally from that, feed THAT array to bridge_ldpc,
and check the result against (a) the known original payload and (b)
spectracuda's own decode() output on the identical corrupted codeword --
a true apples-to-apples same-input comparison, not two different channel
models being compared.
"""
from __future__ import annotations

import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from spectracuda.fec.ldpc import LDPCCode

VARIANT = "ldpc_1944_r12"
QC_PATH = "/tmp/spectracuda_aff3ct_qc/ldpc_1944_r12.qc"
BRIDGE_BIN = str(Path(__file__).resolve().parent / "bridge_ldpc")
P = 0.01  # BSC crossover prob, same contract as LDPCCode.decode()'s own p

rng = np.random.default_rng(1234)

code = LDPCCode(VARIANT)
K = code.k
N = code.n
print(f"variant={VARIANT} K={K} N={N}")

info_bits = rng.integers(0, 2, size=(1, K)).astype(np.uint8)
codeword = np.asarray(code.encode(info_bits))  # (1, N) = [message_part | parity]
assert codeword.shape == (1, N), codeword.shape

flip_mask = rng.random(N) < P
noisy_bits = (codeword[0].astype(np.uint8) ^ flip_mask.astype(np.uint8))[None, :]
n_flipped = int(flip_mask.sum())
print(f"BSC(p={P}): flipped {n_flipped}/{N} bits")

# --- spectracuda's own decode, on this exact noisy codeword ---
own_decoded = np.asarray(code.decode(noisy_bits, p=P))
own_info_bits = own_decoded[0, :K]
print(f"spectracuda decode: matches original info bits = {np.array_equal(own_info_bits, info_bits[0])}")

# --- the exact channel LLR array decode() computes internally from the
#     same noisy bits (see fec/ldpc.py decode()'s own channel_llr line) ---
llr_scale = math.log((1 - P) / P)
channel_llr = (1 - 2 * noisy_bits[0].astype(np.float32)) * np.float32(llr_scale)

# --- bridge_ldpc, on the SAME channel_llr array ---
proc = subprocess.Popen(
    [BRIDGE_BIN, QC_PATH],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
llr_bytes = struct.pack(f"<{N}f", *channel_llr.tolist())
out_bytes, err_bytes = proc.communicate(llr_bytes, timeout=10)
print("--- bridge stderr ---")
print(err_bytes.decode(errors="replace"))

bridge_bits = np.frombuffer(out_bytes, dtype="<i4").astype(np.uint8)
print(f"bridge returned {bridge_bits.size} int32 values (expected K={K})")

bridge_matches_original = np.array_equal(bridge_bits, info_bits[0])
bridge_matches_own_decode = np.array_equal(bridge_bits, own_info_bits)
print(f"bridge decode: matches original info bits = {bridge_matches_original}")
print(f"bridge decode: matches spectracuda's own decode output = {bridge_matches_own_decode}")

if not bridge_matches_original:
    diff = np.nonzero(bridge_bits != info_bits[0])[0]
    print(f"MISMATCH vs original at {diff.size}/{K} positions, first few: {diff[:10]}")
    # Check if it's a simple reversal/permutation red flag: same multiset
    # of bits, different order -- would point at an info_bits_pos
    # ordering mismatch between spectracuda's [message|parity] convention
    # and AFF3CT's own transform_H_to_G_identity()-derived positions.
    print(f"popcount original={info_bits[0].sum()} bridge={bridge_bits.sum()}")
    sys.exit(1)

print("PASS: bit-exact match")
