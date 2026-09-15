# Forward error correction

spectracuda's FEC layer presents one bits-in/bits-out interface over three
codes with very different natural units: a streaming convolutional code,
a byte-symbol Reed--Solomon code, and fixed-block QC-LDPC codes. The public
{py:class}`~spectracuda.fec.FEC` wrapper owns the packing, block splitting,
shortening, and exact encoded-length calculation needed to make those codes
interchangeable inside the OFDM pipeline.

This is not a collection of thin wrappers around liquid-dsp. liquid-dsp
delegates its convolutional and Reed--Solomon implementations to an external
`libfec`; spectracuda therefore has its own reference implementations of the
same standard algorithms. LDPC is a deliberate extension beyond liquid-dsp.

## Available schemes

| Scheme | Code | Natural unit | Correction model |
|---|---|---|---|
| `conv_v27` | rate-1/2, K=7, generators 171/133 octal | arbitrary bit stream | hard-decision Viterbi |
| `rs_m8` | shortened RS(255,223) over GF(256) | bytes | up to 16 erroneous symbols per codeword |
| `ldpc_648_r12` ... `ldpc_1944_r56` | IEEE 802.11n QC-LDPC, 12 length/rate combinations | fixed bit blocks | normalized min-sum belief propagation |
| `none` | no coding | arbitrary bits | pass-through |

The LDPC names cover codeword lengths 648, 1296, and 1944 at rates 1/2,
2/3, 3/4, and 5/6.

```python
import numpy as np
from spectracuda.fec import FEC

bits = np.random.default_rng(0).integers(0, 2, size=(4, 1000), dtype="uint8")
fec = FEC("conv_v27", backend="numpy")

encoded = fec.encode(bits)
decoded = fec.decode(encoded)
assert np.array_equal(decoded, bits)
```

Always use `encoded_length(k)` and `decoded_length(n)` when sizing frames.
The relationship is code-specific because convolutional termination and
shortened RS/LDPC blocks have different overhead rules.

## Concatenated coding and interleaving

{py:class}`~spectracuda.framing.Packetizer` supports two FEC stages with an
optional interleaver between them:

```text
payload -> CRC -> fec0 -> interleave -> fec1 -> modem/channel
payload <- CRC <- fec0 <- deinterleave <- fec1 <- modem/channel
```

`fec0` is encoded first and decoded last. `fec1` is closest to the channel.
For the traditional arrangement in which Viterbi handles dispersed channel
errors and Reed--Solomon cleans up its bursty residual errors, configure:

```python
from spectracuda.pipeline import Ofdm

ofdm = Ofdm(
    fft_size=256,
    n_pilot=8,
    n_data=216,
    cp_len=32,
    modem="qpsk",
    crc="crc32",
    fec="rs_m8",             # fec0: decoded last
    fec1="conv_v27",         # fec1: closest to the channel
    interleaver="block",
    interleaver_kwargs={"unit_bits": 8},
)
```

Keeping bytes intact during interleaving matters for an outer RS code.
Bit-granularity interleaving can spread a few errors into more erroneous
RS symbols and reduce, rather than improve, its effectiveness.

## Convolutional code and Viterbi decoder

`conv_v27` uses the standard rate-1/2, constraint-length-7 code with
generators 171 and 133 octal. Six zero tail bits return the encoder to the
known zero state, making traceback unambiguous at both ends.

The portable reference decoder precomputes the 64-state trellis and
vectorizes each add-compare-select step across states and batches. Only the
time axis remains sequential. On a NumPy backend, transparent native
dispatch may select one of several implementations:

1. the spectracuda fast Viterbi kernel;
2. an SSE or NEON implementation when appropriate;
3. portable vendored C;
4. the Python/array reference when native compilation is unavailable.

The dispatch order is based on measurements rather than the presence of an
instruction set alone. An early NEON implementation was correct but slower
than portable C and was not promoted; its replacement was selected only
after measurements on ARM hardware showed a real improvement. See
{doc}`fec-c-lib-acceleration` for the implementation history and benchmark
methodology.

## Reed--Solomon

`rs_m8` is systematic RS(255,223) over GF(256), using primitive polynomial
`0x11d` and 32 parity symbols. Its reference decoder implements syndrome
calculation, Berlekamp--Massey, Chien search, and a small GF(256) Gaussian
elimination to recover error magnitudes.

The direct linear solve deliberately replaces a more convention-sensitive
Forney shortcut. With no more than 16 error locations the matrix is small,
while the formulation is easier to verify. Uncorrectable inputs raise
`ValueError` instead of silently returning an unchecked message.

### Shortened RS blocks

A payload does not need to occupy all 223 message bytes. For a `real_k`-byte
message, the encoder behaves as though `223 - real_k` known leading zeros
were present, but does not transmit them:

```text
[real_k message bytes | 32 parity bytes]
```

This preserves the 16-symbol correction budget and avoids turning small MAC
control messages into 255-byte codewords. Longer inputs are divided into
full blocks followed by at most one shortened block; the extra blocks are
folded into the batch dimension for native processing.

## QC-LDPC

The LDPC implementation expands the published IEEE 802.11n quasi-cyclic base
matrices into a parity-check graph. Encoding is systematic. At construction,
the matrix is split into message and parity portions,

$$
H = [H_m\;|\;H_p], \qquad
p = H_p^{-1} H_m m \pmod 2,
$$

and the derived parity generator is reused for every encode call.

The normalized min-sum decoder stores messages in flat per-edge arrays.
Precomputed index tables provide check-node and variable-node views using
gathers only. This avoids backend-dependent scatter-add behavior and makes
the fixed graph operations batch-parallel on both NumPy and CuPy.

The array implementation operates through NumPy or CuPy. An opt-in persistent
AFF3CT subprocess provides a faster CPU decoder for installations that build
that dependency explicitly. See {doc}`ldpc` for backend setup, measurements,
and design details. A Numba min-sum decoder with syndrome-based early
termination is under active development, but is not part of the committed
release described by this page yet.

LDPC shortening uses the same implicit-leading-zero concept as RS. Because
belief propagation consumes per-bit likelihoods, the reinserted positions
are assigned a very large positive LLR: they are known zeros, not noisy bits
observed from the channel.

## Native acceleration contract

Native acceleration does not change the FEC API or wire format:

- native C is attempted only for `backend="numpy"`;
- CuPy inputs are not copied to the CPU merely to use a native CPU codec;
- shared libraries are cached under a source-derived hash;
- cache installation uses an atomic rename to tolerate concurrent startup;
- instruction-specific libraries are runtime-gated before loading;
- failure of an automatic optimization falls back to the reference path.

An explicitly requested backend is different. For example,
`LDPCCode(..., decoder="aff3ct")` fails clearly when AFF3CT is unavailable
instead of silently changing the requested decoder.

## Current limitations

The most important limitation is that the common modem/FEC boundary carries
hard bits rather than soft log-likelihood ratios. LDPC therefore synthesizes
uniform-magnitude LLRs from an assumed binary crossover probability `p`:

$$
L = (1 - 2b)\log\left(\frac{1-p}{p}\right).
$$

This is a valid binary-symmetric-channel model, but it discards confidence
information available at the demapper and leaves coding gain on the table.
The same interface also prevents soft-decision Viterbi and HARQ combining.

Other explicit boundaries are:

- only the rate-1/2 convolutional code is implemented; there is no puncturing;
- RS has no erasure-input interface;
- batch decode failure currently raises for the call rather than returning a
  per-codeword status vector;
- Viterbi and RS need additional interoperability vectors from an independent
  implementation such as GNU Radio or libfec;
- CuPy is an array backend, not yet a custom, streamed CUDA FEC pipeline;
- AFF3CT is optional and requires a separate local build.

The reference implementations and optimized implementations are kept side by
side intentionally: the former provide a portable correctness oracle, while
the latter address the sequential loops that dominate real frame latency.

## Further reading

- {doc}`fec-c-lib-acceleration` -- native Viterbi/RS and AFF3CT design
- {doc}`ldpc` -- detailed LDPC design record and backend measurements
- {doc}`book/ch06_fec_ldpc` -- tutorial treatment and measured coding curves
- {doc}`architecture` -- how FEC fits into the full PHY
- {doc}`todo` -- validation and performance work still open
