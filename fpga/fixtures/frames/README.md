# Stage-boundary reference dumps

Captured once from the **pre-refactor RTL** (branch `HDL_SPECTRA`, after
`bbdd110`: rate-invariant front end + CFO fix) by `capture_golden.sh`, with
the pinned Python reference `ad0a396` (`golden_ref.py`). The refactored
stages (fpga/docs/rx_modular_architecture.md section 11) must reproduce these
**bit-exactly**: they are the proof that the refactor changed structure,
not behaviour.

Every case was:
- bit-exact against Python (decoded bytes), and
- byte-identical in every file below at C=1 and C=10 clocks per sample
  (Rule 0). Only the C=1 copy is stored; `SHA256SUMS` covers it.

Do not regenerate after the refactor starts: `tb/rx/capture_taps.vh` reads
the old hierarchy and will not build against the new one.

## Cases

| Directory | Payload bits | Modem | CFO | EVM |
|---|---|---|---|---|
| `f2000_qam64` | 2000 | 64-QAM | 0 | 0 |
| `f16384_qpsk` | 16384 | QPSK | 0 | 0 |
| `f512_qam16` | 512 | 16-QAM | 0 | 0 |
| `f64_qpsk` | 64 | QPSK | 0 | 0 |
| `f2000_qam64_cfo0p3` | 2000 | 64-QAM | 0.3 | 0 |
| `f16384_qam64_evm0p12` | 16384 | 64-QAM | 0 | 0.12 |
| `f16384_qam16_cfo0p2_evm0p08` | 16384 | 16-QAM | 0.2 | 0.08 |

All: fft 256, cp 32, 216 data + 8 pilots, 1 training symbol, uncoded BPSK
header, conv_v27 rate 1/2, block interleaver unit 8, no RS, no CRC.

## Files

| File | Boundary | Format (one record per line, space separated) |
|---|---|---|
| `stim.hex` | TD input | IQ sample: 8 hex digits, `I[15:0]` then `Q[15:0]` |
| `i1.txt` | **I1** TD → FD | `fseq sym_idx bin stype re im` |
| `i2.txt` | **I2** FD → BD | `fseq sym_idx sc stype n llr0 llr1 llr2 llr3 llr4 llr5` |
| `c1.txt` | **C1** config | `key value` |
| `o1.txt` | BD output | one deinterleaved byte (decimal) per line |
| `meta.json` | — | run parameters + host config inputs (`cfg_encoded_bits`, `cfg_di_units/rows/cols`) |

Field meanings follow the frozen interfaces (design doc section 7,
`rtl/rx/rx_if.vh`):

- `fseq`: frame sequence number (0 for these single-frame cases).
- `sym_idx`: OFDM symbol index in the frame; 0 = the training symbol,
  1 = header, 2.. = body. The preamble is never FFT'd.
- `bin`: natural-order FFT bin 0..255. `re`/`im`: unscaled FFT output
  (25 significant bits, sign-extended).
- `stype`: 0 TRAIN, 1 HEADER, 2 BODY (I1); 1 HEADER, 3 DATA (I2).
- `sc`: data-subcarrier ordinal 0..215 within the symbol.
- `n`: valid LLRs in the group: 1 BPSK (header), 2 QPSK, 4 16-QAM, 6 64-QAM.
- `llrK`: LLR_W = 1, so `0` = bit 0 and `-1` = bit 1; `llr0` is the first
  bit in transmission order; `llrK` for `K >= n` is 0.
- `c1.txt`: `cfg_body_syms` = OFDM symbols after the header (no DMRS/C2
  yet, so = payload symbols); `cfg_mod` uses the header's mod_scheme codes.

I1 has no `frame_end`, by design (TD learns the frame length late); the
dumps contain exactly the symbols the old RTL processed.
