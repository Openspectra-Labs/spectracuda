# Initial TX contract

## Host and BIT (125 MHz)

`cfg_valid/cfg_ready` commits a descriptor: `cfg_payload_bits[15:0]`,
`cfg_mod[2:0]` (1 QPSK, 2 QAM16, 3 QAM64), `cfg_user[63:0]` and
`cfg_fseq[1:0]`. Payload must be nonempty, byte-aligned, fit MAX_BYTES
(default 4095), fit 65535 encoded bits and require at most 128 DATA symbols.
The frozen header describes incoming information payload length. This initial
profile supports no payload CRC/RS; sending host-preencoded RS/CRC under these
header fields is not supported without an explicit profile extension.

`in_valid/in_ready`, `in_byte[7:0]`, `in_last` supplies the committed packet.
MSB-first byte bits enter the outer interleaver, then convolutional encoder.
`in_last` must coincide with the descriptor's final byte. One packet is stored
completely before emission. A descriptor cannot arrive during emission.
`bit_done` means the last coded group was accepted, not RF completion.

## T1: BIT → FD (125 → 100 MHz)

Every transaction holds all fields stable until valid AND ready:

| Field | Width | Meaning |
|---|---:|---|
| bits | 6 | Ordered coded bits, earliest bit at bit 0 |
| n | 3 | Valid bits: HEADER=1; DATA=2/4/6; TRAIN=0 |
| sc | 8 | Logical data-carrier index 0..215 |
| sym_idx | 8 | TRAIN=0, HEADER=1, DATA starts at 2 |
| stype | 3 | TRAIN=0, HEADER=1, DATA=3 |
| fseq | 2 | Host-assigned frame identity |
| frame_start/end | 1 each | Start on TRAIN token; end on final DATA sc215 |

TRAIN is one token without coded bits. HEADER and DATA each contain 216
groups. Padding remains BIT-owned after convolutional tail bits. Future inner
interleaver2 must be placed after padding and before these grouped outputs;
it does not require FD changes. The complete 32-bit transaction crosses a
32-entry Gray-pointer asynchronous FIFO. Both clocks require common reset;
local reset release is synchronized. Hardware CDC timing constraints and CDC
signoff are still required.

## T2: FD → TD (100 MHz)

Valid/ready stream: signed `re/im[15:0]` with 14 fractional bits, natural
`bin[7:0]` 0..255, sym_idx/stype/fseq/frame_start/end, and `bad`: held on all
256 bins of a symbol whose T1 input broke the protocol (sc order, metadata
change mid-symbol, illegal n/stype). Every symbol is
complete and ordered. Frame start accompanies TRAIN bin0; frame end accompanies
last DATA bin255. FD owns physical carrier placement, pilots and training.
BIT owns frame geometry and authoritative symbol numbering. FD and TD validate
ordering; they do not infer position from clock count.

FD has two complete grid banks. TD has two frequency banks and three time
banks (two were measured too shallow for the real xfft latency at 40 MSPS /
100 MHz: the vendor-netlist sim underran). An IFFT job reserves its result
bank before launch. TD starts a frame only when initial TRAIN and HEADER time
symbols are ready. Once RF transmission starts, the sample stream cannot
stall. If the next symbol is late (underrun) or arrives `bad`, TD cuts the
frame: one idle sample carrying `out_frame_end=1, out_active=0`, then the
frame's remaining symbols are drained unsent and TX waits for the next frame.
No reset is needed; `st_abort_count` counts cut frames.

## TD and sample transport (100 MHz)

`sample_ce` requests the next sample. At 40 MSPS it averages 0.4 enables per
100 MHz clock. `out_valid` is registered on the requested tick; `out_active`
distinguishes waveform samples from idle zeros. Signed IQ has 15 fractional
bits. Frame start/end and fseq accompany emitted waveform samples. Preamble
is 256 samples without CP; each following OFDM slot is 256+32 samples.
An actual AD9361 clock-domain/packing interface is not included.

Grid quantization rounds Python normalized constellations to Q14. The vendor
IFFT is intended to produce an unscaled inverse sum in 25 significant bits.
TD divides by 128 with nearest rounding, ties away from zero, and saturates to
16 bits: this combines 1/256 inverse normalization and Q14→Q15 conversion.
The vendor core (structural sim netlist, `run_top_xsim.sh`) matches the
oracle within one final LSB on all 15648 test samples.
Preamble ROM is directly quantized Q15.

## Errors and frame lifetime

Status flags are sticky until reset. BIT rejects illegal descriptors and
malformed packet length before anything is radiated. FD/TD flag malformed
stream metadata and the affected frame is cut (see T2), never radiated past
the bad symbol. BIT error belongs to clk_bit; other status
belongs to clk_sample. Supervisory logic must cross status explicitly.
The host must not reuse an fseq until its RF frame_end has retired. Reset must
flush the entire pipeline; partial-domain reset is unsupported.
