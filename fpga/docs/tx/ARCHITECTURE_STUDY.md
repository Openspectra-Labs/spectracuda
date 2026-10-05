# TX HDL architecture study

Date: 2026-10-04. Repository HEAD inspected: `fb551ae`.
Original architecture study, followed by the implementation described in README.md.
Standalone synthesis/routing was subsequently run; see reports/VIVADO_MEASUREMENT.md.
INTERFACES.md is the current
initial RTL contract; later-format proposals below are not implemented.

## 1. Decision and reference baseline

Proceed with a separate TX implementation area and standalone processing
domains. First reproduce the RX-compatible frozen air format at Python
`ad0a396`; add protected header, interleaver2, and DMRS only in a separately
versioned format phase coordinated with RX. Soft decisions change RX, not the
transmitted coded-bit alphabet by themselves.

Current history includes `6a448df` (Step 4b-1), `5f75b06` (overlapped hard
decoder), and `fb551ae` (soft evaluation). They do not establish TX support.
The inspected `fpga/src` has no TX chain. Existing untracked CDC helpers
are not assumed reviewed or reusable without a separate gate.

Do not use current Python defaults to generate frozen TX vectors: current
`spectracuda/pipeline/ofdm.py` includes protected-header/C2/DMRS/interleaver2
behavior absent from the pinned reference. Record the exact reference commit,
constructor arguments, scheme IDs, seeds, and fixed-point rules in every
vector manifest. Do not modify RX's generated constants to accommodate TX.

## 2. Processing ownership

```text
host frame descriptor + payload bytes
                 |
          TX_BIT_DOMAIN @125 MHz
     host CRC/RS-prepared bytes -> outer block interleave
          -> rate-1/2 convolutional encode
          -> filler -> optional interleaver2
     header serialization/coding; frame geometry
                 |
          asynchronous coded-group stream
                 |
          TX_FREQ_DOMAIN @100 MHz
     constellation mapper -> grid/pilots/training/DMRS
                 |
           complete frequency-bin stream
                 |
          TX_TIME_DOMAIN @100 MHz
     IFFT -> scaling/quantization -> CP insertion
     preamble/frame assembly -> sample pacing
                 |
          AD9361-facing sample interface
```

Filler and interleaver2 remain in BIT before its coded-group output. FD maps
and scatters those ordered coded groups. The ownership is:

| Function | Owner | Reason |
|---|---|---|
| Descriptor validation and encoded/body length | BIT | Coding geometry has one authority |
| Payload CRC and RS encoding | Host in the initial profile | Frozen regression uses CRC none / RS none; preserve deliberate host ownership |
| Outer byte block interleaver | BIT | Operates between RS and convolutional coding |
| Rate-1/2 K=7 encoder and six tail bits | BIT | Coded-bit processing |
| Final DATA-symbol filler and interleaver2 | BIT | Padding precedes inner permutation |
| Header packing, scrambling, later FEC/CRC | BIT | Header coding belongs with bits |
| Mapper, bin placement, pilots, training/DMRS sequences | FD | Frequency-domain content |
| IFFT, CP insertion, preamble waveform, sample pacing | TD | Time-domain waveform generation |
| MAC, retransmission policy, application handling | Host | Not PHY datapath functions |

No separate architectural header stage. Preamble ROM is TD-local; known
training-bin ROM is FD-local. FD generates symbols that carry no coded payload
from immutable frame configuration, rather than requiring dummy coded bits.

TX configuration originates at the host and travels forward with the frame.
Do not force RX's backwards C1 direction onto TX. Apply the same underlying
rule: explicit configuration, explicit streams, and no leaked internal state.
`tx_top` contains instances, wires, CDC boundaries, and top-level I/O.

## 3. Proposed contracts, to freeze before RTL

Host ingress accepts an immutable descriptor before its associated bytes.
Fields include frame tag, raw payload bit length, modulation, supported
CRC/FEC profile, format version, header user data, and later DMRS configuration.
Outer-interleaver settings absent from the air header must be fixed by an
explicit agreed link profile. Reject unsupported combinations before launch.
Initially support byte-aligned payloads and rate 1/2 only.

BIT->FD: valid/ready; up to six hard coded bits in transmission order; `n`,
data-subcarrier ordinal, symbol index/type, frame tag, symbol start/end, and
frame start/end. Header groups are BPSK. Group packing must match the declared
first-bit convention; do not silently inherit RX FIFO's reversed physical
packing. This interface carries bits, not LLRs. Training/DMRS have no bit groups.

FD->TD: valid/ready complex bins, natural bin order 0..255, symbol index/type,
frame tag, frame boundaries. TD can backpressure before launch through bounded
symbol buffering. All configuration transfers must arrive before associated
data, be tagged, and remain immutable through frame retirement.

125->100 MHz crossings use atomic descriptor and stream transfers. No bus of
independently synchronized bits. Use one ordered descriptor stream per consumer
or acknowledged mailbox with sufficient queued contexts. Transfer frame tags
with data and prove no descriptor overwrite while an older frame is resident.
Use an explicit tag width and retirement contract; do not copy modulo-4 tags
without bounding the outstanding frame count.

Samples: declare the DAC-side clock and sample-enable contract before freeze.
For a 100 MHz scheduler and nominal 40 MSPS, 2/3-clock intervals average 2.5
clocks/sample; this models sample strobes, not an assumed physical DAC clock.
If AD9361 requires another clock domain, add a separately verified crossing.
Once a frame launches, radio output must never starve. Admission must guarantee
all remaining symbol deadlines, or prebuffer the complete frame. Record underrun
as a fatal frame error; never silently insert samples.

## 4. Exact frozen-format work

Read pinned sources for each algorithm rather than assuming the current
scheme-name semantics. Initial profiles should cover QPSK/16-QAM/64-QAM and
the exact golden RX cases, including profiles where RS is disabled.

- Header: 112 content bits spread across 216 BPSK data slots; preserve
  serialization, scramble mask (seed 42), selection positions, and filler
  (seed 2024). Known codes are not proof that every code is supported in HDL.
- Convolutional encoder: K=7, polynomials octal 171 then 133, left-shift
  convention and first-output ordering matching the pinned Python; six zero
  tail input bits. No puncturing in this scope.
- Outer block interleaver: row-major write, column-major read, skip virtual
  cells; byte units when the RS profile requires them. Default rows are
  `1+floor(sqrt(n_units))`, columns `ceil(n_units/rows)`.
- RS: current Python identifies GF polynomial 0x11D, primitive element 2,
  RS(255,223), 32 parity bytes, first root 1. Confirm these and multi-codeword
  shortening/segmentation against pinned codec wrappers before implementing.
  A final short block transmits its real bytes plus parity, not 223-byte filler.
- CRC: select and verify each admitted profile's polynomial, initialization,
  reflection, final XOR, and appended byte order from pinned code. Existing
  `crc_gen.v` is a candidate helper, not evidence every profile is supported.
- DATA filler: reproduce the modulation-dependent seeded sequence (7777).
- Grid: FFT 256, CP 32, 216 DATA bins, eight pilots, 32 nulls. Pilot bins are
  1,33,65,97,159,191,223,255; frozen pilots are +1. Verify exact mapper tables,
  normalization, grid ordering, training and preamble seeds in the manifest.
- IFFT: Python uses the 1/256 inverse normalization. Existing RX transform is
  unscaled forward; inverse output cannot be connected directly to the DAC.
  Specify Q formats, rounding, gain, saturation and clipping counters first.

Exact comparisons apply to bits, bytes, metadata and fixed-point reference
outputs. Floating Python IFFT comparison requires declared error/EVM bounds;
do not promise bit identity between floating arithmetic and vendor fixed point.

## 5. Throughput and launch analysis

At 40 MSPS and 100 MHz, one 288-sample OFDM symbol occupies 720 clocks.
64-QAM needs 216*6=1296 coded bits per DATA symbol: 180 Mbit/s.
Rate 1/2 needs 90 Mbit/s encoder input, including coding overhead before it.
A one-input-bit/clock encoder at 125 MHz supplies 250 Mbit/s coded output.
One coded group/clock at FD is 216 cycles per symbol, before grid scheduling.

An RS encoder accepting one byte/clock at 125 MHz has substantial rate
headroom, but parity emission and interleaver memory access must be included.
Do not serialize all 32 GF parity updates across 32 cycles per byte without
a workload budget: that naive architecture is too slow at maximum throughput.
Constant GF multipliers can use XOR networks rather than FPGA DSP multipliers.

The outer interleaver needs whole-block storage; continuous frame preparation
may need two banks. CP insertion needs the final 32 IFFT samples before the
first transmitted useful sample, so at least symbol buffering is mandatory.
Use ping-pong buffers where production and sample drain overlap. Account for
training/header/preamble airtime and shorter packets when measuring rates.

## 6. Shared FFT/IFFT: optional second implementation

Do not share whole RX TD/FD modules: synchronization, equalization and
demapping are RX-specific. Share the transform accelerator behind explicit
request/result ports while keeping independent TX/RX state and wrappers.

Steady simultaneous RX+TX at 40 MSPS requires 256+256=512 transform input
beats per 720 clocks, or 71.1% input-port utilization. Output-port traffic has
the same average. This is a feasibility calculation, not a deadline proof.
Count configuration gaps and actual transform initiation interval; pipeline
latency may overlap input processing. Known preamble/training waveform ROMs
can reduce demand, but savings are not assumed here.

The existing `cp_fft.v` sends forward configuration once and ties output ready
high. Sharing requires per-job direction, output routing/context queue,
complete-symbol staging, and deadline arbitration. Input readiness, arbitrary
TX/RX phase, RX bursts/replay, reset, output capacity and scaling must all be
verified using the actual vendor core. Avoid head-of-line blocking that stalls
RX acquisition or misses DAC deadlines. Add the sharing service only after a
standalone TX passes; do not put global arbitration in `rx_top` or `tx_top`.

## 7. Resource evidence and budget gate

Existing routed hierarchy dated 2026-10-04 05:42:44, with hard overlapped
Viterbi: 14,641 hierarchical LUTs, 14,153 FF, 20 BRAM36+17 BRAM18 (28.5 tiles),
58 DSP. Hierarchical LUT totals differ from adjusted flat utilization; retain
that distinction when making comparisons. This is not the final soft RX.
XC7A50T capacity: 32,600 LUT, 65,200 FF, 75 BRAM36-equivalent tiles, 120 DSP.

The existing FFT instance alone reports 1,977 LUT, 3,674 FF, three BRAM18,
nine DSP. This is an indicative cost for a separate similar transform, not
a measured TX IFFT estimate. Sharing could avoid that duplication but adds
buffers/control. It is not required to establish initial TX correctness.

Proposed working budgets must explicitly cover: payload block banks, symbol
interleaver banks, frequency/time symbol buffers, CDC queues, header/training/
preamble ROMs, CRC/RS logic, mapper, encoder, IFFT and sample output buffering.
No invented total is accepted as a fit result. Preserve placement/timing and
distributed-RAM headroom, not merely aggregate LUT capacity. A whole-frame
soft RX FIFO must be included in the combined system budget.

## 8. Implementation sequence for Claude

1. Freeze the supported frozen profiles, descriptor/stream contracts, Q formats,
   seeds, maximum lengths and reference manifest. Resolve RS/CRC placement:
   target full TX in BIT; allow explicitly selected host-preencoded bring-up.
2. Implement and verify rate-1/2 encoder, profile CRC/RS encoders and outer
   interleaver as internal BIT helpers. Emit and compare stage vectors.
3. Build standalone `tx_bit_domain`: descriptor ownership, independent header
   path, exact lengths/tails/filler, backpressure and frame retirement.
4. Build standalone `tx_freq_domain`: mapping and natural-order grid creation,
   explicit training/header/DATA sequencing, no upstream position reconstruction.
5. Build standalone `tx_time_domain`: dedicated IFFT initially, normalization,
   CP and preamble assembly, buffering and deterministic sample interface.
6. Integrate `tx_top` and CDC. Gate on frozen RX interoperability and sustained
   40 MSPS with realistic host arrival and arbitrary clock phase.
7. Measure TX and combined resource/timing budgets; optionally implement shared
   transform service and prove simultaneous TX/RX deadlines.
8. Move TX/RX format references together: protected header, complete padded
   symbol interleaver2, then DMRS. C2 is a separate scope, not silently enabled.

The proposed scopes are parallel development domains, not permission to modify
RX while TX is under construction. No sub-agents were used for this study.

## 9. Required verification evidence

- Stage dumps: CRC bytes, RS codewords, outer permutation, convolutional bits,
  padded/inner-permuted symbols, mapper values, full grids, IFFT and CP samples.
- Length edges: one byte, exact RS boundaries 222/223/224 bytes at RS input,
  multiple blocks, exact/partial OFDM symbols, maximum frame, invalid/zero
  lengths with explicitly specified handling. CRC expansion changes RS input.
- Contract checks: valid payload stable under stall; metadata/data atomic;
  frame tags not reused while resident; immutable per-frame configuration;
  no broadcast resets; no silent overflow/underflow or unsupported profile.
- Consecutive frames with different length/modulation/profile, interrupted host
  arrivals, random stalls, reset in either clock domain, all launch conditions.
- Independent pinned Python TX comparison plus RTL RX interoperability; loopback
  alone can hide equal TX/RX mistakes. Protected-format interoperability waits
  for matching RX support.
- Numeric limits: impulse/tone IFFT tests, complex sign/bin order, QAM energy,
  deterministic training/preamble, clipping/EVM, CP exactness and DAC amplitude.
- Sustained test includes symbol-buffer deadlines, CDC behavior and real IFFT
  readiness. Shared mode additionally sweeps relative TX/RX alignment.
- Preserve logs/manifests tagged with RTL commit and tool/configuration versions;
  distinguish functional success, throughput success and routed timing closure.

## 10. Gate

GO for Claude to begin contract/reference preparation and TX helper
implementation in this directory after contracts are reviewed. No claim is
made that TX is implemented or that full simultaneous TX/RX already fits.
