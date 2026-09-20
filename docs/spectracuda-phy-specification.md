# SpectraCUDA Flexible-Link Modem — PHY specification and validation record

**Document status:** Draft 0.1  
**Date:** 2026-09-20  
**Implementation:** `spectracuda`  
**Purpose:** implementable waveform description, conformance guide, and
measured-validation record

## 1. Scope

This document specifies the over-the-air physical-layer behavior implemented
by SpectraCUDA. It is written so that an independent transmitter or receiver
can be built without depending on SpectraCUDA's Python object model.

The modem is a configurable burst-mode OFDM link. It provides frame detection,
coarse carrier-frequency-offset correction, channel estimation, frequency-
domain equalization, payload modulation, optional CRC and FEC, arbitrary-chunk
streaming reception, and a header that tells a receiver how to decode the
payload. A MAC supporting transparent, unacknowledged, and acknowledged modes
exists above the PHY, but its PDU format is outside the normative PHY sections
of this document.

This is not the FlexLink waveform and does not claim wire compatibility with
FlexLink, LTE, Wi-Fi, or liquid-dsp. Its document structure is intentionally
similar to an implementable modem specification: waveform rules are stated
first, followed by receiver behavior, conformance tests, measurements, and
known limitations. Design ideas taken from other systems are identified as
precedent rather than presented as validation.

The executable reference is the repository source and test suite. If this
draft disagrees with released code, the discrepancy is a specification defect
until one side is deliberately changed and the protocol version is considered.

### 1.1 Requirement language

The words **shall**, **shall not**, **should**, and **may** describe required,
prohibited, recommended, and optional behavior respectively.

Status labels have the following meanings:

- **Implemented:** present in the reference implementation and covered by
  automated tests.
- **Measured:** exercised on two physically separate Raspberry Pi 5 +
  ADALM-PLUTO nodes over RF.
- **In progress:** under development on a feature branch; not part of the
  stable waveform yet.
- **Planned:** design intent only. It shall not be used to claim compatibility.

## 2. System model

The PHY operates on complex baseband IQ samples. A frame is a finite burst:

```text
      acquisition             channel       robust       data region
         signal              reference      control
┌──────────────────┬────────────────────┬──────────────┬────────────────────┐
│ preamble, no CP  │ training symbol(s) │ header OFDM │ payload OFDM slots │
└──────────────────┴────────────────────┴──────────────┴────────────────────┘
```

All OFDM symbols after the preamble use an FFT of size `N` and a cyclic prefix
of `Ncp` samples. Their physical slot length is therefore

```text
Nslot = N + Ncp samples.
```

The preamble deliberately has no cyclic prefix. Adding one produces a broad
Schmidl-Cox timing plateau and makes the detected boundary ambiguous.

SpectraCUDA uses complex64/float32 processing. The optional `iq_dtype` setting
quantizes samples at the transmitter output and receiver input boundaries; it
does not change the internal FFT arithmetic to half precision.

## 3. OFDM numerology and resource grid

### 3.1 Configurable parameters

The two ends shall agree out of band on the following waveform parameters:

| Parameter | Meaning |
|---|---|
| `fft_size` | IFFT/FFT size, `N` |
| `cp_len` | cyclic-prefix length, `Ncp` |
| `n_data` | number of data-bearing subcarriers |
| `n_pilot` | number of pilot subcarriers |
| `dc_null` | whether the DC carrier is unused |
| `preamble_seed` | deterministic acquisition-sequence seed |
| `training_seed` | deterministic training-symbol seed |
| `n_training_symbols` | repeated training-symbol count, at least one |
| `interleaver` and parameters | payload interleaver configuration |

These values are not fully described by the frame header. A receiver using a
different grid, preamble, training sequence, or interleaver is not expected to
decode the frame.

The commonly exercised profile is:

| Parameter | Value |
|---|---:|
| FFT size | 256 |
| cyclic prefix | 32 samples |
| slot length | 288 samples |
| data subcarriers | 216 |
| pilot subcarriers | 8 |
| DC carrier | nulled |

At 4 MSps this profile has a 72 microsecond OFDM slot. At 10 MSps it has a
28.8 microsecond slot. Sample rate and RF center frequency belong to the radio
configuration rather than the baseband wire header.

### 3.2 Resource allocation

The resource-grid implementation assigns non-DC active subcarriers between
pilots and data and leaves the remainder as guards/nulls. Transmitter and
receiver shall use identical index sets. The exact index-generation algorithm
in `spectracuda/ofdm/resource_grid.py` is currently the normative reference;
explicit interoperable index tables will be added before this document is
declared stable.

Payload pilots carry known complex values in every payload data symbol. They
are used for continuous common-phase-error correction in the current receiver.
The full known training symbol estimates the frequency-selective channel.

## 4. Frame signals

### 4.1 Preamble

The default preamble is generated by the Schmidl-Cox synchronizer. It contains
two repeated halves so that a receiver can detect the frame and estimate coarse
normalized CFO. The default receiver declares a frame only when the detection
metric is at least 0.3, unless configured otherwise.

The value 0.3 is empirical rather than universal. In the recorded calibration,
pure noise scored no more than 0.21 for Schmidl-Cox and 0.13 for Zadoff-Chu
across FFT sizes 64 and 256 in 30 trials each; genuine embedded signals at
10 dB SNR scored at least 0.74 and 0.89 respectively. Implementations using a
different numerology or detector should recalibrate the threshold.

The preamble shall not have a cyclic prefix. The first following training
symbol shall begin at the detected OFDM boundary after CFO correction.

### 4.2 Training symbols

The training grid contains deterministic QPSK values on every data subcarrier
and the known pilot values on every pilot subcarrier. QPSK values are selected
using `training_seed` (default 999) from

```text
{(+1+j), (+1-j), (-1+j), (-1-j)} / sqrt(2).
```

When more than one training symbol is configured, the identical symbol is
repeated and the receiver may average estimates to reduce noise. Training
symbols use the normal cyclic prefix.

### 4.3 Header symbols

The header is always BPSK-modulated, independent of payload modulation.

The 112 header information bits carry their own CRC and FEC before reaching
the modem:

```text
112 information bits
  -> CRC-16 append          -> 128 bits
  -> rate-1/2 K=7 conv_v27  -> 268 wire bits
  -> XOR scramble (seed 42, mask sized to the wire length)
  -> BPSK
  -> frequency spreading

n_header_symbols = ceil(268 / n_data).
```

At `n_data = 216` that is two header symbols, one more than the unprotected
header needed. CRC-16 rather than CRC-8 because both land on the same two
symbols once the convolutional code has expanded them (252 versus 268 bits),
so the stronger check costs nothing.

Real header bits are distributed across the available data positions for
frequency diversity. Unused positions receive deterministic random filler.

Scrambling and nonconstant filler are waveform requirements, not cosmetic
choices. An earlier mostly constant header produced a measured time-domain
peak-to-average power ratio of approximately 181, versus approximately 5 for
ordinary payload content, and failed specifically under real multipath.

A receiver **shall** verify the header CRC before acting on any field. A
header that fails it **shall** be rejected rather than interpreted: a
corrupted but syntactically plausible header would otherwise select the wrong
payload interpretation while the receiver believed it had understood the
frame.

Measured, injecting increasing noise into the header symbols only, 40 frames
per level: no corruption level produced a silently-wrong header. Frames either
decoded correctly or were rejected outright. Under full-frame AWGN the header
and the payload now succeed at the same SNR (48% at 6 dB, 93% at 8 dB, 100% at
10 dB with an `rs_m8 + conv_v27` payload), so the header is no longer the
weakest link in the frame.

The cost is one additional header OFDM symbol per frame: 28.8 us at 10 MSps,
which also places the payload 288 samples further from the training symbol and
so accumulates marginally more residual CFO before payload demodulation
begins.

### 4.4 Payload region

Payload bits are CRC-appended, FEC-encoded, interleaved when configured,
modulated, and packed into `n_data` resource elements per OFDM data symbol.
The final partial symbol is filled with deterministic non-information bits;
the receiver discards them using the original payload length from the header.

The payload region is limited to 128 physical OFDM slots. This is a
coherence-time and bounded-receiver-work limit rather than a field-width
limit.

## 5. PHY header wire format

The header contains 14 bytes (112 bits), transmitted most-significant bit
first before scrambling:

| Byte | Bits | Field |
|---:|---|---|
| 0 | 7:0 | protocol version |
| 1-2 | 15:0 | uncoded payload length in bits, big-endian |
| 3 | 7:0 | payload modulation code |
| 4 | 7:5 | CRC code |
| 4 | 4:0 | inner FEC (`fec0`) code |
| 5 | 4:0 | outer FEC (`fec1`) code |
| 5 | 7:5 | reserved in protocol version 1 |
| 6-13 | 63:0 | user-defined bytes |

The protocol version is currently 1. Receivers presently report but do not
reject a mismatched version; interoperability software should treat this as a
known gap rather than relying on permissive behavior.

### 5.1 Modulation codes

| Code | Modulation | Bits/symbol |
|---:|---|---:|
| 0 | BPSK | 1 |
| 1 | QPSK | 2 |
| 2 | 16-QAM | 4 |
| 3 | 64-QAM | 6 |
| 4 | 256-QAM | 8 |

### 5.2 CRC codes

| Code | Scheme |
|---:|---|
| 1 | none |
| 2 | checksum |
| 3 | CRC-8 |
| 4 | CRC-16 |
| 5 | CRC-24 |
| 6 | CRC-32 |

CRC code 0 and unassigned values are invalid.

### 5.3 FEC codes

| Code | Scheme |
|---:|---|
| 0 | none |
| 1 | rate-1/2, constraint-length-7 convolutional code (`conv_v27`) |
| 2 | RS(255,223) over GF(256) (`rs_m8`) |
| 3-14 | the twelve IEEE 802.11n QC-LDPC variants, in repository-defined order |
| 15-31 | reserved |

The precise LDPC code-to-name table is generated from the sorted variant names
in `spectracuda/fec/ldpc_tables.py`. A future stable protocol revision should
publish that table explicitly instead of deriving wire assignments from source
ordering.

Encoding order is CRC, inner FEC (`fec0`), then outer FEC (`fec1`). Decoding
reverses that order. The header's payload length always describes the original
information bits before CRC and FEC.

## 6. Periodic channel refresh (DMRS)

**Status: implemented through header signaling, MAC capacity, and streaming
reception on `feat/dmrs-periodic-channel-refresh`; hardware validation remains
in progress. This is not yet a stable protocol-version-1 requirement.**

Long frames can outlive the estimate obtained from the initial training
symbol. The extension inserts a full known training waveform into the payload
region and re-runs the existing channel estimator. It does not define a new
reference sequence or a partial frequency-domain DMRS grid.

Supported intervals count data symbols:

| Header code | Interval | Meaning |
|---:|---:|---|
| 0 | 0 | disabled |
| 1 | 16 | refresh after each complete run of 16 data symbols |
| 2 | 32 | refresh after each complete run of 32 data symbols |
| 3 | 64 | refresh after each complete run of 64 data symbols |

The implemented feature-branch code occupies byte 5 bits 6:5 and leaves bit 7
reserved. The number of inserted reference symbols is

```text
n_dmrs = 0                              when interval = 0
n_dmrs = floor((n_data_symbols - 1)/I) otherwise.
```

The subtraction suppresses a trailing reference symbol that would refresh a
channel estimate used by no data. Data and DMRS together shall not exceed 128
payload-region slots. The maximum data counts are therefore 128, 127, 125, and
121 for OFF, 64, 32, and 16 respectively.

Segment zero uses the initial training estimate. Every later data segment uses
the estimate from its immediately preceding DMRS. Per-symbol pilots continue
to correct common phase error.

Old receivers already mask byte 5 to five FEC bits and can parse the header,
but they cannot decode a DMRS-bearing payload correctly because they do not
skip the inserted slots. This extension is header-parse-compatible, not
end-to-end backward compatible.

The receiver resolves this interval from the decoded header rather than from
its own transmit configuration. Feature-branch conformance tests verify that a
receiver constructed with DMRS disabled decodes independently generated frames
at intervals OFF, 16, 32, and 64, including the time-varying two-ray case.

MAC capacity and streaming reception account for the inserted physical slots.
At the tested QPSK/CRC-32 profile, the maximum segment budgets are:

| Interval | MAC segment | Data | DMRS | Total slots |
|---:|---:|---:|---:|---:|
| OFF | 55,232 bits | 128 | 0 | 128 |
| 64 | 54,800 bits | 127 | 1 | 128 |
| 32 | 53,936 bits | 125 | 3 | 128 |
| 16 | 52,208 bits | 121 | 7 | 128 |

Streaming conformance includes differently populated back-to-back frames, so
an incorrect first-frame eviction boundary cannot pass merely because the
first frame itself decoded.

## 7. Receiver processing

A conforming receiver performs the following logical sequence:

1. Search for the configured preamble and apply the detection threshold.
2. Estimate and remove coarse CFO.
3. remove cyclic prefixes and FFT the training symbol(s).
4. Estimate `H[k]` from known active training subcarriers.
5. Demodulate and decode the BPSK header.
6. Resolve payload modulation, CRC, and FEC from the decoded header rather
   than assuming the receiver's own transmit configuration.
7. Compute the encoded payload length and required physical slot count; reject
   impossible or over-limit frames before extracting them.
8. Equalize payload subcarriers, correct common phase from pilots, demodulate,
   deinterleave, FEC-decode, and validate CRC.
9. Report payload bits and measurements including synchronization metric, CFO,
   channel estimate, EVM, RSSI, and CRC status.

The batch API accepts an already bounded IQ array. The streaming API accepts
arbitrary, unaligned chunks and maintains `SEEKING`, `WAITING_HEADER`, and
`WAITING_PAYLOAD` state while reusing the same header and payload decoder.

## 8. Conformance and reproducible test data

Automated test results are evidence for the implementation revision under
test, not a guarantee for every parameter combination.

### 8.1 Frame acquisition and recovery

The 256-subcarrier reference scenario is run with deterministic seeds 0, 1,
and 2 at 25 dB SNR and normalized CFO 0.15. Required results are:

- detected timing within five samples of the inserted start;
- CFO estimate within 0.01 of the true normalized CFO;
- uncoded payload BER below 0.02.

At 5 dB SNR with seed 0, timing remains within five samples, CFO error remains
within 0.05, and BER remains below 0.2. These bounds intentionally allow the
small residual timing error created by multipath delay-spread smearing.

### 8.2 Detector threshold calibration

For Schmidl-Cox and Zadoff-Chu strategies, FFT sizes 64 and 256 were tested
over 30 pure-noise trials and embedded signals at 10 dB SNR:

| Detector | Maximum noise metric | Minimum signal metric |
|---|---:|---:|
| Schmidl-Cox | 0.21 | 0.74 |
| Zadoff-Chu | 0.13 | 0.89 |

The default threshold of 0.3 lies between the recorded populations.

### 8.3 DMRS arithmetic conformance

The in-progress DMRS extension fixes these test vectors:

| Data symbols | Interval | DMRS | Total slots | DMRS slot indices |
|---:|---:|---:|---:|---|
| 31 | 32 | 0 | 31 | `[]` |
| 31 | 16 | 1 | 32 | `[16]` |
| 53 | 16 | 3 | 56 | `[16, 33, 50]` |
| 125 | 32 | 3 | 128 | `[32, 65, 98]` |
| 127 | 64 | 1 | 128 | `[64]` |

For 125 data symbols at interval 32, channel-estimate segment lengths are
`[32, 32, 32, 29]`. The combined framing, transmitter, and receiver DMRS
suite passes 172 focused tests.

### 8.4 Time-varying-channel DMRS validation

Receiver validation uses a deterministic two-ray model whose delayed tap
rotates at 100 Hz:

```text
rx[n] = tx[n] + 0.5 exp(j 2 pi 100 n/Fs) tx[n-1].
```

This produces a genuinely time-varying, frequency-selective `H[k,t]`; it
cannot be removed by the existing common-phase correction alone. The test
uses `Fs=10 MSps`, 30 dB SNR, QPSK, CRC-32, and 125 payload data symbols.
Measured mean EVM per channel-estimate segment was:

| Segment | DMRS disabled | DMRS interval 32 |
|---:|---:|---:|
| 0 | 0.1662 | 0.1641 |
| 1 | 0.4102 | 0.1790 |
| 2 | 0.5619 | 0.1848 |
| 3 | 0.6978 | 0.1773 |

Without refresh, EVM increased by approximately 4.2 times from the first to
the final segment and the frame failed CRC. With interval-32 refresh, EVM
remained approximately flat and the frame passed CRC. Segment zero remained
essentially unchanged, as required: both cases use the original training
estimate there. Additional tests found progressively lower aggregate EVM in
the order OFF, 64, 32, 16 and verified batch-major estimate alignment using
different payloads in each batch item.

The comparisons use the same deterministic channel model, configuration, and
random seed. They are matched-condition transmissions rather than the exact
same received IQ samples: inserted DMRS changes subsequent symbol timing and
the ON/OFF arrays have different lengths.

## 9. Over-the-air validation record

The following results were measured using two physically separate Raspberry
Pi 5 + ADALM-PLUTO nodes. Unless noted, the characterization used a 4 MSps
sample rate. Capture rate means frames decoded divided by frames transmitted;
it includes frames lost before payload decoding.

### 9.1 Decoder integrity

Across the recorded session, all three tested payload modulations—QPSK,
16-QAM, and 64-QAM—produced zero observed CRC failures among frames that
reached decode. The remaining losses were traced primarily to frame detection
and capture, not to the FEC/equalizer/demapper chain.

This statement is limited to runs where CRC-valid counts were actually
recorded. Some frequency-table entries reported decoded counts only and are
identified as such in the detailed session log.

### 9.2 2.4 GHz modulation sweep

Each modulation point used 500 transmitted packets:

| Frequency | QPSK capture | 16-QAM capture | 64-QAM capture |
|---|---:|---:|---:|
| 2.402 GHz | 91.4-92% | 90% (450/450 decoded CRC-valid) | 82.6% (413/413 decoded CRC-valid) |
| 2.483 GHz | 94.8-97% | 89.8% (449/449 decoded CRC-valid) | 66% (330/500 decoded; CRC count unconfirmed) |

The result does not support a single environment-independent “best” center
frequency. The frequency with better QPSK capture produced substantially worse
64-QAM capture in the same room.

### 9.3 5 GHz QAM64 sweep

| Frequency | Observed capture |
|---|---:|
| 5.785 GHz, busy channel center | 77.6%, 83.6%, 89.4% over three runs |
| 5.840 GHz, band edge | 92% |
| 5.850 GHz, band edge | 95% |
| 5.860 GHz, band edge | 95.6% |
| 5.870 GHz, band edge | 95.6% |

Moving the same QAM64 implementation from an occupied channel center to a
quieter edge improved observed capture without a PHY-code change.

### 9.4 Dominant observed limitation

In a lab containing more than 20 nearby mesh Wi-Fi devices, the Schmidl-Cox
detector observed approximately 650-700 false triggers per second while only
about 80-90 real frames were decoded in the same 30-second window. Wi-Fi
preambles also contain repeated structure, so a self-correlation detector
cannot reliably identify waveform ownership. A sequence-specific
cross-correlation confirmation stage is proposed but not implemented.

These measurements characterize one RF environment and hardware setup. They
shall not be generalized into receiver-sensitivity, range, Doppler, or
regulatory claims.

## 10. Known limitations and open validation

- The header lacks its own error-detection code.
- Protocol-version mismatch is decoded but not currently rejected.
- Grid indices and LDPC wire-code assignments need frozen tables before an
  independent interoperability release.
- Preamble seeds, training seeds, grid configuration, and interleaver settings
  require out-of-band agreement.
- Schmidl-Cox acquisition is vulnerable to repeated structures from unrelated
  emitters, particularly nearby Wi-Fi.
- Periodic DMRS software integration is implemented on the feature branch.
  Pi/Pluto measurements remain the final planned completion gate.
- The periodic-DMRS interval is motivated by deployed-system precedent, not by
  a completed SpectraCUDA Doppler/coherence-time measurement campaign.
- Reported capture rate combines acquisition and decode behavior and is not
  equivalent to BER, PER conditioned on synchronization, or RF sensitivity.
- The ADALM-PLUTO 5 GHz measurements used a community device-identification
  override to expose the wider AD9363 tuning range; this is not a vendor-
  supported production configuration.

## 11. Traceability

The detailed implementation and evidence live in:

- `spectracuda/pipeline/ofdm.py` — frame generation and receiver pipeline;
- `spectracuda/framing/header.py` — 112-bit wire header;
- `spectracuda/ofdm/resource_grid.py` — subcarrier allocation;
- `spectracuda/framing/packetizer.py` — CRC and FEC composition;
- `docs/hardware-validation.md` — summarized OTA results;
- `docs/2026-09-06-rx-packet-loss-and-ism-band-characterization.md` — raw
  characterization session record;
- `docs/2026-09-20-dmrs-periodic-channel-refresh-plan.md` — DMRS design and
  acceptance criteria;
- `docs/flexlink-spec-review.md` — defects found in the external FlexLink
  document while scoping related ideas.

Future revisions should add frozen resource-grid tables, complete modulation
constellation/bit-labeling tables, exact preamble samples or generation
pseudocode, CRC polynomials, convolutional-code generator polynomials, LDPC
code assignments, and golden IQ/header/payload vectors. Those are required
before this can serve as a standalone third-party interoperability standard
rather than an implementation-aligned draft.
