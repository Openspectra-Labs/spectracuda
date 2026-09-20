# FlexLink PHY spec review — defects found on read-through

Source document: `FlexLinkPhy202x_v11.docx`
Version 00.10, April 23 2023, author Andreas Schwarzinger, status Preliminary.

Reviewed 2026-09-20 while scoping which FlexLink features to adopt into
spectracuda. This file records only the **defects** — internal contradictions,
text/code mismatches and arithmetic that doesn't close. Feature adoption is
tracked separately.

Section numbers below refer to the document's own numbering.

---

## 1. AGC burst duration contradicts itself (§2.1.1 vs §3.1)

- §2.1.1 "Preamble": *"a four microsecond long wideband AGC (automatic gain
  control) burst"*
- §3.1 "The AGC Burst": *"The AgcBurst shall occupy 5 microseconds, which at
  20.48MHz yields 102 samples."*

102 samples at 20.48 MHz is 4.98 µs, so §3.1 is self-consistent and §2.1.1 is
the wrong one. (A true 4 µs burst would be 82 samples, matching the CP length —
which may be where the 4 µs came from.)

**Suggested fix:** change §2.1.1 to "five microsecond".

---

## 2. PreambleA duration contradicts its own reference code (§3.2)

The "Duration" paragraph states:

> The duration of PreambleA for this scenario shall be 5120 samples, which at a
> clock rate of 20.48MHz yields 250 microseconds.

The `GeneratePreambleA()` listing immediately below computes:

```python
NumSamples = math.floor(220e-6 / Ts)
```

which at `SampleRate = 20.48e6` is 4505 samples / 220 µs, not 5120 / 250 µs.

**Suggested fix:** decide which is normative and make the other match. 5120 is
the better choice — it is an integer number of the 128-sample tone period
(5120 / 128 = 40 exactly), whereas 4505 is not (4505 / 128 = 35.2), so the 220 µs
version truncates mid-period.

---

## 3. PreambleA's short form is labelled "PreambleB" (§3.2)

> For the case that frequency offset acquisition and removal are not requires,
> **PreambleB** shall feature a length of 512 samples or 25 microseconds.

This sentence is in §3.2 PreambleA, describing PreambleA's detection-only form,
and it agrees with §2.1.1 ("PreambleA shall consist of a 512-sample waveform if
one packet detection is desired"). PreambleB is defined elsewhere as one OFDM
symbol. The word should be PreambleA.

(Also "are not requires" → "is not required".)

---

## 4. Signal field: stated protected-bit count doesn't match the table (§2.3)

Table 2-2 rows, excluding the CRC:

| Field | Bits |
|---|---|
| EBS1 | 2 |
| NTB1 | 16 |
| FEC1 | 2 |
| RM1 | 3 |
| BPS1 | 2 |
| EBS2 | 2 |
| NTB2 | 16 |
| FEC2 | 2 |
| RM2 | 3 |
| BPS2 | 2 |
| User Bits | 24 |
| **Total** | **74** |

Plus the 16-bit CRC gives 90, which matches the table's own total row. But the
CRC row's description reads *"Cyclic Redundancy check bits that protect the 72
signal field bits"*.

74 ≠ 72. Either the description should say 74, or two bits are missing from the
table that the author intended to remove.

---

## 5. Control information: prose says 12 bits, table totals 14 (§2.2.3.1, §2.2.5)

Table 2-1 rows: 4 + 2 + 1 + 1 + 1 + 1 + 1 + 3 = **14**, and the table's own
total row says 14.

But §2.2.3.1 says, twice, that there are 12:

> The **12** control information bits are only sent from Port 0 and are BPSK
> mapped into the resource elements assigned to them.

> Thus, if the bandwidth has 300 triplets, and we wish to map **12** control
> bits, then the vectors will feature a length of 25.

This one matters more than a typo, because the repetition-vector length is
derived from it: with 300 triplets, 12 bits gives length-25 vectors, but 14 bits
gives 300/14 = 21.43, which does not divide evenly. The mapping rule as written
has no defined behaviour when the triplet count is not an integer multiple of
the control-bit count.

**Suggested fix:** state the bit count once, normatively, and define the
rounding rule for the vector length (floor, with the remaining resource elements
left unused, is the usual choice).

---

## 6. Signal field FEC is specified twice, with two different codes (§1.1.7.2 vs §2.3)

- §1.1.7.2 "FEC – Forward Error Correction": *"The signal field will use a
  256-bit polar encoder / decoder coding."*
- §2.3 "Forward Error Correction in the Signal Field": *"The 90 bits are then
  protected by a rate 1/3 (L = 7) convolutional encoder, and rate matching is
  then used to repeat the encoded bits..."*

These are different codes with different decoders. Only one can be normative.

The §2.3 version is the more likely intent — a 90-bit payload does not fit a
256-bit polar block without its own rate matching, and the rate-1/3 K=7
convolutional coder is the 802.11a-compatible choice consistent with the rest of
the document's WLAN lineage.

---

## 7. `DetectPreambleA()` reads index −1 on the first iteration (§3.2.1)

```python
for Index in range(0, RxLength - PeriodSamples):
    ...
    CurrentCovariance[Index] = CurrentCovariance[Index - 1] + A * np.conj(B)
    CurrentVariance[Index]   = CurrentVariance[Index - 1]   + B * np.conj(B)
```

At `Index == 0` this reads `CurrentCovariance[-1]` / `CurrentVariance[-1]`,
which in NumPy wraps to the **last** element of the array rather than raising.
Those elements are still at their initialised values (0 and 1 respectively) on
the first pass, so the visible output happens to be correct, and the
`if Index < 100: Ratio[Index] = 0` guard hides the transient anyway.

It is still wrong as written, and it breaks the moment the function is called
twice on the same preallocated buffers, or ported to a language where `[-1]` is
an out-of-bounds access. The loop should start at `Index = 1`, or the
accumulator update should be special-cased at 0.

---

## 8. 5 MHz bandwidth has a different symbol duration from the others (§1.1.6)

Figure 1-1's CP row gives, for the four bandwidths:

| BW (MHz) | FFT | Fs (MHz) | CP samples | CP (µs) | Symbol (µs) |
|---|---|---|---|---|---|
| 5 | 256 | 5.12 | 20 | 3.906 | **53.906** |
| 10 | 512 | 10.24 | 41 | 4.004 | 54.004 |
| 20 | 1024 | 20.48 | 82 | 4.004 | 54.004 |
| 40 | 2048 | 40.96 | 164 | 4.004 | 54.004 |

The 5 MHz case needs 20.5 samples to match, which is not an integer, so it was
rounded down to 20. The result is that a 5 MHz symbol is 0.098 µs shorter than a
symbol at every other bandwidth — a 0.18% difference.

This is small but it is not harmless: it means a frame structure defined as a
fixed number of OFDM symbols has a *different wall-clock duration* at 5 MHz than
at 10/20/40 MHz. Any scheme that aligns frames or reference symbols to an
absolute time grid (e.g. a 1 ms slot boundary) will drift at 5 MHz only.

**Suggested fix:** either state explicitly that 5 MHz is exempt from
symbol-duration equality, or define the 5 MHz CP as 21 samples (4.102 µs,
symbol 54.102 µs) and accept the error in the other direction, or drop to a
40 kHz subcarrier spacing at 5 MHz so the CP lands on an integer.

---

## 9. Doppler worked example uses 50 m/s where the text says 55 m/s (§2.2.1)

> The highest speed difference supported by the specification is 200Kph, or 55
> m/sec. The maximum Doppler frequency is computed as follows for a center
> frequency of 5.9GHz.

The text following the (image-only) equation uses **983 Hz**:

> The reference signal symbols should therefore appear every 0.25 / 983 Hz, or
> approximately every 250 microseconds.

But 983 Hz at 5.9 GHz implies v = 983 × c / 5.9e9 = **49.98 m/s**, i.e. 180 kph,
not 200. The correct figure for 200 kph (55.56 m/s) is:

```
f_d = 55.56 × 5.9e9 / 3e8 = 1093 Hz
```

which moves the reference-symbol interval from 254 µs to **229 µs** — a 10%
tighter requirement than the document states. Also note 200 kph is 55.56 m/s,
so even the "55 m/sec" in the prose is rounded down.

**Suggested fix:** restate as 1093 Hz / 229 µs, or drop the supported speed to
180 kph if 983 Hz was the intended design point.

---

## 10. Minor / editorial

- §1.1.7.1: *"The Flex link should support..."* — "FlexLink" elsewhere.
- §2.2.4: *"Phase noise reference signals shall feature the **BSPK** value
  1 + 0j"* — BPSK.
- §2.2.4: *"Phase noise reference signals are **place** according to"* — placed.
- §2.3 / Table 2-2 and §7 Address Map: **LDCP** appears eight times where LDPC is
  meant.
- Table 2-2 / Address Map: `FEC1`/`FEC2` map `[0, 1, 2, 3] → [½, 2/3, ¾, ¾]`.
  Code 3 duplicates code 2. If the 5/6 rate that the LDPC variant list implies
  were intended for code 3, the table should say so; otherwise code 3 should be
  marked reserved.
- §3.1: *"The FFT output shall be mapped into a length 1024 IFFT input
  **bugger**"* — buffer.
- Revision history: version 00.05 is skipped, and 00.07 and 00.09 have no date.
- Table of contents lists §5 "Data Link Layer (DLL)" and §6 "Transaction Layer
  (TL)", but revision 00.10's own changelog says *"remove data link layers and
  MAC chapters"* and the body has no such sections — the TOC was not regenerated
  after the removal. The body instead has an unnumbered "Address Map" chapter
  that the TOC does not list.
- §4.1 figure is captioned "Figure 88" in a document where every other figure
  follows a `chapter-index` scheme (Figure 3-3, Figure 2-4, …).

---

## Not defects, but worth flagging to the author

**The PreambleA tone spacing is stated in MHz where kHz is meant.** In
`ProcessPreambleA()`'s comments:

```python
if   OffsetIndex <  -320:  # Then the maximum peak is the one belonging to -1920MHz
```

`CosineFrequencyA = 4*160e3` = 640 kHz and `CosineFrequencyB = 12*160e3` =
1920 kHz, so all four comments naming "-1920MHz", "-640MHz", "+640MHz",
"+1920MHz" should read kHz. The code is correct; only the comments are wrong.

**The fine-frequency estimate is limited to ±20 kHz.** `FreqOffset` is derived
from the phase of `Tone[1] * conj(Tone[0])` with the two samples 512 apart at
20.48 MHz, so it aliases once the offset exceeds half of 20.48e6/512 = ±20 kHz.
The coarse peak search resolves the rest, but the document never states the
resulting total acquisition range. It would be worth adding — it is the number
an implementer needs in order to choose an oscillator.
