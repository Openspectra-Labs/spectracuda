# Multipath severity: what actually breaks, and why

**Status: characterization complete, 2026-09-21. Simulation only.**
No PHY change was made to obtain these results. The soft-decision work
that followed from them is opt-in and off by default — see the last
section.

Reproduce with `examples/multipath_stress_study.py` (`--phase 2|3|4`),
`examples/multipath_phase5_diagnosis.py`, and
`examples/multipath_stress_report.py`. Raw JSON and rendered tables in
`debug/multipath_stress/`.

Reference PHY throughout: 20 MSps, `fft_size=256`, `cp_len=64`
(320-sample symbol = 16.0 µs, 78.125 kHz spacing — the 802.11ax
numerology), 16QAM, 15 dB, 8 pilots, 2 training symbols,
`timing_advance=2`, 5575 B, `rs_m8` + block interleaver + `conv_v27`,
DMRS `iv=32` unless stated. CP is 3.2 µs, so every delay used is well
inside it.

**Delay quantization is exact.** At 20 MSps one sample is 50 ns, so
50/100/200/500/1000 ns are 1/2/4/10/20 samples with no rounding.

## Three mechanisms, kept apart

| mechanism | isolated by |
|---|---|
| static frequency-selective fade | running `delta_f = 0` first |
| time-varying multipath (ageing) | adding differential Doppler at fixed fade |
| sync/acquisition failure | counting undetected frames separately from CRC failure |

## 1. A single echo breaks the link with nothing ageing

Packet success of 300, `delta_f = 0`:

| echo a | 50 ns | 100 ns | 200 ns | 500 ns | 1000 ns |
|---|---|---|---|---|---|
| 0.2 | 300 | 300 | 300 | 300 | 300 |
| 0.4 | 298 | 300 | 300 | 300 | 300 |
| **0.6** | **0** | **0** | **0** | **202** | **284** |
| 0.8 | 0 | 0 | 0 | 0 | 1 |
| 1.0 | 0 | 0 | 0 | 0 | 0 |

The threshold is between a=0.4 and a=0.6 — i.e. **7.4 dB of fade is
survivable, 12.0 dB is not**.

## 2. Short delays are the damaging ones

Counter-intuitive but unambiguous. Null **depth** is
`20·log₁₀((1+a)/(1−a))` and depends on amplitude alone — identical at
every delay:

| a | null depth | \|H\| range | EVM @ deep decile | EVM @ peak decile |
|---|---|---|---|---|
| 0.2 | 3.5 dB | 0.800–1.200 | 0.307 | 0.278 |
| 0.4 | 7.4 dB | 0.600–1.400 | 0.351 | 0.278 |
| 0.6 | 12.0 dB | 0.400–1.600 | 0.477 | 0.279 |
| 0.8 | 19.1 dB | 0.200–1.800 | 0.794 | 0.275 |
| 1.0 | ~50 dB | 0.006–2.000 | 0.926 | 0.275 |

What delay changes is how many nulls span the band, hence how **wide**
each one is. A 1-sample echo puts one broad null across 256 bins; a
20-sample echo puts twenty narrow ones.

**Failures track the nulls.** EVM at the healthy decile is flat at ~0.27
regardless of `a` — healthy subcarriers never degrade. Correlation
between per-subcarrier EVM and |H| runs −0.31 to −0.61 throughout.

**The channel estimator is not the problem.** Its normalized shape error
against the analytic `H[k] = 1 + a·e^{−j2πkδ/N+jφ}` is 0.064–0.072 and no
worse in the faded bins than elsewhere.

> **Methodology note.** These statistics are computed per frame against
> that frame's own nulls and only then averaged. The echo phase is
> redrawn every frame, so averaging the *spectra* first erases the nulls
> — an a=1.0 channel then reports a 3 dB ripple with min|H| = 1.000,
> which is impossible. The first version of this analysis made exactly
> that mistake.

## 3. It is Viterbi that breaks, not the demapper

Same echo amplitude, same ~16% of band faded, same raw demapper BER, no
Doppler — only fragmentation differs:

| delay | faded regions | widest | pre-Vit BER | post-Vit BER | RS min/mean/max | over 16 | pass |
|---|---|---|---|---|---|---|---|
| 50 ns | 1.1 | 36 bins | 0.0517 | **0.0706** ← amplified | 9 / 37.8 / 69 | 99% | 0/12 |
| 200 ns | 3.8 | 10 bins | 0.0478 | 0.0345 | 7 / 22.7 / 40 | 87% | 0/12 |
| 1000 ns | 17.7 | 2 bins | 0.0487 | **0.0044** | 0 / 2.9 / 17 | 0% | 11/12 |

The demapper delivers identical damage in all three. With one wide null
Viterbi **amplifies**; with eighteen narrow ones it corrects cleanly.

**The interleaver cannot help, structurally.** TX is
`RS → interleaver → conv`, so RX is `Viterbi → deinterleave → RS`: the
interleaver sits *after* Viterbi and spreads Viterbi's output across RS
codewords. It never shields Viterbi from a channel burst. At 50 ns even
the *best* RS codeword carries 9 errors — the deinterleaver faithfully
spread a hopeless Viterbi output evenly rather than leaving some clean.

Error clustering is measured as peak local **density** over a one-symbol
window, not run length: a faded subcarrier raises error probability
rather than forcing every bit wrong, so even badly faded regions give
short runs (max 4–5 bits) and run length cannot separate clustered from
uniform damage.

## 4. Ageing vs fade: DMRS only helps one of them

`iv=32` vs `iv=16`, 300 frames:

| a | delay | Δf | iv=32 | iv=16 | verdict |
|---|---|---|---|---|---|
| 0.4 | 200 ns | 300 | 0/300 | 190/300 | ageing |
| 0.4 | 500 ns | 300 | 0/300 | **300/300** | ageing, fully rescued |
| 0.6 | 200 ns | 100 | 0/300 | 0/300 | fade |
| 0.6 | 500 ns | 100 | 3/300 | 68/300 | improved, not rescued |
| 0.8 | any | any | 0/300 | 0/300 | fade |

At a=0.4 it is ageing and a faster refresh fixes it; at a≥0.6 it is a
fade and refresh changes nothing. a=0.4 fails at `iv=32` where a=0.2
passed in the earlier Doppler study, matching
`|ΔH| = 2a·|sin(π·Δf·ΔT)|` scaling linearly in `a`.

## 5. Sync holds, but acquisition does not always

`start_index` moves toward late as echoes strengthen (45% late at a=0.2,
92% at a=1.0) but **never exceeds 1 sample**, so `timing_advance=2`
remains sufficient. Separately, at a=1.0 / 50 ns **20.7% of frames are
never detected at all** — an acquisition failure, not a window-placement
one.

## 6. Randomized multi-tap ensembles

120 channels × 20 frames, LOS 1.0 plus 2–5 reflections (amplitude 0.1–0.8
uniform, delay 50–1000 ns on the 50 ns grid, uniform phase, differential
Doppler ±300 Hz):

- `iv=32`: median PER 100%, **8/120 (7%) fully clean**
- `iv=16`: median PER 100%, **9/120 (8%) fully clean**

This is a deliberately harsh draw — median strongest-echo/LOS is 0.64, so
half the ensemble starts past the a≈0.6 cliff. It characterizes that
distribution, not real UAV channels.

**The important finding: "strongest echo / LOS" is the wrong severity
metric.** Several moderate echoes add coherently at the worst subcarrier,
so composite null depth outruns any individual amplitude:

```
channel 6:  0.318 @300ns, 0.396 @450ns  -- no echo above 0.40
            composite null 15.0 dB      -> 100% PER
channel 58: 0.368 @50ns,  0.431 @250ns
            composite null 17.0 dB      -> 100% PER
```

A single 0.4 echo is 7.4 dB and passes; a single 0.6 echo is 12.0 dB and
fails. **Severity should be judged from composite |H| null depth, not
from the largest tap.**

## 7. Soft-decision Viterbi recovers most of this

The diagnosis above pointed at one specific information loss: hard
demodulation returns 0/1 and discards how close the symbol was to the
decision boundary, so a faded subcarrier hands the decoder wrong bits
marked *maximally confident*. That was implemented as an opt-in flag —
`Ofdm(soft_decision=True)`, see `tests/test_soft_decision.py`.

40 frames/cell, same seeds as above:

| case | hard | soft |
|---|---|---|
| a=0.2 clean | 40/40 | 40/40 |
| a=0.6 d=50 ns | 0/40 | 12/40 |
| a=0.6 d=200 ns | 0/40 | **40/40** |
| a=0.6 d=500 ns | 30/40 | **40/40** |
| a=0.8 d=1000 ns | 0/40 | **36/40** |
| a=0.4 d=500 ns Δf=300 | 0/40 | **40/40** |

Note the last row: it previously required `iv=16` to survive, and soft
decision rescues it at `iv=32` — retaining the lower DMRS overhead.

The a=0.6 / 50 ns case remains largely unrecovered, and that is
informative rather than disappointing: one broad null removes too much
contiguous coded information for any amount of confidence weighting to
reconstruct. **That is the residual regime — broad, deep fades — and it
is where diversity or a stronger code, not reliability weighting, would
be the lever.**

### What soft decision costs

Two optimizations landed after the first measurement, and both moved it a
long way. The figures below supersede the 6.6x originally recorded here.

`rx_process`, clean frame both paths decode successfully:

| | first measured | after numba demapper | after SSE soft decode |
|---|---|---|---|
| hard | 1.9 ms | 1.9 ms | 1.9 ms |
| soft | 12.8 ms | 10.3 ms | — |
| ratio | 6.6x | 5.3x | ~2x (see below) |

The two pieces, measured separately:

* **Soft demapper** 2.80 ms -> 0.56 ms. A numba kernel mirroring the
  existing hard-decision one, decomposed PER AXIS: square Gray QAM is
  separable, so an I-bit's LLR depends only on the I axis and the Q term
  cancels in the difference. 16 distances + 64 comparisons per 16QAM
  symbol becomes 8 + 16.
* **Soft Viterbi** 11.56 ms -> 1.61 ms at k=51000, a 7.3x gain that
  needed no new C at all: libcorrect already ships
  `correct_convolutional_sse_decode_soft` and the SSE build already
  exported it. Output is bit-identical to the portable loop.

A caution that cost a wrong conclusion once: **libcorrect's soft decoder
is strongly data-dependent.** At k=51000 the portable loop measures
3.6 ms on 0/255 rails, 3.2 ms on all-erasure, but 10.9 ms on realistic
graded values. An early benchmark fed it rails and reported a 7x
soft/hard gap when the real figure was ~22x. Always measure it on graded
input.

End-to-end RX SDU throughput (`examples/benchmark_x86_stages_v3.py
36000 <modem> cp=64 dmrs=32 [soft]`), 20 Msps budget:

| modem | hard | soft | hard vs budget |
|---|---|---|---|
| qpsk | 10.7-10.9 Mbps (25.1-25.5 Msps) | 6.6-7.2 Mbps | **OK** |
| qam16 | 18.1-19.2 Mbps (16.4-17.4 Msps) | 9.6-9.8 Mbps | 0.82-0.87x short |
| qam64 | 22.6-23.3 Mbps (13.8-14.2 Msps) | 11.1-11.5 Mbps | 0.69-0.71x short |

Only qpsk with hard decision makes real time. Soft roughly halves Msps
across the board -- 1.4-2.4x, against 4-6x before these two changes.

**Two gaps remain, both real:**

* **ARM is untouched.** `sse_available()` is False on the Pi-5 target, so
  soft decode there still runs the 11.5 ms portable loop. libcorrect has
  no NEON soft kernel and one cannot be honestly validated from an x86
  machine.
* **The `fast` kernel has no soft variant**, so soft does not reach the
  hard path's best backend (1.24 ms against 0.44 ms in the RX stage
  table). Its header explains why that is a design change rather than a
  port: path metrics are 8-bit, justified by the K=7 survivor spread
  being "<= 12 **for hard decisions**". Soft branch metrics are far
  larger. A soft variant needs 16-bit metrics (halving states per vector,
  giving back much of the gain) or branch metrics quantized tightly
  enough to keep the spread bounded -- which the 4-bit LLR result below
  suggests may be feasible, but it needs an overflow-safety argument, not
  an assumption.

## 8. How many LLR bits does it need?

**This question is worth money in hardware and nothing in software.** The
distinction matters and is easy to get backwards.

In an FPGA the LLR width propagates straight into gate count. The
add-compare-select unit computes `PM_new = min(PM_old + BM)`, so the
branch-metric width sets the adder and comparator widths, the path-metric
width follows from it, and both are instantiated 64 times for a K=7
trellis. Halving 8 bits to 4 is a real area and timing saving, repeated
across every state.

In software it buys nothing. libcorrect's soft decoder takes a `uint8`
per bit whatever the LLR resolution is -- quantizing to 4 bits only means
fewer DISTINCT VALUES inside that byte, not a smaller byte, and the
decoder does identical work. The only measurable effect is one extra
round-and-divide per bit in the demapper, i.e. marginally SLOWER.

So the throughput figures in the section above and the widths below are
independent results. The benchmark runs at the library defaults
(`soft_llr_bits=None`, full 8-bit, `soft_llr_clip=6.0`); nothing in the
timing tables is quantized, and quantizing it would not improve them.

Run against exactly the seeds above
(`examples/soft_llr_quantization_study.py`), 40 frames/cell.

Signed quantization, 2L+1 levels with L = 2^(b-1)-1, at the default
clip of 6 nats:

| case | hard | 2-bit | 3-bit | 4-bit | 5-bit | 6-bit | float |
|---|---|---|---|---|---|---|---|
| a=0.2 clean | 40/40 | **32/40** | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 |
| a=0.6 d=50 ns | 0/40 | 0/40 | 0/40 | 2/40 | 5/40 | 10/40 | 12/40 |
| a=0.6 d=200 ns | 0/40 | 0/40 | 0/40 | 25/40 | 40/40 | 40/40 | 40/40 |
| a=0.6 d=500 ns | 30/40 | **0/40** | **20/40** | 40/40 | 40/40 | 40/40 | 40/40 |
| a=0.8 d=1000 ns | 0/40 | 0/40 | 0/40 | 31/40 | 34/40 | 36/40 | 36/40 |
| a=0.4 d=500 ns Δf=300 | 0/40 | 0/40 | 39/40 | 40/40 | 40/40 | 40/40 | 40/40 |

**2-bit and 3-bit are WORSE THAN HARD DECISION** in some cells -- 2-bit
loses the clean channel (32/40) and destroys a=0.6/500 ns (0/40 against
hard's 30/40). A coarse LLR is not a weakened soft decoder, it is a
harmful one. Anything sized on the assumption that low bit widths degrade
gracefully would be wrong.

### Clipping matters more than width -- but only once quantized

Same 4-bit LLR, sweeping what those 15 levels SPAN:

| case | clip 2 | clip 3 | clip 4 | clip 6 | clip 10 | clip 20 | float |
|---|---|---|---|---|---|---|---|
| a=0.6 d=50 ns | 7/40 | 4/40 | 3/40 | 2/40 | 1/40 | 0/40 | 12/40 |
| a=0.6 d=200 ns | **40/40** | 40/40 | 38/40 | 25/40 | 0/40 | 0/40 | 40/40 |
| a=0.6 d=500 ns | 40/40 | 40/40 | 40/40 | 40/40 | 39/40 | 0/40 | 40/40 |
| a=0.8 d=1000 ns | 32/40 | 33/40 | 35/40 | 31/40 | 15/40 | 0/40 | 36/40 |
| a=0.4 d=500 ns Δf=300 | 40/40 | 40/40 | 40/40 | 40/40 | 40/40 | 15/40 | 40/40 |

At clip 6 the fixed-width table would have said 5 bits. At clip 2-3,
**4 bits reaches float in three of five cells and comes within 4 frames
on a fourth** -- so the width read off a single clip setting was one bit
too conservative. Loose clipping is catastrophic: at clip 20 nearly
everything quantizes onto the zero level and the information is gone.

At FULL 8-bit precision the same clip sweep is flat (12/40, 40/40, 40/40,
33-36/40, 40/40 across clip 2 to 10), so clip sensitivity is purely a
quantization effect and the shipped default of 6.0 needs no change.

### It does not degrade gracefully -- 4 bits is a floor, not a knee

The intuition to discard: that fewer bits means proportionally less
coding gain, so a designer short of area could take 3 bits and accept a
small loss. Below 4 bits the gain goes NEGATIVE -- worse than not doing
soft decision at all. A coarse LLR tells the decoder "fairly sure" about
bits it should be flagging as unknown, and confident wrong information is
worse than the hard decoder's honest ignorance.

So 4 bits is not a point on a curve to be traded against area. It is a
floor with a cliff underneath it.

### Clipping is the other half of the width decision

`clip` sets what the quantizer's levels SPAN, in LLR units (nats). Too
wide and almost everything lands on the middle level -- at clip 20 the
4-bit case collapses to 0/40. Too narrow and levels are spent resolving
bits that were already certain.

The same 4-bit LLR on the same channel reads 25/40 at clip 6 and 40/40 at
clip 2. Read from the default clip alone, the answer looks like 5 bits;
with clip tuned, 4 is enough. **A whole bit of hardware, decided by a
parameter with nothing to do with the decoder.** Any width quoted without
its clip is meaningless.

### Sizing conclusion

**4-bit signed LLR with a clip of 2-3 nats** is the hardware target: it
keeps essentially all the coding gain outside the known broad-fade
residual, at half the branch-metric width. 5-bit buys insurance. 3-bit
and below must not be used.

This is a simulation result at one SNR, one modulation and one payload.
The clip optimum was found by sweeping on the SAME five cells it is then
quoted against -- fitting and reporting on one dataset, which will always
flatter itself. Before this sizes RTL it should be re-checked on channels
that were not used to choose it. That is a caveat about method; the 4-bit
floor itself held across the full range of cases and does not depend on
the clip tuning.

## 9. The worst channel: reflections AND Doppler together

Nothing above tested the combination. The phase 3 sweep had Doppler plus
reflections but neither soft decision nor `interleaver2`; the
`interleaver2` matrix had reflections and both features but `delta_f = 0`,
so DMRS was inert throughout it.

15 dB, 16QAM, iv=32, f_los=1600 Hz, 100 frames/cell
(`examples/worst_case_matrix.py`):

| a | delay | Δf | baseline | il2 | soft | both |
|---|---|---|---|---|---|---|
| 0.6 | 50 ns | 0 | 0 | **94** | 30 | 97 |
| 0.6 | 50 ns | 100 | 0 | 72 | 12 | **99** |
| 0.6 | 50 ns | 300 | 0 | **0** | **0** | 10 |
| 0.6 | 200 ns | 0 | 0 | 93 | 98 | 98 |
| 0.6 | 200 ns | 100 | 0 | 58 | 88 | 92 |
| 0.6 | 200 ns | 300 | 0 | **0** | **0** | 1 |
| 0.8 | 50 ns | 0 | 0 | 10 | 3 | **80** |
| 0.8 | 50 ns | 100 | 0 | 4 | 0 | **70** |
| 0.8 | 50 ns | 300 | 0 | 0 | 0 | 0 |
| 0.8 | 200 ns | 0 | 0 | 29 | 0 | **83** |
| 0.8 | 200 ns | 100 | 0 | 0 | 0 | **52** |
| 0.8 | 200 ns | 300 | 0 | 0 | 0 | 0 |

`interleaver2` degrades exactly as its mechanism predicts -- 94 -> 72 -> 0
as Δf rises at a=0.6/50 ns. It disperses damage that sits in FIXED
subcarriers, and a moving null has less fixed structure to disperse. That
degradation is the confirmation of the mechanism, not a disappointment:
had it helped equally at 300 Hz, the explanation in §3 would have been
wrong.

Δf=300 collapses the whole matrix at iv=32 because that is squarely the
AGEING regime -- `2a|sin(π·Δf·ΔT)|` is 0.57 at a=0.6 with ΔT=528 µs.
Sweeping the DMRS interval with both features on, 60 frames/cell:

| a | delay | iv=32 (528 µs) | iv=16 (272 µs) | iv=8 (144 µs) |
|---|---|---|---|---|
| 0.6 | 50 ns | 7/60 | **57/60** | **59/60** |
| 0.6 | 200 ns | 0/60 | **50/60** | **59/60** |
| 0.8 | 50 ns | 0/60 | 25/60 | 33/60 |
| 0.8 | 200 ns | 0/60 | 5/60 | 16/60 |

### Conclusion: three levers, three mechanisms, none interchangeable

| lever | fixes | the only thing that works at |
|---|---|---|
| DMRS interval | channel ageing | Δf=300 (il2 and soft both give 0) |
| `interleaver2` | frequency-contiguous fade | Δf=0 (soft alone gives 30) |
| soft decision | deep-but-present subcarriers | a=0.8, once the other two are in place |

The a=0.6 / 50 ns cell shows the whole thing: dead at every Doppler
without help, recovered at every Doppler with it -- but by a DIFFERENT
lever each time (il2 at Δf=0, both at 100, DMRS at 300).

**Practical blocker, now with a concrete case.** `iv=8` is what nearly
closes a=0.6 at Δf=300 (59/60) and it has no wire code: the 2-bit
`dmrs_period` field carries {0,16,32,64}. This was already recorded in
`docs/2026-09-21-dmrs-interval-operating-guidance.md` as a blocker for the
10 MSps high-mobility mode; there is now a specific channel that needs it.

**Remaining edge of the envelope.** a=0.8 at Δf=300 stays broken with all
three levers (33/60, 16/60). A -2 dB echo carrying 300 Hz of differential
Doppler is past what this PHY does, and that is a boundary rather than a
missing feature.

## What this does not establish

- **Simulation only**, deterministic taps with randomized phase — no
  fading process, no angle-of-arrival geometry, no standards channel
  model. The Pi/Pluto gate named in the PHY specification is untouched.
- **One SNR (15 dB), one payload, one modulation.** Integer-sample delays
  only; fractional delay is unsupported and not faked.
- The randomized ensemble's amplitude range is a **stress draw**, chosen
  to find the cliff, not to represent a deployment.
