# DMRS interval vs differential Doppler: what actually ages H[k]

**Status: characterization complete, 2026-09-21. Simulation only. No PHY,
DMRS format, pilot tracker, equalizer, FEC or wire-protocol change was
made, and none is proposed here without the hardware gate below.**

Reproduce with `examples/dmrs_doppler_study.py` (`--part sync | common |
sweep | analytic | genie`).

Related:

- `docs/2026-09-20-dmrs-periodic-channel-refresh-plan.md` — the feature;
- `docs/2026-09-20-dmrs-static-channel-cost.md` — the earlier retraction,
  whose methodology rules (trailing samples, length-independent noise)
  this study follows;
- `docs/spectracuda-phy-specification.md` §11 — still correctly states
  that the DMRS interval rests on deployed-system precedent rather than
  our own measurement. **This study does not close that gap**: it is
  simulation, and the Pi/Pluto mobility run remains the completion gate.

## Summary

Three results, in order of how much they change the picture.

1. **Channel aging is governed by DIFFERENTIAL Doppler between multipath
   components, not by absolute platform Doppler.** Putting 667 Hz on
   *both* paths delivers every frame with a flat EVM profile; putting
   667 Hz of *difference* between them loses every frame, and does so
   identically whether the common Doppler is 0 or 667 Hz.

2. **The production 512 us minimum DMRS interval tolerates roughly
   300 Hz of differential Doppler at a = 0.2 / 16QAM / 25 dB.** An
   earlier reading of this work concluded 512 us was about 4x too coarse.
   That was wrong, and came from imposing a full platform Doppler on the
   echo against a stationary LOS — which is a 667 Hz *differential*
   spread, a severe channel, not a restatement of platform speed.

3. **A one-sample synchronization offset costs ~0.09 EVM on its own**, is
   unrelated to Doppler, and is worth fixing independently. A cyclic
   prefix protects the FFT window in the EARLY direction only.

## Methodology and its limits

Fixed configuration throughout: `fft_size=256`, `cp_len=64` (320-sample
symbol, 32.0 us at 10 MSps), `n_pilot=8`, `n_data=216`, 16QAM,
`fec="rs_m8"`, `fec1="conv_v27"`, `interleaver="block"` with
`unit_bits=8`, `crc32`, `n_training_symbols=2`, 5575 B payload, 25 dB
SNR, echo amplitude a = 0.2 (Rician K = 14 dB). Pilot CPE correction is
always on; it has no flag.

The channel is

```
rx[n] = e^{j2*pi*f_los*n/fs} * tx[n]
      + a * e^{j2*pi*f_echo*n/fs} * tx[n-1]
```

with `delta_f = f_echo - f_los`. Doppler is applied sample-by-sample and
runs continuously through the frame; it is never reset at a DMRS or
segment boundary, and the channel has no knowledge of the frame
structure. The echo delay is an integer 1 sample, not fractional.

**This is a single deterministic rotating tap, not a Jakes/WSSUS fading
model.** It has no fading statistics, no delay-spread profile and no
angle-of-arrival geometry. Its phase wraps past 2*pi within a long
refresh interval, so the 667 Hz column can score *better* than 400 Hz at
the coarsest interval — an aliasing artifact, not physics. Consequently:

- the tolerance figures below are the **shape** of the requirement, not
  certified limits;
- **`delta_f` is deliberately never converted to km/h.** That conversion
  needs a scattering geometry this model does not have. Absolute platform
  Doppler and differential multipath Doppler spread are different
  quantities and are kept separate throughout.

Two harness-only extensions, both reverted before exit and neither
touching the production PHY:

- **Independent LOS/echo Doppler.** `sim/channel.py` offers a constant
  carrier offset, which is a pure phase rotation the CPE correction
  already removes, so it cannot age `H[k]` at all.
- **DMRS intervals outside the 2-bit wire codes {0,16,32,64}.** The wire
  field is not widened; the harness re-points what its four codes *mean*
  for the duration of a run. Short intervals also exceed
  `MAX_PAYLOAD_SYMBOLS=128` at this payload (128 us needs 150 slots), so
  that guard is raised on the instance only.

**A naming trap.** `framing/dmrs.py` defines `interval=n` as "a DMRS
after every n DATA symbols", so consecutive DMRS are `(n+1)` slots apart.
The labels 128/256/512/1024 us are data-symbol spans; the true refresh
periods are **160/288/544/1056 us**. The analytic fit only closes when
the true value is used.

## 1. Common vs differential Doppler

`a=0.2`, iv=32, 10 frames per case (`--part common`).

| case | f_los | f_echo | delta_f | CRC | mean EVM | EVM at t = 32/256/512/768/1024 us |
|---|---|---|---|---|---|---|
| 1 | 0 | 667 | 667 | 0/10 | 0.308 | 0.186 0.270 0.347 0.376 0.341 |
| 2 | 667 | 667 | **0** | **10/10** | **0.148** | 0.175 0.192 0.177 0.190 0.170 |
| 3 | 667 | 767 | 100 | 10/10 | 0.166 | 0.180 0.199 0.188 0.215 0.207 |
| 4 | 667 | 867 | 200 | 9/10 | 0.201 | 0.183 0.208 0.211 0.264 0.273 |
| 5 | 667 | 1067 | 400 | 0/10 | 0.263 | 0.181 0.227 0.271 0.341 0.361 |
| 6 | 667 | 1334 | 667 | 0/10 | 0.308 | 0.185 0.272 0.345 0.372 0.344 |

Cases 1 and 6 carry the same `delta_f` with 0 and 667 Hz of common
Doppler and are indistinguishable. Case 2 carries 667 Hz on both paths
and is flat — no sawtooth, no loss. CFO and per-symbol CPE absorb common
Doppler completely.

## 2. Packet success vs DMRS interval and differential Doppler

100 frames per cell, `f_los = 667 Hz` present in **every** cell
(`--part sweep`).

| DMRS | 0 | 50 | 100 | 200 | 300 | 400 | 500 | 667 Hz |
|---|---|---|---|---|---|---|---|---|
| 128 us | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| 256 us | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 92 |
| 512 us | 100 | 100 | 100 | 100 | 99 | 86 | 62 | 29 |
| 1024 us | 100 | 100 | 100 | 89 | 43 | 0 | 0 | 0 |

Mean EVM:

| DMRS | 0 | 50 | 100 | 200 | 300 | 400 | 500 | 667 |
|---|---|---|---|---|---|---|---|---|
| 128 us | 0.114 | 0.114 | 0.116 | 0.118 | 0.121 | 0.125 | 0.131 | 0.140 |
| 256 us | 0.114 | 0.114 | 0.117 | 0.124 | 0.132 | 0.143 | 0.155 | 0.175 |
| 512 us | 0.114 | 0.116 | 0.122 | 0.140 | 0.161 | 0.183 | 0.206 | 0.238 |
| 1024 us | 0.115 | 0.122 | 0.138 | 0.180 | 0.218 | 0.250 | 0.275 | 0.299 |

Worst-symbol EVM (mean over frames of each frame's maximum):

| DMRS | 0 | 50 | 100 | 200 | 300 | 400 | 500 | 667 |
|---|---|---|---|---|---|---|---|---|
| 128 us | 0.162 | 0.162 | 0.164 | 0.167 | 0.172 | 0.182 | 0.196 | 0.220 |
| 256 us | 0.161 | 0.161 | 0.164 | 0.175 | 0.193 | 0.216 | 0.239 | 0.272 |
| 512 us | 0.160 | 0.162 | 0.172 | 0.206 | 0.245 | 0.283 | 0.320 | 0.366 |
| 1024 us | 0.160 | 0.169 | 0.201 | 0.278 | 0.344 | 0.381 | 0.396 | 0.403 |

Tolerance at >= 99% delivery, with the overhead each interval costs:

| nominal | true dT | tolerable delta_f | DMRS overhead | slots for 5575 B |
|---|---|---|---|---|
| 128 us | 160 us | >= 667 Hz | 20% | 150 |
| 256 us | 288 us | ~500 Hz | 11% | 135 |
| 512 us | 544 us | ~300 Hz | 5.9% | 128 |
| 1024 us | 1056 us | ~100-150 Hz | 3.0% | 122 |

## 3. The decoder threshold is EVM ~0.17, not 0.10

An earlier note assumed a 0.10 EVM budget for 16QAM and derived a 121 us
refresh requirement from it. **Both the budget and the requirement were
wrong.** Measured against the full shipped chain:

| DMRS | delta_f | raw BER | post-Viterbi | RS byte-err/cw | cw>16 | PER% |
|---|---|---|---|---|---|---|
| 512 us | 200 | 0.0099 | 0.0016 | 0.9 | 0/25 | 0 |
| 512 us | 300 | 0.0129 | 0.0041 | 2.3 | 0/25 | 1 |
| 512 us | 400 | 0.0180 | 0.0093 | 5.0 | 0.5/25 | 14 |
| 512 us | 500 | 0.0243 | 0.0178 | 9.5 | 5.4/25 | 38 |
| 512 us | 667 | 0.0368 | 0.0376 | 20.2 | 16.2/25 | 71 |
| 1024 us | 200 | 0.0183 | 0.0086 | 4.8 | 0.5/25 | 11 |
| 1024 us | 300 | 0.0307 | 0.0291 | 15.8 | 11.3/25 | 57 |
| 1024 us | 400 | 0.0449 | 0.0558 | 29.7 | 22.3/25 | 100 |

The chain turns unreliable at **raw demapper BER ~0.018-0.024**, i.e.
**mean EVM ~0.16-0.18**, worst-symbol EVM ~0.28.

Two structural notes fall out:

- **`conv_v27` crosses from correcting to amplifying at raw BER ~0.03**
  (1024 us / 400 Hz: 0.0449 in, 0.0558 out). The Viterbi decoder is
  hard-decision by deliberate choice (see `hls/rundown.md`); soft LLRs
  are the obvious lever and are *not* proposed here.
- **The interleaver sits between the two codes**, not between the inner
  code and the modulator: TX is `CRC -> RS -> interleaver -> conv ->
  16QAM`, so RX is `Viterbi -> deinterleave -> RS`. It spreads Viterbi's
  burst output across RS codewords; it never shields Viterbi from channel
  bursts, and cannot help with this failure by construction. Measured
  channel errors are not bursty anyway (max run 2 bits pre-Viterbi) — a
  slowly rotating notch makes a large fraction of symbols mildly bad, not
  a few symbols catastrophically bad.

## 4. The analytic relation

For a single delayed echo, a zero-order-hold estimate held for `dT`
leaves

```
|dH| = 2*a*|sin(pi*delta_f*dT)|
```

Measured against the receiver's own successive DMRS estimates, de-rotated
by the common phase between them, noiseless (`--part analytic`):

| nominal | delta_f | true dT | measured | analytic | ratio |
|---|---|---|---|---|---|
| 128 us | 667 | 160 us | 0.1275 | 0.1316 | 0.97 |
| 256 us | 200 | 288 us | 0.0704 | 0.0720 | 0.98 |
| 512 us | 400 | 544 us | 0.2480 | 0.2526 | 0.98 |
| 512 us | 667 | 544 us | 0.3595 | 0.3634 | 0.99 |
| 1024 us | 200 | 1056 us | 0.2455 | 0.2464 | 1.00 |
| 1024 us | 667 | 1056 us | 0.3198 | 0.3204 | 1.00 |

Ratio 0.94-1.00 across the full 4x5 grid. Control at `delta_f = 200 Hz`,
`dT = 512 us`, three different common Dopplers:

| f_los | f_echo | measured \|dH\| |
|---|---|---|
| 0 | 200 | 0.1319 |
| 667 | 867 | 0.1319 |
| 2000 | 2200 | 0.1319 |

Identical to four decimals: the relation depends on `delta_f` and not at
all on absolute Doppler.

Note the `sin` is not monotonic. Beyond `delta_f*dT = 0.5` the residual
turns over, which is why the coarsest interval's 667 Hz column can beat
its 400 Hz column. With a single tone that is aliasing; a real Doppler
spectrum would not behave this way, and no tolerance claim should be read
past the first turning point.

## 5. Genie confirmation

Replacing `h_hat` with a perfectly-tracked channel — as a *relative*
correction, so the receiver's static calibration cancels and only aging
is isolated — at iv=32 (`--part genie`):

| equalizer | delta_f | CRC | mean EVM | EVM@start | EVM@end |
|---|---|---|---|---|---|
| normal (self-check) | 0 | PASS | 0.165 | 0.164 | 0.179 |
| genie (self-check) | 0 | PASS | 0.165 | 0.164 | 0.179 |
| normal | 400 | FAIL | 0.270 | 0.173 | 0.361 |
| **genie** | **400** | **PASS** | **0.171** | 0.170 | 0.185 |
| normal | 667 | FAIL | 0.310 | 0.177 | 0.346 |
| **genie** | **667** | **PASS** | **0.172** | 0.171 | 0.187 |

The `delta_f = 0` self-check is what makes this trustworthy: the genie
ratio is identically 1 there and reproduces the normal path exactly. An
absolute closed-form `H_true` does **not** work as a genie, because
`h_hat` also absorbs the receiver's sync-offset phase ramp; two earlier
attempts failed on exactly this and the self-check is what caught it.

## 6. The one-sample sync offset (separate issue)

Independent of Doppler. Static channel, forced FFT-window offset
(`--part sync`); negative puts the window EARLY, inside the CP:

| offset | data EVM | **EVM with no noise at all** |
|---|---|---|
| -4 | 0.073 | 0.0017 |
| -1 | 0.073 | 0.0017 |
| 0 | 0.073 | 0.0017 |
| **+1** | **0.163** | **0.1497** |
| +2 | 0.190 | 0.1801 |
| +4 | 0.256 | 0.2494 |

The noiseless column is the proof: at +1 with zero noise EVM is 0.1497,
at -4 it is 0.0017.

`H_hat` is **not** failing to absorb the timing ramp. Its measured phase
slope matches `2*pi*dN/N` within 3-6% at every offset, including the
failing positive ones. What it cannot absorb is the leakage from a LATE
window, which runs past the symbol's last sample into the *next* symbol
and therefore depends on the neighbouring symbol's data. That is not a
per-subcarrier transfer function, so no channel estimate — DMRS-refreshed
or not — can represent it. **A cyclic prefix buys 64 samples of freedom
in the early direction and zero in the late direction.**

Why the synchronizer lands there: Schmidl-Cox tracks the energy centroid,
and a +1-sample echo pulls it right. At 40 dB the peak is sharp enough to
land on 0 (EVM 0.013); at 25/30 dB noise tips it to 1.

At 16QAM this consumes a large share of a ~0.17 budget before Doppler is
considered. It is not what loses the frames above — the genie passes at
EVM 0.175 with the same offset present — but it is free margin.

## Recommendation

**No change is proposed yet.** What the data supports, for whoever picks
this up:

1. **512 us is not the 4x shortfall an earlier reading suggested.** It
   covers `delta_f` to ~300 Hz. Whether that covers the deployment is a
   channel-modelling question this study cannot answer, and answering it
   needs a real differential-Doppler measurement, not a tighter model.
2. **If a high-Doppler mode is wanted, 256 us is the better buy** —
   ~500 Hz for 11% overhead, against ~667 Hz for 20% at 128 us.
3. **2048 us is dead weight.** 1024 us already fails at 300 Hz. The four
   2-bit codes could be re-pointed to {off, 256, 512, 1024 us} to gain a
   high-Doppler mode with **no wire-format width change** — but this
   breaks interoperability with any already-deployed build and must not
   be done silently.
4. **`MAX_PAYLOAD_SYMBOLS=128` blocks both short intervals at this
   payload** (150 and 135 slots). Any such mode needs that guard revisited
   together with the airtime budget, not raised in isolation.
5. **The sync offset and hard-decision Viterbi are separate, cheaper
   levers** than changing the DMRS format, and neither has been costed.

Everything above is simulation against a single rotating tap. The
Pi/Pluto mobility measurement named in the PHY specification remains the
completion gate and is not replaced by any of this.
