# RESOLVED: the "DMRS costs 3-6% on a static channel" finding was a measurement artifact

Status: **resolved 2026-09-20. There is no static-channel penalty.** The
original claim in this file was wrong; it came from two flaws in the
simulation harness, not from the receiver. Corrected numbers below.

Related: `docs/2026-09-20-dmrs-periodic-channel-refresh-plan.md` (the
feature, steps 1-6 implemented on branch
`feat/dmrs-periodic-channel-refresh`).

---

## Summary

An earlier version of this document reported that enabling DMRS cost
3-6 percentage points of packet delivery on a static channel, and
proposed an unverified estimator-noise hypothesis to explain it.

Both the finding and the hypothesis were wrong. The measurement harness
had two defects that together manufactured an apparent,
interval-dependent packet error rate out of nothing. With both fixed,
**every DMRS interval delivers 600/600 frames on a frozen channel** at
16/18/20/24 dB.

The high-Doppler benefit is real and, once measured correctly, is
cleaner and larger than what was originally published.

---

## The two harness defects

### 1. Frames ended flush with the simulated array

Multipath sometimes moves the detected frame start one sample late. The
simulated IQ array ended exactly at the last transmitted sample, so with
`start_index = 1` the payload extraction needed one sample past the end,
and the bounds check in `_decode_payload_from_header()` raised.

The sweep counted that raise as a failed packet — but the payload decode
was never attempted. It was a simulation boundary, not a decode failure.

A real capture always contains samples after a frame, so this cannot
happen on the air.

At 18 dB, 150 trials with no trailing samples:

| interval | passed | bounds failures |
|---|---|---|
| OFF | 115 | 35 |
| 16 | 132 | 18 |
| 32 | 118 | 32 |
| 64 | 116 | 34 |

Every single failure was the one-sample timing offset. Appending 1024
trailing zeros: 150/150 for every interval.

### 2. The noise depended on the frame length

The channel helper generated complex noise as:

```python
rx + s * (rng.standard_normal(rx.shape) + 1j * rng.standard_normal(rx.shape))
```

Two separate draws. The second starts wherever the first ended, so **the
imaginary part depends on the array length**:

```
len=100: real [ 0.3047 -1.04    0.7505]   imag [-0.3782  1.2992 -0.3563]
len=101: real [ 0.3047 -1.04    0.7505]   imag [ 1.2992 -0.3563  0.7375]
                                                 ^ shifted by one draw
```

Different DMRS intervals produce different frame lengths, so the same
seed handed each interval **different noise on the preamble**. Different
preamble noise means different sync behaviour, which means a different
number of one-sample-late timing estimates per interval — which is
exactly the interval-dependent PER that looked like a DMRS cost.

The two defects compound: defect 2 varies how often defect 1 fires.

---

## Corrected results

Methodology now: trailing capture samples appended; paired complex noise
drawn once at a fixed maximum length and sliced, so every interval sees
identical noise; bounds/extraction failures counted separately from CRC
failures.

Config: `fft_size=256, cp_len=32, n_data=216, qpsk`, `rs_m8 + conv_v27`
with block interleaving, 22768-bit PDU, 10 MSps.

### Frozen channel — no penalty

150 trials per cell, zero bounds failures throughout.

| SNR | OFF | every 16 | every 32 | every 64 |
|---|---|---|---|---|
| 16 dB | 100% | 100% | 100% | 100% |
| 18 dB | 100% | 100% | 100% | 100% |
| 20 dB | 100% | 100% | 100% | 100% |
| 24 dB | 100% | 100% | 100% | 100% |

600/600 for every interval. **DMRS is free when there is nothing to
track.**

Cross-check on the same received frames, decoded four ways at 18 dB —
normal refresh, initial estimate reused throughout, phase-aligned
refresh, and 50% averaging — all passed 80/80, with normal refresh and
reuse showing essentially identical EVM (0.21796 vs 0.21822). No
estimator-quality difference to find.

### Doppler sweep — the benefit, measured properly

Two-ray channel, half-strength echo one sample late, echo phase rotating
at `fd`. SNR 20 dB, 150 trials, zero bounds failures.

| fd | `f_d*T_frame` | OFF | every 16 | every 32 | every 64 |
|---|---|---|---|---|---|
| 0 | 0.000 | 100% | 100% | 100% | 100% |
| 25 Hz | 0.094 | 100% | 100% | 100% | 100% |
| 50 Hz | 0.188 | 99% | 100% | 100% | 100% |
| 75 Hz | 0.283 | **0%** | 100% | 100% | 100% |
| 100 Hz | 0.377 | **0%** | 100% | 100% | 93% |
| 150 Hz | 0.565 | **0%** | 100% | 100% | **0%** |
| 200 Hz | 0.754 | **0%** | 100% | 98% | **0%** |

Cleaner than the contaminated version: DMRS-on holds at 100% exactly
where DMRS-off collapses to 0%, with no cost anywhere.

### Every cliff lands on `f_d * T_refresh ~ 0.25`

The most useful thing to come out of the corrected run. Each setting
fails when the channel turns over by about a quarter cycle *between its
own refreshes* — not between frames:

| setting | refresh period | predicted cliff | observed |
|---|---|---|---|
| OFF | 3.77 ms (whole frame) | 66 Hz | between 50 and 75 Hz |
| every 64 | 1.84 ms | 136 Hz | between 100 and 150 Hz |
| every 32 | 0.92 ms | 271 Hz | still 98% at 200 Hz |
| every 16 | 0.46 ms | 543 Hz | still 100% at 200 Hz |

Four independent cliffs, all consistent with the standard `0.25`
guideline. This is a measured confirmation of the interval ladder,
arrived at without relying on any external spec's assertion.

---

## What this means for using DMRS

Simpler than before, because the trade-off it was hedging against does
not exist:

- **DMRS is free when the channel is static.** No reason to avoid it.
- **DMRS is the difference between a working and a dead link** once
  `f_d * T_refresh` exceeds about 0.25.
- Pick the interval so `T_refresh` stays under `0.25 / f_d` for the
  fastest closing speed you need to support.

For context, at a ~1 ms transmission opportunity the whole frame is
already inside the budget up to ~250 Hz Doppler (112 km/h at 2.4 GHz,
46 km/h at 5.8 GHz), so short frames still do not need DMRS. It is long
frames and fast platforms that do.

Still simulated. Whether the real link sees enough channel variation for
this to matter is step 7 of the plan, on the Pi-5 / two-Pluto rig; it
cannot be measured on the WSL2 dev machine, which tunnels all USB over
TCP.

---

## Remaining real technical debt (separate from DMRS)

**The receiver's one-sample-late timing under multipath is real**, and
independent of DMRS — it reproduces with `dmrs_interval=0`. It is benign
on the air because a real capture has trailing samples, so it does not
imply any DMRS regression. But it is worth understanding on its own:
whether the sync peak is genuinely landing a sample late under
multipath, or whether the frame-start convention is off by one.

It is not tracked here; this file is about the DMRS measurement.

---

## Lessons for future PHY simulation work

These cost a day and are easy to repeat:

1. **Always append trailing capture samples.** A frame ending flush with
   the array turns an ordinary sync offset into a fake packet loss.
2. **Generate paired complex noise independently of frame length.**
   `standard_normal(n)` twice couples the imaginary part to `n`. Draw
   once at a fixed maximum length and slice, or use
   `standard_normal((2, n))`. Otherwise any change that alters frame
   length silently changes the noise realization, and A/B comparisons
   are invalid.
3. **Count extraction/bounds failures separately from CRC failures.** A
   harness that lumps "the decoder raised" in with "the CRC failed"
   cannot tell a real regression from a harness bug. An earlier sweep in
   this same effort printed all zeros for a completely unrelated reason
   (an oversized PDU) because a bare `except Exception: pass` swallowed
   the `ValueError`.
4. **Sanity-check A/B comparisons against a control** where the effect
   under test is switched off by construction. Here, `fd = 0` with
   DMRS-off passing 100% would have exposed the artifact immediately.

The test helper in `tests/test_ofdm_dmrs_rx.py::time_varying_two_ray`
now implements 1 and 2, with the reasoning inline so it does not get
"simplified" back.

---

## Where the code is

| what | where |
|---|---|
| slot arithmetic (pure, no OFDM) | `spectracuda/framing/dmrs.py` |
| DMRS estimate from a slot | `Ofdm._estimate_channel_from_dmrs()` |
| per-segment estimate expansion | `Ofdm._expand_over_segments()` |
| where it is applied | `Ofdm._decode_payload_from_header()` |
| training estimate for comparison | `Ofdm._decode_header_from_sync()` |
| header field (2 bits, byte 5 [6:5]) | `spectracuda/framing/header.py` |
| corrected channel helper | `tests/test_ofdm_dmrs_rx.py` |
| tests | `tests/test_ofdm_dmrs_*.py`, `tests/test_framing_dmrs.py` |
