# Open issue: DMRS costs 3-6% packet delivery when the channel is NOT changing

Status: **measured and reproducible, cause NOT identified.** Written
2026-09-20 for whoever picks this up next.

Related: `docs/2026-09-20-dmrs-periodic-channel-refresh-plan.md` (the
feature itself, steps 1-6 implemented on branch
`feat/dmrs-periodic-channel-refresh`).

---

## The one-paragraph version

DMRS re-measures the radio channel partway through a long frame, so the
receiver stops equalizing against a stale estimate. When the channel
really is changing it works enormously well — it is the difference
between 0% and 90% of packets getting through. But when the channel is
**static**, turning DMRS on makes things slightly *worse*: about 3-6
fewer packets out of every 100. It should be free in that case, and it
is not. Nobody has explained why yet.

This is not blocking. DMRS is off by default and the cost only appears
where DMRS has nothing to do. But it means DMRS is insurance with a
premium, not a free win, and the reason matters before we recommend
turning it on broadly.

---

## Background: what DMRS does, in plain terms

An OFDM receiver has to know what the radio channel did to the signal
before it can undo it. It learns that from a **training symbol** — a
known waveform at the start of every frame. Comparing what it received
against what it knows was sent gives `H[k]`, the channel's effect on
each subcarrier.

The catch: that measurement is taken once, at the start. Every payload
symbol in the frame is then corrected using it. On a short frame that is
fine. On a long frame — and especially if the radios are moving — the
channel has drifted by the end, and the receiver is correcting with
stale information.

**A DMRS is simply the training symbol sent again, partway through the
payload.** The receiver measures `H[k]` fresh from it and uses that
newer estimate for the symbols that follow. Nothing else about the PHY
changes: the per-symbol pilots keep doing their usual phase tracking in
between.

The interval is configurable — `OFF`, or a refresh every 16 / 32 / 64
data symbols — and travels in the frame header, so a receiver does not
need to be told.

---

## The measurement

Config: `fft_size=256, cp_len=32, n_data=216, qpsk`, FEC
`rs_m8 + conv_v27` with block interleaving, 22768-bit PDU, 10 MSps.
Numbers are **percentage of frames whose CRC passed**, so higher is
better.

### Where DMRS wins (channel changing)

Two-ray channel: direct path plus a half-strength echo one sample later,
whose phase rotates at `fd` Hz. That makes `H[k]` both
frequency-selective and time-varying — the notch moves across the band
while the frame is being sent.

`f_d * T` below is Doppler times frame duration: how much the channel
turns over during one frame. The frame here is 3.77 ms.

| fd | `f_d*T` | DMRS off | every 16 | every 32 | every 64 |
|---|---|---|---|---|---|
| 0 (frozen) | 0.000 | **95%** | 85% | 80% | 92% |
| 10 Hz | 0.038 | **95%** | 85% | 80% | 92% |
| 25 Hz | 0.094 | **95%** | 85% | 80% | 92% |
| 50 Hz | 0.188 | **92%** | 85% | 80% | 92% |
| 100 Hz | 0.377 | **0%** | 85% | 82% | 92% |
| 200 Hz | 0.754 | **0%** | 85% | 82% | 0% |

40 trials per cell. Two things to read off this:

1. DMRS-off is healthy until `f_d*T ~ 0.19`, then **falls off a cliff**
   rather than degrading gradually. That cliff sits almost exactly at
   the textbook `f_d*T ~ 0.25` guideline, which is a nice independent
   confirmation we did not plan for.
2. DMRS-on is flat all the way across. That is the feature working.

### Where DMRS loses (channel frozen)

Same echo, `fd = 0`, so the channel never changes and there is nothing
for a refresh to discover. 150 trials per cell, `+-` is one standard
error.

| SNR | DMRS off | every 16 | every 32 | every 64 |
|---|---|---|---|---|
| 16 dB | **79%**+-3.3 | 77%+-3.4 | 75%+-3.5 | 83%+-3.1 |
| 18 dB | **87%**+-2.8 | 83%+-3.0 | 79%+-3.3 | 85%+-2.9 |
| 20 dB | **92%**+-2.2 | 87%+-2.7 | 86%+-2.8 | 89%+-2.5 |
| 24 dB | **97%**+-1.3 | 94%+-1.9 | 95%+-1.7 | 94%+-1.9 |

Each individual gap is only about 1-2 standard errors, so no single cell
proves anything. But **the sign is consistent across every interval and
every SNR**, which is what makes it look real rather than scatter.

This is the open issue.

---

## What it is NOT — two dead ends, already checked

Please do not spend time on these; both were proposed and disproven.

### Dead end 1: "the training estimate is averaged, the DMRS is not"

`_decode_header_from_sync` does average its channel estimate over
`n_training_symbols`. But **`n_training_symbols` defaults to 1**
(`pipeline/ofdm.py`), so the loop runs once and divides by one. There is
no averaging in the default configuration, and none in any measurement
above. Both estimates come from exactly one OFDM symbol.

### Dead end 2: "the DMRS estimate is lower quality"

It is not. `_estimate_channel_from_dmrs()` runs the **same** demod ->
known-subcarrier-extract -> `channel_estimator` sequence the training
symbol uses, over all 224 used subcarriers (216 data + 8 pilot
positions of the known symbol) — not just the 8 pilots. Measured mean
squared error against the true `H[k]` on a frozen channel at 20 dB:

```
training-symbol estimate : 1.1e-2 .. 1.3e-2
DMRS estimate            : 1.3e-2 .. 3.7e-2
```

Modestly worse, not enough to explain a 3-6 point delivery gap.

**Warning about how to measure this.** A first attempt reported the DMRS
estimate as 118x worse. That was a broken diagnostic, not a real result.
Comparing a DMRS estimate against a *static* analytic `H[k]` is invalid:
the DMRS correctly captures accumulated residual-CFO phase, which the
static reference does not contain, so the comparison punishes the DMRS
for being right. The giveaway was error growing monotonically with DMRS
index within a frame. Compare against the training-symbol estimate from
the same frame instead, or subtract the common phase first.

---

## Leading hypothesis (NOT verified)

With DMRS off, the whole frame is corrected using **one** channel
estimate. That estimate has some error, but the error is the *same* for
every symbol — a fixed bias. The per-symbol pilot CPE correction tracks
continuously against that single unchanging reference.

With DMRS on, the frame is corrected using **N+1 independent**
estimates. Each has its own independent error, and each time a DMRS
lands, the CPE correction's reference resets to a newly noisy one. So
DMRS trades:

- one systematic error **plus** growing staleness drift, for
- N independent errors **and** no drift.

When there is drift to remove, that is a huge win. When there is none —
a frozen channel — you pay the N independent errors for nothing.

This fits the data (cost only at `fd = 0`, large benefit at high `fd`),
but it has not been tested.

## How to test it

Inject a **known-perfect** `H[k]` at the DMRS slots instead of measuring
one — i.e. bypass `_estimate_channel_from_dmrs()` and hand the decoder
the true channel. Then re-run the frozen-channel table above.

- If the 3-6% gap **disappears**, the cost is estimation noise in the
  refresh, and the hypothesis is right. The fix would then be to smooth
  or partially average the DMRS estimate against the previous one
  (an alpha filter) rather than replacing it outright — which would keep
  the tracking benefit and drop the noise penalty.
- If the gap **remains**, the cost is structural — something about the
  interaction between per-segment estimates and the CPE correction, or
  the slot layout itself — and the hypothesis is wrong.

Either answer is worth having, and the experiment is maybe an hour.

---

## Does this matter operationally?

Less than the table suggests, for two reasons.

**The cost only appears where you would not turn DMRS on.** The default
is `OFF`, and the cliff analysis says a normal ~1 ms transmission
opportunity does not need DMRS at any realistic UAV speed:

| frame | tolerable Doppler | at 2.4 GHz | at 5.8 GHz |
|---|---|---|---|
| ~1 ms TXOP (31 symbols) | ~250 Hz | 112 km/h | 46 km/h |
| 3.77 ms max-length frame | ~66 Hz | 30 km/h | 13 km/h |

So DMRS is for long frames and fast platforms, which is exactly when the
3-6% cost is swamped by the benefit.

**The measurement is simulated.** Everything above uses a channel model
written for these tests, not the real radio. Whether our actual link
sees enough channel variation for any of this to matter is step 7 of the
plan, on the Pi-5 / two-Pluto rig. It cannot be measured on the WSL2 dev
machine (all USB is tunnelled over TCP).

Current recommendation until the cause is known: **leave DMRS off unless
frames are long or the platform is moving fast.**

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
| tests | `tests/test_ofdm_dmrs_*.py`, `tests/test_framing_dmrs.py` |
