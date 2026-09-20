# Resolved: the apparent DMRS static-channel cost was a simulation artifact

**Status: resolved 2026-09-20. No static-channel DMRS packet-delivery penalty
was reproduced after correcting the experiment.**

Related:

- `docs/2026-09-20-dmrs-periodic-channel-refresh-plan.md` — the feature;
- `tests/test_ofdm_dmrs_rx.py` — controlled time-varying-channel validation;
- `docs/2026-09-06-rx-packet-loss-and-ism-band-characterization.md` — real
  Pi/Pluto packet-loss investigation.

## Summary

An earlier simulation appeared to show that enabling DMRS reduced delivery by
roughly 3–6 percentage points when the channel was not changing. That result
was not caused by channel-estimation quality or by DMRS equalization.

The simulated capture ended exactly at the last transmitted sample. Multipath
occasionally made synchronization select a start one sample late. The receiver
then needed one sample beyond the end of the supplied array and correctly
rejected it as truncated. The experiment counted this extraction failure as a
failed packet even though payload decoding was never attempted.

DMRS intervals produced different frame lengths. A second measurement flaw
made the same random seed generate different preamble noise for those lengths,
so each interval experienced a different number of one-sample-late timing
decisions. Those differing truncation counts looked like a DMRS-dependent
packet-delivery cost.

Appending realistic trailing capture samples removed the effect. Across the
full frozen-channel matrix—OFF/16/32/64 at 16/18/20/24 dB—all 2,400 frames
passed CRC.

**Conclusion: there is no evidence that refreshing `H[k]` harms static-channel
packet delivery under the stated test configuration.**

## Background: what DMRS does

An OFDM receiver sends a known training symbol near the start of every frame.
Because the receiver knows what that symbol should look like, it can measure
the channel response `H[k]` on each subcarrier and undo the channel's effect on
the following payload.

That initial measurement can become stale during a long frame if the radios or
reflectors move. A DMRS is the same known training waveform transmitted again
inside the payload region. Data after that DMRS uses the refreshed estimate.

DMRS can therefore help when all of the following are true:

1. the receiver detected the frame;
2. the header was decoded;
3. the payload arrived;
4. the frequency-selective channel changed enough during the frame that the
   initial `H[k]` was no longer accurate.

DMRS cannot recover a frame whose preamble was missed, a capture dropped by
the host, or a receiver distracted by an unrelated Wi-Fi preamble. It also
does not repair a frame that never reached payload decoding.

## Original observation

The original experiment used:

```text
fft_size=256
cp_len=32
n_data=216
n_pilot=8
QPSK
RS + convolutional FEC with block interleaving
22,768-bit PDU
10 MSps
two-ray channel: direct path + half-strength one-sample-delayed echo
```

The frozen-channel table appeared to show a small delivery difference between
DMRS OFF/16/32/64. Individual differences were close to their sampling error,
and the signs were not actually uniform: interval 64 at 16 dB performed better
than OFF. The original description overstated the consistency and precision of
that evidence.

More importantly, the table combined payload failures with frames rejected as
truncated before payload decoding. That made it unsuitable for diagnosing
channel-estimator behavior.

## Root cause

### 1. The capture ended flush with the frame

The impairment function returned an IQ array with exactly the same length as
the transmitted frame. There were no samples after the last OFDM symbol.

In a real streaming capture, samples normally continue after a frame. A
bounded simulation must model that by including a trailing guard or subsequent
stream samples.

### 2. Multipath sometimes moved timing one sample later

The synchronizer occasionally selected `start_index=1` instead of zero. A
one-sample residual timing offset is normally harmless because it is inside the
32-sample cyclic prefix.

It became fatal only because the artificial input buffer ended at the exact
frame boundary:

```text
correct start: receiver asks for samples 0 ... last
late by one:   receiver asks for samples 1 ... last+1
                                               ^ absent
```

The receiver's bounds check raised `ValueError` for a truncated frame. This was
reported as a packet failure, but CRC decoding had not run.

### 3. “Same seed” did not mean the same preamble noise

The simulation generated noise using two consecutive calls:

```python
rng.standard_normal(rx.shape)       # all real samples
rng.standard_normal(rx.shape)       # all imaginary samples
```

DMRS changes the array length. The second call therefore begins at a different
position in the random-number stream for each interval. Even with the same
seed, different-length frames receive different imaginary noise at the
preamble.

That changed how often the synchronizer selected sample zero versus sample
one. The resulting interval-dependent bounds failures were mistaken for
interval-dependent decoding performance.

## Decisive experiments

### Paired boundary experiment

At 18 dB, 150 trials per interval without trailing samples produced:

| Interval | CRC pass | Truncation before payload decode |
|---:|---:|---:|
| OFF | 115 | 35 |
| 16 | 132 | 18 |
| 32 | 118 | 32 |
| 64 | 116 | 34 |

Every failure corresponded to the one-sample-late boundary condition.

After appending 1,024 trailing zero samples, the same experiment produced
150/150 passes at every interval.

### Complete frozen-channel rerun

With trailing samples included, 150 trials were run for every interval at each
SNR:

| SNR | OFF | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|
| 16 dB | 150/150 | 150/150 | 150/150 | 150/150 |
| 18 dB | 150/150 | 150/150 | 150/150 | 150/150 |
| 20 dB | 150/150 | 150/150 | 150/150 | 150/150 |
| 24 dB | 150/150 | 150/150 | 150/150 | 150/150 |

Total: 2,400/2,400 CRC-valid frames.

### Same received frame, different estimate policy

To separate estimate replacement from slot layout, the exact same received
DMRS-bearing IQ frame was decoded four ways:

1. normal refreshed estimates;
2. reuse the initial training estimate for all segments;
3. phase-align each refreshed estimate to the initial estimate;
4. average each phase-aligned refresh 50/50 with the initial estimate.

At 18 dB all four policies passed 80/80 frames. Mean EVM was:

| Estimate policy | Mean EVM |
|---|---:|
| normal DMRS refresh | 0.21796 |
| reuse initial `H[k]` | 0.21822 |
| 50% averaging | 0.19720 |

Normal refresh and reuse were effectively equal for packet delivery and EVM.
This directly rules out the earlier claim that ordinary refreshed estimates
caused the observed delivery gap.

The averaging result may be useful future estimator work, but it is not needed
to resolve this issue and should not be promoted without time-varying-channel
tests showing that smoothing preserves tracking performance.

## What remains valid from the Doppler experiment

The controlled two-ray tests still demonstrate the intended mechanism:

- without refresh, late-frame EVM rises when the frequency-selective channel
  changes quickly;
- with refresh, late-frame EVM remains approximately flat;
- a frame that fails CRC without DMRS can pass with DMRS;
- denser intervals track faster changes better.

Those conclusions are also covered by focused tests that include trailing
samples. However, the earlier PER percentages and claimed location of the
Doppler cliff must be rerun with:

- trailing capture samples;
- paired noise that is identical over common sample positions;
- separate counters for sync miss, header failure, truncation, CRC failure,
  and successful delivery;
- finer Doppler points if a threshold such as `f_d*T = 0.25` is to be claimed.

The old sample grid only showed that the OFF-path transition occurred somewhere
between tested points; it did not precisely establish a 0.19–0.25 threshold.

## Resolution

No DMRS estimator change is required for this issue.

Simulation and future measurement code should:

1. append enough trailing samples to cover synchronization uncertainty and
   channel delay;
2. generate one maximum-length complex noise realization and slice it for
   different frame lengths, or use separate deterministic real/imaginary RNG
   streams whose common prefix does not depend on array length;
3. use paired payload and noise seeds across OFF/16/32/64 comparisons;
4. classify failure stage rather than treating every exception as a CRC fail;
5. compare channel estimates only after accounting for common phase from
   residual CFO.

The one-sample late-start/end-of-buffer behavior remains separate receiver/test
harness technical debt. It is not evidence of DMRS degradation.

## Operational meaning for the real link

DMRS is not a general cure for packet loss. It addresses one specific failure:
the receiver found the frame, but the channel estimate became stale before the
end of a long payload.

There is strong hardware evidence that SpectraCUDA previously had an in-frame
aging problem. Before per-symbol pilot tracking, approximately 90% of 64-byte
packets arrived, while only about 5–10% of 1,024/2,048-byte packets arrived;
128/246-byte packets were already losing roughly 40–50%. After pilot tracking,
2,048-byte delivery rose to about 95% in a clean channel and 75–80% in a noisy
channel. The dependence on frame duration, followed by the large pilot-tracking
improvement, is compelling evidence that accumulated error within the frame
was real.

That result identifies the dominant old error more specifically than “bad
`H[k]`”: the pilots correct a common phase rotation shared by subcarriers,
normally residual CFO. Periodic DMRS complements them by refreshing the
frequency-dependent channel shape—relative gain and phase across subcarriers.
DMRS may improve the remaining long-frame loss if that shape changes, but it
cannot improve the part already caused by acquisition failure or unrelated
interference.

The existing Pi/Pluto characterization found that much of the observed loss in
the lab occurred earlier: nearby Wi-Fi repeatedly triggered the Schmidl-Cox
detector, causing genuine SpectraCUDA preambles to be missed. DMRS is inside the
payload, so it cannot help a receiver that never acquired the frame.

For an observed 25% loss rate on a roughly 3 ms frame, determine the failure
stage before predicting a DMRS gain:

| Observation | Likely failure | Will DMRS help? |
|---|---|---|
| no frame/header result | preamble missed, false trigger, capture loss | **No** |
| frame found, CRC failure grows toward frame end | stale `H[k]` | **Likely** |
| frame found, similar errors throughout payload | low SNR/interference/static fade | Usually not |
| late-frame EVM improves with DMRS | channel aging confirmed | **Yes** |

The long-packet history justifies testing DMRS on the real link rather than
dismissing it as an acquisition-only problem. Interval 32 is the natural first
setting at 10 MSps. Split results into acquisition success and CRC success, and
compare early- versus late-frame EVM: an unchanged detection rate together
with improved late-frame EVM/CRC delivery is the signature that DMRS is
recovering the remaining frequency-selective channel aging.

## Code map

| Purpose | Location |
|---|---|
| slot arithmetic | `spectracuda/framing/dmrs.py` |
| DMRS channel estimate | `Ofdm._estimate_channel_from_dmrs()` |
| per-segment expansion | `Ofdm._expand_over_segments()` |
| payload equalization | `Ofdm._decode_payload_from_header()` |
| initial training estimate | `Ofdm._decode_header_from_sync()` |
| header field | `spectracuda/framing/header.py` |
| focused tests | `tests/test_ofdm_dmrs_*.py` |
