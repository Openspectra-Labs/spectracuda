# Real-world validation, on real RF

Everything below happened on two physically separate Raspberry Pi 5 +
ADALM-PLUTO rigs talking over the air — not simulation, not a wired
loopback. This page pulls forward the headline results; the full session
records (every bug found, every measurement, every open question) are
linked at the bottom of each section.

## The FEC chain has never been the failure point

Across an entire characterization session — every frequency tested, all
three modulations (QPSK/16-QAM/64-QAM), thousands of packets — **zero CRC
failures were ever observed on a frame that reached decode.** Every
measured packet loss traced back to the sync/capture stage (a frame never
being found in the first place), never to a frame being found and decoded
wrong. That's the concatenated `rs_m8`+`conv_v27` FEC, the LS channel
estimator, and the MMSE equalizer holding up under real RF, real
multipath, and real interference — not just passing simulated-channel
tests.

## What actually limits capture rate: RF interference, not code

Root-caused, not guessed: with the lab environment's **20+ mesh WiFi
devices operating 10–20cm from the hardware**, the receiver's Schmidl-Cox
sync detector was firing **~650–700 false triggers per second** against
~80–90 real decoded frames in the same 30-second window — each false
trigger occupies the receiver's header-wait state long enough to
structurally miss a real preamble arriving during that window. Confirmed
architecturally: Schmidl-Cox is a pure self-correlation detector ("do two
halves of this signal look alike"), and 802.11 WiFi's own preamble has
that same repeated-halves structure — so the detector genuinely can't
tell this system's own preamble from a nearby WiFi one. Left open,
honestly: a cross-correlation confirmation stage against this system's own
specific preamble sequence is the real fix, proposed but not yet built.

Frequency selection, with zero pipeline changes, is the biggest lever
available today:

| Band | Center of band | Band edge |
|---|---|---|
| 2.4 GHz ISM (QPSK) | ~76–88.5% capture (2.425 GHz) | ~91–99% capture (2.402/2.483/2.525 GHz) |
| 5 GHz ISM (64-QAM) | 66% capture (5.785 GHz, busy WiFi ch157) | **95.6% capture** (5.860/5.870 GHz) |

That 5GHz jump — 66% → 95.6%, identical code, identical 64-QAM — came
purely from picking a quieter 4MHz slice of spectrum. Modulation order
matters too, and its cost is frequency-dependent, not fixed: at 2.402GHz,
QPSK→64-QAM only costs ~9 points of capture rate; at 2.483GHz, the same
step costs ~30 points. There's no single "best frequency" independent of
the modulation scheme running on it.

Getting onto 5GHz at all took unlocking the Pluto's stock firmware cap
(3.8GHz max out of the box) via the standard community register-override
trick — real, documented, reproducible, not a vendor-supported path.

Full session record, every number, every bug fixed along the way (TX/RX
FEC-scheme mismatch, a streaming-buffer boundary bug that structurally
dropped any preamble straddling a chunk edge, a libiio teardown segfault):
{doc}`2026-09-06-rx-packet-loss-and-ism-band-characterization`.

## Acknowledged-mode (ARQ) retransmission

The MAC layer's `mode="am"` retransmission path has real two-node
hardware test scripts (`abhi/mac_am_node_a_test.py`/`mac_am_node_b_test.py`)
built specifically to exercise ARQ over a real full-duplex RF link — with
self-interference between this node's own simultaneous TX and RX
pre-characterized first (~0.0–0.06dB delta between TX-on and TX-off
receiver noise floor, at both -10dBm and 0dBm — no measurable
self-interference at either power level). That's the honest state: the
harness and the pre-check are real and done; a full recorded run and
result aren't in yet.

## Native FEC acceleration on real ARM hardware

The 8-lane NEON Viterbi kernel described in
{doc}`fec-c-lib-acceleration` isn't a paper design — it was iterated on
directly against a real Pi 5, including a first attempt that measured as
a genuine **2.1x regression** (kept in the source, correctly never wired
into the dispatch chain) before a wider rewrite got it to a real,
measured win. Full session record:
{doc}`2026-08-27-neon-viterbi-and-rx-throughput`.

```{toctree}
:hidden:

2026-09-06-rx-packet-loss-and-ism-band-characterization
2026-08-27-neon-viterbi-and-rx-throughput
```
