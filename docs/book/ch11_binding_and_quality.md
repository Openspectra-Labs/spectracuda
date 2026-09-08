# Chapter 11 — Binding & Link Quality

Chapter 10 showed a successful bind and moved on. This chapter is about
the two things that made it worth calling a real handshake rather than a
formality: it can genuinely *reject* a mismatched peer, proven through
the actual wire mechanism rather than assumed correct, and both sides
accumulate real link-quality statistics from every frame attempt — not
just successful ones — that can be reported back to the peer on demand.

## A rejection is a real decision, not a clamp

`evaluate_bind_request()` is a deliberately pure function — no `Ofdm`, no
`Mac`, nothing but a request dict and a local capacity number — precisely
so a genuine capacity mismatch can be tested directly, independent of
whether two live radios happen to agree by construction:

```python
from spectracuda.mac.bind import evaluate_bind_request

request = {"mode": "am", "max_segment_bits": 9000, "window_size": 16, "max_retries": 3}
decision = evaluate_bind_request(request, local_max_segment_bits=4000)
decision
# {'mode': 'am', 'max_segment_bits': 9000, 'window_size': 16, 'max_retries': 3,
#  'accepted': False, 'reason': 'segment_too_large'}
```

Notice what it does **not** do: silently clamp the request down to 4000
and proceed. A too-large request is rejected outright — "fail loud," the
same convention this project's FEC/LDPC decoders already hold to for
their own capacity/correctability limits (Chapter 06) — because a session
that silently started with parameters neither side actually agreed to
would just fail somewhere else, more confusingly, later.

Wired into the real handshake exactly this way: `handle_bind_request_iq()`
decodes the arrived request off real IQ, calls `evaluate_bind_request()`
against *its own* `max_segment_bits` (derived from its own `Ofdm`
config — never independently settable when `ofdm_kwargs=` is used, so
the two can never silently drift apart), and encodes whatever decision
comes back into the BIND_RESPONSE it sends.

## Link quality: every attempt counts, not just the successes

`self.quality` (a `LinkQualityTracker`) accumulates on *every* PHY frame
attempt this `Mac` makes or receives — successes and failures alike, RSSI
observed unconditionally even when no frame was found at all:

```python
hw1.quality.report_dict()
# {'n_attempts': 1, 'n_delivered': 1, 'mean_rssi_db': -25.13, 'mean_evm': 0.000999}
```

That report travels to the peer as a real PDU, over real IQ, the same
control-plane pattern as binding:

```python
report_iq = hw1.build_quality_report()          # already IQ -- no extra generate_frame() call needed
peer_view = hw2.handle_quality_report_iq(report_iq)
peer_view
# {'n_attempts': 1, 'n_delivered': 1, 'delivered_ratio': 1.0,
#  'mean_rssi_db': -25.13, 'mean_evm': 0.001}
```

`peer_view` is what actually survived the trip and got decoded on the
other end — not necessarily byte-identical to `hw1.quality.report_dict()`
if the quality-report frame itself experiences loss, in which case
`handle_quality_report_iq()` raises rather than fabricating a report. This
is the same real per-frame RSSI/EVM readout `Ofdm.rx_process()` returns
(Chapter 07) — the MAC layer isn't computing anything new here, just
aggregating and forwarding what the PHY already measured on every attempt.

```{note}
**Both control-plane exchanges — bind and quality-report — are real PDUs
over the shared 32-bit header** (Chapter 09): `TYPE=BIND_REQUEST/
BIND_RESPONSE/LINK_QUALITY`, `SI`/`SN`/`SO` all `0` since control PDUs
don't segment. Nothing about them is a side channel outside the normal
frame path.
```
