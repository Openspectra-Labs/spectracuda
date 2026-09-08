# Chapter 10 — Two Independent Radios

Every `Mac` example so far, and the landing-page hero itself, has one
honest gap worth naming directly: is this actually two separate devices,
or one object quietly playing both roles? `Mac(mode=, ofdm_kwargs=)` is
what closes that gap for real — each side builds and owns its **own**
`Ofdm`, and the only thing that ever crosses between them is an IQ array,
exactly what a real antenna-to-antenna link would carry.

```python
import numpy as np
from spectracuda.mac import Mac

phy = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    modem="qpsk", fec="conv_v27", crc="crc32",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
)

hw1 = Mac(mode="um", ofdm_kwargs=phy)
hw2 = Mac(mode="um", ofdm_kwargs=phy)
hw1.ofdm is hw2.ofdm   # False -- always, by construction
```

Nothing is shared — no config object, no class instance, no in-process
shortcut. `send_iq()`/`receive_iq()` are thin wrappers around exactly
that boundary: `send_iq()` is `self._impl.transmit(sdu) ->
[self.ofdm.generate_frame(pdu) for pdu in pdus]`; `receive_iq()` is
`self.ofdm.rx_process(iq)` → the frame-found/CRC check → handed to
`self._impl.receive(bits)`. Everything above that line is pure,
PHY-agnostic MAC logic (Chapter 09's `TmEntity`/`UmEntity`/`AmEntity`);
everything at or below it is the real OFDM chain from Chapters 01–08.

## Binding is a real handshake, carried as real IQ

`send_iq()` refuses to run — `ValueError` — until both sides complete a
3-message exchange, itself transmitted and decoded through the same
`Ofdm` chain everything else uses, not a side-channel or a flag either
side just sets:

```python
req_iq = hw1.build_bind_request()            # encodes mode/capacity/window/retries
resp_iq = hw2.handle_bind_request_iq(req_iq)  # hw2's OWN accept/reject decision
accepted = hw1.handle_bind_response_iq(resp_iq)
hw1.bound, hw2.bound   # (True, True)
```

`handle_bind_request_iq()` evaluates the request against `hw2`'s own,
independently-derived capacity — it can genuinely *reject* a request
that asks for more than it can carry, not silently clamp it. Chapter 11
covers that evaluation and the link-quality reporting built on the same
pattern in full; this chapter is about the object model the handshake
runs on top of.

## The whole round trip

```python
sdu = np.random.default_rng(0).integers(0, 2, size=896).astype("uint8")
frames = hw1.send_iq(sdu)          # segmentation + real OFDM tx, one Ofdm call per PDU

delivered = []
for iq in frames:
    result = hw2.receive_iq(iq)    # real OFDM rx + CRC check + reassembly
    if result:
        delivered.extend(result)

np.concatenate(delivered)          # bit-identical to `sdu`
```

Each `send_iq()`/`receive_iq()` pair is a real, independent PHY round
trip — swap `hw1.ofdm`/`hw2.ofdm`'s `sync=`/`fec=`/`modem=` and nothing
above this line changes, and route the IQ through a real Pluto instead of
a plain Python handoff and nothing changes either (see
{doc}`../hardware-validation`). This is the object model every real
two-node example in this project — and the `MacLink` convenience wrapper
Chapter 09's earlier examples used — ultimately reduces to.

```{note}
**`MacLink` vs. this chapter's `Mac(mode=, ofdm_kwargs=)` — two
different tools, not competing answers.** `MacLink` (`session.py`) owns
**one shared** `Ofdm` and plays both `tx_mac`/`rx_mac` roles itself — a
convenience harness for exercising a link's *behavior* (bind, send, ARQ
retry counts) without standing up two genuinely separate objects. This
chapter's pattern is what a real two-device topology actually needs, and
what Chapters 11–12's binding/bidirectional-AM material builds on
directly.
```
