# Chapter 12 — Bidirectional AM

TM and UM are one-directional: `send_iq()` on one side, `receive_iq()` on
the other, done — Chapter 10's two-radio model covers them completely.
AM adds retransmission, and retransmission means the receiving side has
to send something real back: a STATUS pdu reporting which PDUs actually
arrived. That STATUS traffic needs its own PHY frame, physically
travelling the *opposite* direction — which means a full bidirectional
AM link needs **four** `Mac`/`Ofdm` objects, two per endpoint, one per
direction, not two.

```{list-table}
:header-rows: 1

* - Object
  - Owns
  - Direction
* - `hw1_tx` (`mode="am"`)
  - `Ofdm_A`
  - hw1 → hw2 (DATA)
* - `hw2_rx` (`mode="am"`)
  - `Ofdm_A'` — same config as `Ofdm_A`, separate instance
  - hw1 → hw2 (DATA)
* - `hw2_tx` (`mode="am"`)
  - `Ofdm_B`
  - hw2 → hw1 (STATUS)
* - `hw1_rx` (`mode="am"`)
  - `Ofdm_B'` — same config as `Ofdm_B`, separate instance
  - hw2 → hw1 (STATUS)
```

```python
import numpy as np
from spectracuda.mac import Mac

phy = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    modem="qpsk", fec="conv_v27", crc="crc32",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
)

hw1_tx, hw2_rx = Mac(mode="am", ofdm_kwargs=phy), Mac(mode="am", ofdm_kwargs=phy)
hw2_tx, hw1_rx = Mac(mode="am", ofdm_kwargs=phy), Mac(mode="am", ofdm_kwargs=phy)

for a, b in [(hw1_tx, hw2_rx), (hw2_tx, hw1_rx)]:      # bind both directions
    req = a.build_bind_request()
    resp = b.handle_bind_request_iq(req)
    a.handle_bind_response_iq(resp)
```

## DATA forward, over `Ofdm_A`

The forward direction is exactly Chapter 10's `send_iq()`/`receive_iq()`
pattern — AM's `transmit()`/`receive_data()` underneath, same as UM's
`transmit()`/`receive()`:

```python
sdu = np.random.default_rng(0).integers(0, 2, size=896).astype("uint8")
frames = hw1_tx.send_iq(sdu)

delivered = []
for iq in frames:
    result = hw2_rx.receive_iq(iq)
    if result:
        delivered.extend(result)

np.concatenate(delivered)   # bit-identical to `sdu`
```

## STATUS backward, over `Ofdm_B` — the wrinkle worth internalizing

`send_iq()`/`receive_iq()` deliberately don't cover this half — a status
report is control-plane traffic with its own physical direction, so it's
built and sent manually, same PDU-header machinery as Chapter 09/11:

```python
status_pdu = hw2_rx.build_status()                       # hw2_rx: knows what IT received
status_iq = hw2_tx.ofdm.generate_frame(status_pdu[None, :])  # sent over the REVERSE PHY

arrived = hw1_rx.ofdm.rx_process(status_iq)               # hw1_rx: owns the matching Ofdm_B'
arrived["frame_found"], arrived["crc_valid"]               # (True, [True])

retransmit = hw1_tx.receive_status(arrived["bits"][0])     # hw1_tx: holds the retry buffer
len(retransmit)   # 0 -- everything arrived, nothing to resend
```

The wrinkle: the STATUS pdu is **decoded** by `hw1_rx` — it owns
`Ofdm_B'`, the receive side matching `Ofdm_B` — but its *content* is only
meaningful to `hw1_tx`, the object that actually holds the retransmission
buffer for the DATA it originally sent. Four distinct roles, four
distinct objects, and the STATUS decode necessarily crosses between two
of them by hand — `receive_status()`'s return value is exactly the list
of PDUs `hw1_tx` needs to resend over `Ofdm_A` if that list is non-empty,
using the same `Ofdm.generate_frame()` call any other DATA PDU would.

```{note}
**This is real, working code, not simplified for the page.** The full
version — including a deliberately induced loss and the resulting
retransmission round, plus the equivalent scenario driven through
`rx_streaming()` instead of batch `rx_process()` — lives in
`examples/mac_bidirectional_am_batch_demo.py` and
`examples/mac_bidirectional_am_streaming_demo.py`. This chapter shows
the mechanism; those scripts show it under actual packet loss.
```

That's the whole book, PHY through MAC: one `Ofdm` object handling sync
through FEC (Chapters 01–08), one MAC layer built on top of it for
segmentation, reliability, and session control (Chapters 09–12) — every
piece independently swappable, every claim on this page backed by code
that actually runs. {doc}`../hardware-validation` is where the same
design meets real RF.
