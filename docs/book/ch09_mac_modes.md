# Chapter 09 — TM, UM, AM

Everything so far has been one PHY, one frame, one call. Real payloads
don't fit in one frame, and a real link needs to know whether "best
effort" or "guaranteed delivery" is what a given stream actually needs —
that's what the MAC layer above `Ofdm` exists for: segmentation,
reassembly, sequence numbering, and (for one of the three modes) genuine
retransmission. Named after 3GPP RLC's TM/UM/AM modes for the *behavior*
they describe, not as a spec-accurate implementation — this project's own
32-bit PDU header, not a real 3GPP bit layout.

## The 32-bit PDU header

```{list-table}
:header-rows: 1

* - Field
  - Width
  - Meaning
* - `TYPE`
  - 3 bits
  - `DATA`(0) / `STATUS`(1) / `BIND_REQUEST`(2) / `BIND_RESPONSE`(3) / `LINK_QUALITY`(4)
* - `SI`
  - 2 bits
  - Segmentation indicator: `FULL`(0, unsegmented) / `FIRST`(1) / `MIDDLE`(2) / `LAST`(3)
* - `SN`
  - 10 bits
  - Sequence number, modulo 1024 — real modular arithmetic, not a naive `a < b`, so it survives wraparound
* - `SO`
  - 16 bits
  - Segment offset **in bits** from the start of the original SDU
* - `RESERVED`
  - 1 bit
  - Zero, rounds the header to a clean 32 bits
```

Every PDU type — data, control, everything — shares this one header
rather than each type inventing its own format; type-specific payload (a
STATUS pdu's ACK/NACK bitmap, a BIND request's requested capacity)
follows immediately after these 32 bits. `SN`'s field is repurposed as a
STATUS report's window base for AM; `SI`/`SN`/`SO` are all just `0` for
control PDU types, which don't segment.

## Three modes, three delivery contracts

| Mode | `transmit`/`receive`-family methods | What it guarantees |
|---|---|---|
| `"tm"` (Transparent) | `transmit()`, `receive()` — **no header at all** | Raw passthrough, one call = one PDU, no segmentation |
| `"um"` (Unacknowledged) | `transmit()`, `receive()` | Segmentation + reassembly + sequence numbering, best-effort — a lost segment is just gone |
| `"am"` (Acknowledged) | `transmit()`, `receive_data()`, `build_status()`, `receive_status()` | Everything UM does, plus real ARQ retransmission of NACKed segments |

TM needs two methods because there's nothing to track. AM needs four
because retransmission means the receiving side has to report back
*which* segments arrived — `build_status()`/`receive_status()` are that
round trip, not present in TM or UM at all (Chapter 12 covers why that
round trip needs its own PHY direction, not just a different method
call).

## Segmentation, PHY-agnostic

`Mac(mode=, max_segment_bits=)` works with zero `Ofdm` involvement —
pure MAC logic, testable without any radio at all:

```python
import numpy as np
from spectracuda.mac import Mac

mac = Mac(mode="um", max_segment_bits=200)

sdu = np.random.default_rng(0).integers(0, 2, size=896).astype("uint8")  # must be byte-aligned
pdus = mac.transmit(sdu)
len(pdus)                 # 5 -- ceil(896 / 200)
[len(p) for p in pdus]    # [232, 232, 232, 232, 128] -- 32-bit header + up to 200 payload bits each

peer = Mac(mode="um", max_segment_bits=200)
reassembled = [peer.receive(p) for p in pdus]
np.concatenate([r for r in reassembled if r])   # -> bit-identical to `sdu`
```

```{note}
**Byte alignment is a real, enforced requirement, not an assumption.**
`segment()` raises `ValueError` on a non-byte-aligned SDU — matching
`Packetizer`'s own CRC byte-alignment requirement (Chapter 07). Pad your
raw SDU to a byte boundary before calling `transmit()`.
```

`peer.receive(pdu)` returns the reassembled SDU only once every segment
has arrived — `[]` otherwise, not an error, since "still waiting on more
segments" is completely normal for UM's best-effort contract. `SN` uses
real modulo-1024 comparison throughout reassembly (`sn_precedes()`, not a
plain `<`) specifically so a long-running link surviving sequence-number
wraparound doesn't silently misbehave the one time it actually matters.

Everything above is deliberately PHY-agnostic — `Mac(mode=,
max_segment_bits=)` never touches an `Ofdm` at all. Chapter 10 wires this
same object up to a real, independently-owned PHY on each side and sends
it over the air.
