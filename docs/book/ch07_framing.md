# Chapter 07 — Framing & the Header

A receiver is never told what a transmitter chose for `modem=`/`fec=`/
`crc=` ahead of time — it has to recover that from the frame itself,
exactly the way a real, physically separate receiving device has to.
That's the job of the 112-bit header every frame carries, and of
`rx_process()` returning a *defined* answer even when no frame is there
at all, not silently marching noise through the rest of the pipeline.

## The header, byte by byte

```{list-table}
:header-rows: 1

* - Bytes
  - Field
  - Width
* - 0
  - `protocol_version`
  - 8 bits
* - 1–2
  - `payload_len_bits` — raw bit count, pre-CRC/pre-FEC
  - 16 bits
* - 3
  - `mod_scheme` — the payload's modulation
  - 8 bits
* - 4
  - `crc` (3 bits) + `fec0` (5 bits)
  - 8 bits
* - 5
  - `fec1`
  - 8 bits
* - 6–13
  - `user_data` — 8 caller-defined bytes
  - 64 bits
```

`fec0`'s 5-bit field has room for 31 codes; `"none"`/`"conv_v27"`/
`"rs_m8"` take 0–2, and the 12 IEEE 802.11n LDPC variants take 3–14,
assigned in sorted order automatically — a new LDPC variant added to
`ldpc_tables.BASE_MATRICES` gets a wire code with no header-format change
needed. The header itself is BPSK-modulated onto its own dedicated
symbol(s), spread across subcarriers for frequency diversity, and
scrambled before transmission — unscrambled, its mostly-repeated field
content constructively interferes into a real, measured time-domain PAPR
spike, a bug found and fixed during development, not a defensive-only
measure.

## `frame_found`: a real null result, not noise marching forward

Every `sync=` strategy is a *best-candidate* search — it always returns
*some* `start_index`, even pointed at pure noise, because "best window
found so far" isn't the same question as "is a frame actually here."
Gating on that candidate's own sync metric against `sync_threshold=`
(default `Ofdm.DEFAULT_SYNC_THRESHOLD`, empirically calibrated against
both `SchmidlCoxSync` and `ZadoffChuSync`) is what turns "no frame" into
an actual defined outcome instead of noise silently propagating into CFO
correction, header decode, and whatever exception happens to catch it
first — or doesn't:

```python
import numpy as np
from spectracuda.pipeline import Ofdm

ofdm = Ofdm(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    modem="qpsk", fec="conv_v27", crc="crc32",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
)

noise = (np.random.default_rng(1).standard_normal((1, 2000))
         + 1j * np.random.default_rng(2).standard_normal((1, 2000))
        ).astype("complex64") * 0.01

result = ofdm.rx_process(noise)
result["frame_found"]  # False
result["bits"]          # None -- every field but frame_found/start_index/
                         # sync_metric/rssi_db is None on a clean miss
```

A real frame's header round-trips back out as a plain dict — this is
what a receiver actually reads to configure itself, never inspecting the
transmitter's own `Ofdm` instance:

```python
bits = np.random.default_rng(0).integers(0, 2, size=(1, 64)).astype("uint8")
tx_iq = ofdm.generate_frame(bits)
result = ofdm.rx_process(tx_iq)

result["header"]
# {'protocol_version': 1, 'payload_len_bits': 64, 'mod_scheme': 'qpsk',
#  'crc': 'crc32', 'fec0': 'conv_v27', 'fec1': 'none',
#  'user_data': b'\x00\x00\x00\x00\x00\x00\x00\x00'}
```

```{warning}
**One gap left, honestly**: a header that decodes to a *plausible-looking
but wrong* value (garbage that happens to land on a real `mod_scheme`/
`fec0` code, say) still raises `ValueError`/`NotImplementedError` rather
than returning a defined `header_valid=False` result — only the sync
stage has a null outcome worked out so far. See `docs/todo.md` §1.1.
```

## Partial-symbol padding, and why it needed no new wire field

`generate_frame()` doesn't require the CRC+FEC-encoded payload to divide
evenly into whole OFDM symbols — it pads up to the next full symbol with
random (not secret, not constant — same PAPR reasoning as the header's
own scrambling) filler bits, and `rx_process()` strips exactly that much
back off using a value it already had: the header's own
`payload_len_bits` (the *raw*, pre-CRC/pre-FEC count) combined with the
decoded `crc`/`fec0`/`fec1` schemes is enough to recompute the true
encoded length and discard the rest — no separate "how much padding"
field was needed. `MAX_PAYLOAD_SYMBOLS` is still enforced against the
*padded* symbol count, so this can't be used to sneak an over-length
frame past the cap.

`spectracuda/framing/` holds all of this as standalone, `Ofdm`-independent
classes — `HeaderCodec` (the bit-level codec above) and `Packetizer`
(CRC-append-then-FEC-encode on tx, FEC-decode-then-CRC-check on rx,
matching liquid-dsp's own `packetizer_encode`/`packetizer_decode` order)
— reusable and testable with zero OFDM/IQ dependency, the same separation
liquid-dsp itself keeps between `packetizer` and `ofdmflexframegen`/
`ofdmflexframesync`. `Ofdm` builds one throwaway `Packetizer` per
`rx_process()` call from the *decoded* header, never the transmitter's
own configuration — see `spectracuda/framing/header.py` and
`packetizer.py`, and `docs/todo.md` §1.1 for the full extraction history.
