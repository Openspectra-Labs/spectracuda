# Periodic DMRS — refreshing H[k] inside long frames

**Status: planned 2026-09-20, not started.**

Agreed after reading the FlexLink PHY spec (`docs/flexlink-spec-review.md`
covers that document's own defects separately). FlexLink is an unvalidated
paper spec with no published measurements, so it is treated here as one
data point on how others structure the problem, **not** as an authority.
The sizing rationale below comes from LTE, which is deployed.

## Goal

One sentence: **when a frame is long enough that the channel estimate taken
at the start of it goes stale, insert a known reference symbol roughly every
1 ms and re-estimate H[k] from it.**

Nothing else about the PHY changes. The existing architecture is sound and
is not being redesigned around FlexLink, LTE or Wi-Fi.

## What is already in place (and stays)

| stage | mechanism | status |
|---|---|---|
| detection / timing / CFO | preamble (Schmidl-Cox, no CP) | unchanged |
| initial H[k] | training symbol, `Ofdm._train_grid_freq` | unchanged |
| continuous phase / CPE tracking | pilots in every payload symbol | unchanged |
| **periodic H[k] refresh** | **DMRS** | **this plan** |
| burst-error repair | interleaver + Viterbi + RS + CRC | unchanged |

Explicitly NOT in this plan: AGC burst, dual payload A/B, SFBC / 2T2R,
LDPC migration, rate matching, MCS table pruning. Those were discussed and
either deferred or rejected.

## The key insight that makes this cheap

A DMRS is not a new signal. `Ofdm.__init__` already builds
`self._train_grid_freq` (`pipeline/ofdm.py:520`) — a full known QPSK symbol
on every data subcarrier plus the known pilots, seeded from
`training_seed=999` — and already constructs `self.channel_estimator`
against `self._train_known_indices` / `train_known_values`
(`pipeline/ofdm.py:529`).

So a DMRS is literally **"re-transmit the training symbol at payload slot N
and re-run the existing estimator."** No new reference sequence, no new
estimator, no new tables, and no 2-D resource-grid refactor. The grid stays
1-D; the payload mapper just skips DMRS slots.

(A 2-D grid would only be needed for FlexLink-style *partial* DMRS — pilots
on every 3rd subcarrier with data in between. Full-symbol DMRS does not
need it, and we are not doing partial DMRS.)

## Decided parameters

**Periodicity is counted in data symbols**, not in total slots and not in
samples. This is the only counting rule that makes the refresh interval
*exactly* constant across sample rates:

| interval `I` | 5 MSps | 10 MSps | 20 MSps |
|---|---|---|---|
| 16 | **921.6 µs** | 460.8 | 230.4 |
| 32 | 1843.2 | **921.6** | 460.8 |
| 64 | 3686.4 | 1843.2 | **921.6** |

With `fft_size=256, cp_len=32` the slot is 288 samples, so `I` data symbols
span `I × 288 / Fs` seconds. Doubling the sample rate halves the symbol and
doubling `I` cancels it exactly — the bold diagonal is 921.6 µs at all three
rates, with zero error. The off-diagonal entries are useful as a plain
density control at a fixed rate.

**Why ~1 ms:** LTE's DM-RS repeats per 1 ms subframe. An LTE symbol (15 kHz
SCS, normal CP) is 71.4 µs, so that 1 ms is 14 LTE symbols. Our symbols at
10 MSps are 28.8 µs, so the same wall-clock interval is ~35 of ours — which
is why `I=32` (921.6 µs) is the right diagonal entry and lands within 8% of
LTE's subframe.

**Ladder: `OFF / 16 / 32 / 64`.** No `8`. The case for `8` came from
FlexLink's `0.25/f_d` rule of thumb, which is unvalidated and, for our
architecture, pessimistic — we already correct CPE from per-symbol pilots,
so the DMRS only has to catch residual *frequency-selective* drift, not bulk
phase rotation. If measurement later shows a denser setting is needed, bit 7
of the same header byte is still free.

`OFF` earns its code: at 10 MSps a ~1 ms TXOP is 31 payload symbols, so
`I=32` inserts zero DMRS anyway. Short frames never pay for this.

### Insertion rule

```
n_dmrs = 0                      if I == 0 (OFF)
n_dmrs = (n_data_symbols - 1) // I    otherwise
```

A DMRS is emitted after each complete run of `I` data symbols, and the
`-1` suppresses a trailing DMRS with no data after it (which would carry a
channel estimate nothing uses). Resulting cost:

Every row below respects `n_data + n_dmrs <= 128` (see the next section):

| `n_data` | `I` | DMRS | total slots | airtime @10 MSps | DMRS overhead |
|---|---|---|---|---|---|
| 31 (~1 ms TXOP) | 32 | 0 | 31 | 976.0 µs | 0.0% |
| 31 (~1 ms TXOP) | 16 | 1 | 32 | 1004.8 µs | 3.1% |
| 127 (max, I=64) | 64 | 1 | 128 | 3769.6 µs | 0.8% |
| 125 (max, I=32) | 32 | 3 | 128 | 3769.6 µs | 2.3% |
| 121 (max, I=16) | 16 | 7 | 128 | 3769.6 µs | 5.5% |

At the diagonal setting (`I=32` @ 10 MSps) the cost is ~2% of a
maximum-length frame and zero on a normal one. Note the three max rows all
have the same 3769.6 µs airtime — that is the point: DMRS trades payload
capacity, never airtime.

### MAX_PAYLOAD_SYMBOLS — 128 is a TOTAL, DMRS included

**The rule, stated once:**

```
n_data_symbols + n_dmrs_symbols  <=  MAX_PAYLOAD_SYMBOLS (128)
```

DMRS comes **out of** the 128, never on top of it. A frame never occupies
more airtime with DMRS enabled than the same limit allowed without it. The
limit exists to cap airtime (it is documented at `pipeline/ofdm.py:260` as a
coherence-time bound), so letting DMRS push past it would defeat the reason
it is there.

Concretely, the data-symbol ceiling drops as DMRS density rises:

```
I=OFF: 128 data +  0 DMRS = 128 slots
I=64:  127 data +  1 DMRS = 128 slots
I=32:  125 data +  3 DMRS = 128 slots
I=16:  121 data +  7 DMRS = 128 slots
```

So enabling DMRS costs payload capacity, as it should — it does not buy
extra airtime.

**Implementation note — do not use a closed form.** The obvious
`floor((128·I + 1) / (I + 1))` is wrong: for `I=32` it gives 124, but 125 is
correct (`125 + (125-1)//32 = 125 + 3 = 128`). Derive `max_data_symbols` by
counting down from `MAX_PAYLOAD_SYMBOLS` until
`n + n_dmrs_symbols(n, I) <= MAX_PAYLOAD_SYMBOLS` holds. It runs once at
construction (and on `reconfigure_tx_scheme`), at most 128 iterations, and
is obviously correct by inspection — which a subtly-off closed form is not.

**Both guard sites check the total, not the data count:**

- `generate_frame()` (`ofdm.py:775`) — compute `n_data_symbols`, then
  `n_dmrs`, then raise if `n_data + n_dmrs > MAX_PAYLOAD_SYMBOLS`.
- `_decode_header_from_sync()` (`ofdm.py:1056`, and the explicit-override
  branch at `:1069`) — same, from the decoded `payload_len_bits` plus the
  decoded interval.

Both error messages should name the data count, the DMRS count and the
total, so a frame rejected at 125+3 is not mistaken for one rejected at 128
data symbols.

### How it is configured

`Ofdm.__init__` gains `dmrs_interval: int = 0`, and
`reconfigure_tx_scheme()` gains a matching `dmrs_interval=None` argument —
the same path adaptive MCS already uses. **Not** a `generate_frame()`
per-call argument, and **not** "rebuild the Ofdm object".

Three reasons:

1. **The insertion rule is self-gating, so it is set once.** Because
   `n_dmrs = (n_data - 1) // I`, a permanent `dmrs_interval=32` already
   yields 0 DMRS on a ~1 ms / 31-symbol TXOP and 3 on a maximum-length
   frame. Per-frame control buys nothing for the normal case; it would only
   matter for varying *density* (16 vs 32) from measured Doppler, which is
   an adaptive-controller concern and already belongs on
   `reconfigure_tx_scheme()`.
2. **No RX-side counterpart is needed.** The receiver resolves the interval
   from the decoded header, exactly as it resolves `mod_scheme`. This puts
   it in the easy category described in `reconfigure_tx_scheme()`'s own
   docstring — unlike `fec`/`fec1`, where a `strict_fec_check=True` receiver
   must be reconfigured in step with the sender. Calling it on one end alone
   is safe.
3. **Rebuilding the object is wrong for the reason already documented
   there:** the same `Ofdm` owns the RX streaming state
   (`_stream_buffer`/`_stream_header`) for the *other* direction of a
   full-duplex `Mac`, and discarding it would lose a partial frame purely
   because this side's outgoing setting changed.

`dmrs_interval` is scheme-derived only — it touches no grid, codec, trellis
or GF table — so unlike a `fec` change it costs nothing to apply.

### MAC capacity — do not miss this

`mac/capacity.py:59` computes

```python
limit = ofdm.MAX_PAYLOAD_SYMBOLS * ofdm.bits_per_ofdm_symbol
```

i.e. it assumes all 128 symbols are available for data. With DMRS on, only
121 / 125 / 127 data symbols fit inside the 128-slot airtime cap, so the MAC
would segment to 128 data symbols, `generate_frame()` would add DMRS on top,
and the total-slot check would raise.

`bits_per_ofdm_symbol` does not change here — the *symbol budget* does — so
`reconfigure_tx_scheme()`'s existing return value does not cover it. Add:

```python
@property
def max_data_symbols(self) -> int:   # MAX_PAYLOAD_SYMBOLS minus this interval's DMRS
```

and have `compute_max_segment_bits()` use it in place of the raw constant.

### Wire format

Two bits in the header. `HeaderCodec` byte 5 is `fec1 & 0x1F`
(`framing/header.py:144`), so **bits [7:5] of byte 5 are unused**, and the
decoder already masks them off (`header_bytes[5] & 0x1F`,
`framing/header.py:163`). Take **bits [6:5]** for `dmrs_period`, leaving bit
7 free:

```
byte 5:  [7] reserved  [6:5] dmrs_period  [4:0] fec1
         dmrs_period: 0 = OFF, 1 = every 16, 2 = every 32, 3 = every 64
```

This is backward-compatible on decode (old decoders already ignore those
bits) and costs nothing — no header length change, and `user_data`'s 8 bytes
stay untouched.

---

## The one hard problem, isolated

Everything in this plan is bookkeeping **except step 3**.

`_decode_payload_from_header` currently does:

```python
h_hat_combined = xp.repeat(h_hat_data, n_payload_symbols, axis=0)   # ofdm.py:1142
```

— one channel estimate, broadcast across every payload symbol. That single
line is the assumption DMRS breaks. With DMRS, payload symbols belong to
*segments*, and each segment uses the estimate from the DMRS that precedes
it (segment 0 uses the training symbol's estimate, as today).

The same applies to `h_hat_pilots` (`ofdm.py:1177`).

The fix is structurally small — `xp.repeat` with a per-segment repeat count
instead of a uniform one — but it is where the batching, the segment
boundaries and the estimator all have to agree, and it is the step that can
quietly produce *plausible but wrong* equalization rather than an exception.
It gets built and verified on its own, against a forced DMRS layout, before
any header plumbing exists.

---

## Steps

### 1. Slot map helper — pure function, no I/O

`spectracuda/framing/` gains:

```python
def dmrs_slot_map(n_data_symbols: int, interval: int) -> np.ndarray
    # -> uint8 array over total slots; 0 = data, 1 = DMRS
```

Plus `n_dmrs_symbols(n_data_symbols, interval)`. Fully testable with no OFDM
machinery. Table-driven tests against the rule above, including the
`OFF`/`n_data <= I`/trailing-DMRS edge cases.

### 2. TX insertion, forced

`Ofdm.__init__` gains `dmrs_interval: int = 0`. `generate_frame` builds the
slot map, tiles `train_time_one` into the DMRS slots and the payload symbols
into the rest, replacing the flat reshape at `ofdm.py:811-818`.

Verify: TX-only. Frame length in samples matches `OVH + total_slots × 288`,
and the samples at each DMRS slot are bit-identical to the training symbol.

### 3. RX per-segment H[k] — **the hard step**

In `_decode_header_from_sync` / `_decode_payload_from_header`:

1. derive the slot map from `n_payload_symbols` + interval,
2. extract the DMRS slots and run `self.channel_estimator` on each,
3. build `h_hat_combined` by repeating each segment's estimate over that
   segment's data symbols, and the same for `h_hat_pilots`,
4. keep the existing bounds check (`ofdm.py:1115-1134`) correct against
   total slots rather than payload symbols.

Verify, in this order:
- **`dmrs_interval=0` must be bit-exact against today's output.** Non-negotiable
  regression gate — run the existing test suite unchanged.
- With DMRS on and a *static* channel, decoded bits must be identical to
  DMRS off (the refreshed estimate should agree with the original).
- With DMRS on and a *time-varying* channel (`sim/channel.py`), EVM at the
  end of a long frame must improve measurably vs DMRS off. **This is the
  step's actual success criterion** — if it does not improve, the feature
  does not work, however clean the plumbing looks.

### 4. Header field

Replace the step-2 constructor kwarg as the source of truth with the 2-bit
field in `HeaderCodec` byte 5 (`dmrs_period` encode/decode + the
code/name tables). The constructor kwarg stays as the TX-side setting;
the RX reads the header.

Round-trip tests for all four codes, plus a test that byte 5's bit 7 stays
zero and that `fec1` still decodes correctly alongside a non-zero
`dmrs_period`.

### 5. MAC capacity

Add `Ofdm.max_data_symbols` and switch `compute_max_segment_bits()`
(`mac/capacity.py:59`) onto it. Also add `dmrs_interval` to
`reconfigure_tx_scheme()`.

Verify: with `dmrs_interval=0` the segment size is unchanged from today
(regression gate), and for each interval a maximum-size MAC segment
produces a frame of **exactly 128 total slots** (data + DMRS) rather than
raising — and never 129+. Assert the totals directly: 128/0, 127/1, 125/3,
121/7 for OFF/64/32/16.

### 6. `rx_streaming`

`frame_end = pos_scalar + h["n_payload_symbols"] * self.slot_len`
(`ofdm.py:1474`) must use **total slots**, not payload symbols, or streaming
truncates every DMRS-bearing frame. One line, but it is the site most likely
to be missed — it is the only frame-length computation outside
`_decode_payload_from_header`.

Verify with the existing streaming tests plus one long-frame DMRS case.

### 7. Measure

On the Pi-5 / two-Pluto rig, not on WSL2 (see `wsl2-benchmark-timing-noise`
and `wsl2-usb-tunneling-blocks-native-throughput` — throughput cannot be
measured here):

- PER and EVM vs frame length, DMRS off / 16 / 32 / 64.
- Whether the improvement shows up at all at realistic UAV speeds, or only
  under deliberate motion. If it only helps under motion, that is a fine
  result — it tells the MAC when to turn it on.
- Per-frame cost of the extra channel estimates (step 3 runs the estimator
  up to 7 extra times per frame).

## Open questions, to be answered by measurement not by design

- Does 921.6 µs actually help on our link, and at what frame length does the
  static estimate start to cost us? The whole feature is justified by LTE's
  precedent rather than by anything we have measured on this radio.
- Should the MAC pick the interval adaptively (e.g. from measured EVM slope
  across the frame) rather than being configured? Out of scope here; the
  wire format supports it either way.
