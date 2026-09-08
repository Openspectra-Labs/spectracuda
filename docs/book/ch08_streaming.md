# Chapter 08 — Streaming Receive

Every chapter so far has called `rx_process(iq)` on one already-bounded
frame's worth of IQ — the batch-shape contract that makes GPU throughput
worth anything on a Jetson-class part (`docs/architecture.md`'s "Batch-
shape contract"), and it stays exactly that: unchanged, batch-only, by
deliberate design. A real receive chain doesn't get samples that way —
it gets arbitrary, unaligned chunks off a continuous ADC stream, with no
relationship to frame or symbol boundaries at all. `rx_streaming()` is
the additive answer to that case, not a replacement for `rx_process()`.

## One call, one chunk, at most one frame back

```python
import numpy as np
from spectracuda.pipeline import Ofdm

ofdm = Ofdm(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    modem="qpsk", fec="conv_v27", crc="crc32",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
)

bits = np.random.default_rng(0).integers(0, 2, size=(1, 64)).astype("uint8")
tx_iq = ofdm.generate_frame(bits)[0]                    # 1D IQ stream
stream = np.concatenate([np.zeros(50, dtype="complex64"),
                          tx_iq,
                          np.zeros(50, dtype="complex64")])

CHUNK = 97                                                # deliberately non-aligned
result = None
for i in range(0, len(stream), CHUNK):
    result = ofdm.rx_streaming(stream[i : i + CHUNK])
    if result is not None:
        break

result["bits"]   # matches `bits`, bit-identical to a batch rx_process() call
```

`rx_streaming(chunk)` returns `None` while still accumulating or
searching, and a `rx_process()`-shaped result dict the instant a complete
frame finishes decoding — feed it any chunk size, any alignment, in a
loop, and it behaves like a real streaming receiver rather than a batch
call in a trenchcoat.

## Same state machine liquid-dsp already proved out

This isn't invented from scratch — it's checked directly against
liquid-dsp's own `ofdmframesync_execute()` state machine (`SEEKPLCP` →
`PLCPSHORT0/1` → `PLCPLONG` → `RXSYMBOLS`) before being written, reusing
the same key insight: once sync is found, every remaining length is
already known from the object's own config, so the receiver never has to
guess how many more samples it needs. Where this implementation
deliberately diverges from a literal port: rather than a per-sample
NCO/timer, it re-runs this project's existing *batch-vectorized*
sync/CFO/channel-estimator/equalizer/demod primitives against
accumulated slices once enough samples exist for the current stage — a
stated simplification, not an oversight, and single-stream (`n_batch=1`)
only for now; multi-stream batched streaming is a separate, unattempted
extension.

```{note}
**A real correctness bug, caught by direct comparison, not assumed
fixed.** The CFO-corrected buffer computed at the header stage goes
stale by the time payload samples arrive — it was corrected over a
buffer that didn't yet include them. Fixed by re-applying `cfo.correct()`
to the full, now-complete buffer with the *same* already-estimated
`cfo_estimate`, rather than reusing the stale array. Caught by comparing
streaming output against `rx_process()`'s own EVM directly, matched to
`1e-6` — not "close enough," bit-identical.
```

## Bounded search, and failures that don't stop the receiver

`Ofdm.STREAM_SEARCH_WINDOW_SYMBOLS` (default 8× `fft_size`) caps how much
gets accumulated while still *searching* — a long silent or noisy stream
with no frame in it doesn't grow the buffer, or the matched-filter
correlation cost, without bound. That cap is never applied once a frame
is actually being decoded.

Failure handling deliberately diverges from `rx_process()`'s own "fail
loud" convention (Chapter 06): a corrupted header, a false-positive sync
trigger, or uncorrectable FEC does **not** raise here — it's silently
discarded (`None` returned), and the state machine resumes searching. A
one-shot `rx_process()` call has a caller waiting on that exact result;
a streaming receiver has to survive one bad frame and keep running
indefinitely, so the two contracts are different on purpose, not an
inconsistency.

```{warning}
**`STREAM_CHUNK` equal to the search-window cap is a real, documented
footgun** — not a hypothetical one. If a caller's own chunk size exactly
equals `STREAM_SEARCH_WINDOW_SYMBOLS * fft_size`, the SEEKING-state trim
("keep only the last `cap` samples") discards the *entire* previous
chunk on every call, leaving zero overlap between consecutive search
windows — any preamble straddling a chunk boundary becomes structurally
unrecoverable. Found on real two-Pluto hardware, confirmed in pure
simulation (6/64 offsets missed at `chunk == cap`, 0/32 missed once
`chunk` was set to half that) — see
{doc}`../2026-09-06-rx-packet-loss-and-ism-band-characterization` §1b.
Keep your chunk size comfortably smaller than the search-window cap.
```

Verification: `tests/test_ofdm_streaming.py` (bit-identical match against
`rx_process()` across 5 chunk sizes including a non-aligned one, multiple
frames back-to-back, pure-noise streams that never falsely complete, and
a corrupted-frame-then-recovers case) plus
`tests/test_ofdm_streaming_combination_matrix.py` (49 tests — the full
`fft_size`×modem×FEC grid, every valid sync/CFO pairing, and the real
multipath/AWGN/CFO subset, all fed through `rx_streaming()` with a
deliberately non-aligned chunk size). See `docs/todo.md` §2.5 for the
full design history, including the header/payload-decode refactor
`rx_streaming()` and `rx_process()` now both call through rather than
risking two copies of the same delicate math drifting apart.
