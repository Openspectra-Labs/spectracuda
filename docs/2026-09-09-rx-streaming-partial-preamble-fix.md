# 2026-09-09 session: rx_streaming() alignment-dependent frame loss (fixed) + streaming overhead re-measured

Started as a follow-up measurement to `docs/2026-09-09-numba-sync-kernel.md`
("re-measure `rx_streaming()`'s per-call overhead now that sync is 5x
faster") and turned up a real correctness bug in the live receive path
along the way. Both are recorded here. Machine: the x86 WSL2 dev box,
pinned to one core; every number below is measured, not estimated.

## 1. FIXED: two alignment-dependent frame losses in SEEKING

Found by feeding a real, clean (zero-noise), bound QPSK frame through
`Ofdm.rx_streaming()` with a 4096-sample zero lead-in in 64-sample
chunks: `payload_fail=1`, no frame delivered. Same frame with no lead-in
decoded fine. Instrumenting every SEEKING `sync.process()` call showed
the trigger firing with only **192 of the preamble's 256 samples in the
buffer**, metric **0.435** (threshold 0.3), `start_index` **64 samples
early** -- past the 32-sample CP, so ISI from the preceding samples and
a failed decode.

Two distinct mechanisms, both pure functions of where the preamble
start lands relative to chunk boundaries (nothing to do with channel
quality):

1. **Premature trigger on a partially-arrived preamble.** With k of the
   preamble's fft_size samples present (k > L = fft_size/2), the
   Schmidl-Cox metric at the very last candidate offset
   (`buffer_len - fft_size`) is already `(2(k-L)/k)^2`: 0.44 at k=192,
   0.73 at k=224 -- over the 0.3 default threshold -- with a start index
   `(fft_size - k)` samples early. Verified exactly: predicted 0.444 /
   -64 samples at k=192, measured 0.435 / -64.
2. **The SEEKING cap chopping preamble heads.** The old
   `buffer = buffer[-cap:]` (cap = 8 symbols = 2048 samples) applied
   after the concat discarded the ENTIRE previous buffer whenever one
   chunk was as long as the cap -- and 2048 is exactly what
   `examples/pluto_air_unit.py` / `pluto_ground_unit.py` feed. A
   preamble that started in the previous chunk's tail with too few
   samples to trigger (k < ~177) had its head evicted and was never
   found.

Clean-channel loss, swept over preamble alignment within one chunk
(`tests/test_ofdm_streaming_alignment.py` reproduces the failing bands):

| chunk | lost before fix | after |
|---|---|---|
| 2048 (the Pluto scripts' size) | **9.4%** of alignments (both mechanisms) | 0 |
| 1024 | 3.1% (coarse sweep; ~2% analytically) | 0 |
| 256 | 6.2% | 0 |
| 64, steady-state (long lead-in) | ~30% analytically (failing band = k_trig in 177..~196) | 0/64 |

Why nobody saw it before: `mac_streaming_demo.py` and friends feed each
frame starting at buffer offset 0, where the premature candidate index
is negative -- the one alignment that can't fail. Real hardware, with
continuous noise between frames, hits arbitrary alignments; this is
plausibly a real slice of the ~25% first-shot loss seen in the
20-WiFi-network interference test.

### The fix (`spectracuda/pipeline/ofdm.py`, `rx_streaming()` SEEKING branch)
- Bound the SEEKING buffer to `history + len(chunk)` (history =
  STREAM_SEARCH_WINDOW_SYMBOLS * fft_size), not a fixed total, so the
  previous chunk's tail always survives one more call.
- Trailing-edge guard: accept a detection only if
  `start_index + fft_size + L <= buffer_len`, i.e. L samples past the
  detected preamble's end are already in the buffer. A partial preamble
  can never out-score the full one, so by then the true start is the
  argmax. L is the exact bound independent of `sync_threshold`: with
  <= L preamble samples present the two halves don't overlap and the
  metric is ~0. Costs at most L samples of detection latency, which is
  irrelevant (WAITING_HEADER needs the whole header anyway).

### Verification
- `tests/test_ofdm_streaming_alignment.py` (new, 6 tests): every
  previously-failing alignment band at chunk 64/256/1024/2048 now
  decodes; the partial-preamble mechanism is checked directly (192
  samples in -> must NOT trigger; full preamble + L -> triggers at the
  exact true start); the cap-chop mechanism is checked in isolation.
- Post-fix sweep: 0 losses at every chunk size, 0/64 in the chunk=64
  steady-state case, 0/40 at random alignments with ~20 dB SNR.
- Existing streaming suites (`test_ofdm_streaming.py`,
  `test_ofdm_streaming_combination_matrix.py`,
  `test_mac_bidirectional_am.py`, `test_mac_rs_viterbi_matrix.py`): all
  pass; `test_search_buffer_stays_bounded_on_a_long_noise_only_stream`
  updated to assert the new `history + chunk` bound (still bounded, just
  no longer a fixed total).

## 2. MEASURED: rx_streaming() per-call overhead is no longer the problem

The "~40-150us near-fixed per-call overhead" note in
`examples/pluto_air_unit.py` predates the numba sync kernel. Re-measured
(x86, pinned core, real 37696-sample QPSK frame = 3.77 ms airtime at
10 Msps):

| chunk | idle SEEKING us/call (numba on / off) | real frame: non-decode us/call | decode call | wall / airtime |
|---|---|---|---|---|
| 64 | 11.9 / 33.3 | 5.8 | 3.9 ms | 2.04 |
| 256 | 12.0 / 31.8 | 7.6 | 4.1 ms | 1.47 |
| 1024 | 12.6 / 33.2 | 15.2 | 4.0 ms | 1.18 |
| 2048 | 13.0 / 33.6 | 19.4 | 4.1 ms | 1.18 |

The per-call floor is ~12 us (sync kernel on), flat across chunk sizes
-- 12% of a 1024-sample chunk's airtime. At >= 1024-sample chunks the
streaming machinery is negligible; the remaining wall is the ~4 ms
decode of the frame itself vs 3.77 ms of airtime (1.18x -- this x86 box
does not quite keep up single-core at 10 Msps, and that is the decode,
not streaming). 64-sample chunks are still 2x airtime: don't.

## 3. NOT DONE: where the decode's "everything else" bucket goes

`benchmark_x86_stages_v3.py`'s unbucketed remainder is now the largest
RX line item on x86 (~1.2-1.5 ms/frame, ~40%) and #2 on the Pi-5. A
cProfile map over 40 `receive_iq()` frames (cProfile inflates ~1.3-1.5x,
the ranking is what matters) puts the fat at:
- `Modem.demodulate` ~0.68 ms/frame (the generic M-QAM path:
  descale, round/clip per axis, gray, bit unpack, concat -- ~20 array
  passes and 22 `astype` copies per frame for what is a sign test at
  QPSK).
- **EVM computed on every frame via `payload_modem.modulate(demod_bits)`
  + `compute_evm` ~0.5 ms/frame** -- pure diagnostics; consumed by
  `Mac._apply_rx_result` for link-quality bookkeeping, so it can't just
  be dropped, but the `modulate()` round-trip (bits -> gray -> int ->
  PAM level) is unnecessary: the nearest constellation point is already
  known inside `demodulate()`.
- `_decode_payload_from_header` self time ~0.3 ms (slot gather + CPE
  math), `compute_rssi_db` ~0.1 ms.

Proposed next step: one fused, `nogil` numba hard-decision kernel
returning bits AND the nearest-point symbols (and the EVM power sums) in
a single pass, replacing demodulate + modulate + compute_evm on the
payload path. Same two-gate discipline as every kernel before it:
bit-exact against the existing `Modem` on all five schemes first
(`tests/test_modem.py`), benchmark second.
