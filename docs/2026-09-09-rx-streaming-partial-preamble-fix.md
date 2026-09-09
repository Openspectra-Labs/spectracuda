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

## 3. DONE: fused hard-decision + EVM kernel (the "everything else" bucket's biggest slice)

`benchmark_x86_stages_v3.py`'s unbucketed remainder was the largest RX
line item on x86 (~1.2-1.5 ms/frame, ~40%) and #2 on the Pi-5. A
cProfile map over 40 `receive_iq()` frames (cProfile inflates ~1.3-1.5x,
the ranking is what matters) put the fat at:
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

### What shipped
`spectracuda/modem/_numba_mapper.py`: one fused, `nogil` numba kernel
doing the hard decision (descale -> per-axis level index -> gray ->
bits) AND accumulating the EVM error/reference power sums per row in
the same pass. Wired in as `Modem.demodulate()` (transparent dispatch,
bits only) and a new `Modem.demodulate_stats()` (bits + the two sums);
`_decode_payload_from_header()` now calls the latter and derives EVM
from the sums, dropping the `modulate(demod_bits)` round trip and the
separate `compute_evm` sweep. Same `backend != "cupy"` gating as the
sync/CFO kernels, plus one more: complex64 input only (see below).

**Bit-exactness was the whole design constraint** -- these bits are the
FEC's input. The kernel reproduces numpy's float32 arithmetic step for
step, including a non-obvious one: numpy's `symbols / norm`
(complex64 / python float) is not a division -- numpy's complex-division
loop computes the float32 reciprocal once and multiplies each component
by it. Round-half-to-even via `np.rint`, clip, then int. A complex128
input keeps the numpy path rather than risk a rounding-boundary
mismatch. Only the EVM sums accumulate in float64 (EVM agrees with the
old `compute_evm` to ~1e-6 relative; it is a diagnostic).

### Verification (`tests/test_modem_numba_acceleration.py`, 8 tests)
Bit-for-bit equality with the numpy path for all five schemes at 30 /
10 / 0 / -5 dB SNR (low SNR is what exercises the round/clip
boundaries), and on symbols placed exactly on / one float32 ulp either
side of every per-axis decision threshold (qam16/64/256); EVM sums
match `compute_evm(symbols, modulate(demodulate(symbols)))` to
rtol=1e-5; the dispatch actually reaches the kernel for numpy/complex64
and never for complex128 or `backend="cupy"`; and end-to-end
`rx_process()` on a real noisy bound QAM16 frame returns identical
payload bits and equal EVM with the kernel on vs forced off. Full
suite: 945 passed, 34 skipped, the same 4 pre-existing CPE-commit
failures, nothing new.

### Measured (x86, pinned core, one frame's 128x216 payload symbols, median of 200)
| scheme | old demodulate+modulate+compute_evm | fused `demodulate_stats` | | `demodulate` alone, old -> new |
|---|---|---|---|---|
| qpsk | 0.987 ms | 0.166 ms | **5.9x** | 0.426 -> 0.170 ms (2.5x) |
| qam16 | 2.634 ms | 0.178 ms | **14.8x** | 0.784 -> 0.174 ms (4.5x) |
| qam64 | 2.344 ms | 0.191 ms | **12.3x** | 0.862 -> 0.191 ms (4.5x) |

End-to-end (`benchmark_x86_stages_v3.py 32000`, same machine, same
session): RX **~3.2-3.3 -> ~2.71 ms/frame** at QPSK, the "everything
else" bucket ~1.2-1.5 -> ~0.8 ms; QAM64 RX 3.56 -> 3.24 ms/frame. (WSL2
absolute numbers drift run to run; the isolated A/B above is the
trustworthy figure, the end-to-end one is corroboration.)

### If resuming here
- The kernel is ~6 ns/symbol -- not tapped out (the per-bit inner loop
  and `int(np.rint())` are the obvious places), but it is no longer the
  bottleneck: what's left in "everything else" is the slot gather + CPE
  math in `_decode_payload_from_header` (~0.3 ms) and the header path.
- Next largest RX items on x86 are now Viterbi (~0.78 ms) and OFDM
  decode (~0.45 ms -- 128 x 256-pt FFTs should be ~50 us in pocketfft,
  so the CP-strip gather/reshape copies are the suspect there).
