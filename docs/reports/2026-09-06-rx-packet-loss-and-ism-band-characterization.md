# 2026-09-06 session: RX packet-loss root-causing + 2.4/5GHz ISM band characterization

**Goal driving this session:** `abhi/pluto_rx_standalone_v2.py` (RX-only,
threaded reader/decoder) was reporting a persistently high live-RF packet
loss (initially looking like ~30-40%) on a real two-Pi5+Pluto simplex
link, despite the reader thread's own instrumentation showing ~100% ADC
coverage and 0% queue drops. This doc is the record of every bug found
and fixed, every real-RF measurement taken (~30+ separate test runs), and
the open architectural finding that wasn't fixed this session.

Machine/setup: two physically separate Raspberry Pi5 + ADALM-PLUTO rigs,
one-way simplex link (one TX-only, one RX-only), same center
frequency, in a lab room with **at least 20 mesh WiFi devices operating
10-20cm from the hardware**. RX Pluto at `ip:192.168.3.1`. Branch:
`perf/native-fec-and-ofdm-batching`.

Files touched:
- `abhi/pluto_rx_standalone_v2.py` -- the RX test script, iterated on
  throughout (bug fixes + temporary-then-removed diagnostics).
- `abhi/pluto_tx_standalone_test.py` -- a corrected copy of the TX
  script, written here for the user to copy onto the TX-side machine.
- `spectracuda/pipeline/ofdm.py` -- one real fix (none needed beyond
  that); temporary diagnostic prints added and then fully reverted
  (verified clean against `git diff`).

---

## 1. DONE: three real bugs found and fixed

### 1a. TX/RX FEC scheme mismatch
The TX script (`~/Documents/spectracuda/debug/pluto_tx_standalone_test.py`
on the other machine) had `fec="none", fec1="none"`. The RX script has
always run `fec="rs_m8", fec1="conv_v27", strict_fec_check=True` -- which
exists specifically to reject any frame whose header declares a
different fec0/fec1 than configured. A `none`/`none` TX frame is exactly
what that check is designed to reject. Fixed by writing a corrected copy
to `abhi/pluto_tx_standalone_test.py` with matching `fec="rs_m8",
fec1="conv_v27"`, for the user to copy onto the TX machine.

### 1b. `STREAM_CHUNK` set equal to `rx_streaming()`'s SEEKING-state cap
Both `abhi/pluto_rx_standalone_v2.py` and (untouched this session)
`debug/pluto_rx_pingpong_test.py` used `STREAM_CHUNK = 2048`, which
**equals** `Ofdm.STREAM_SEARCH_WINDOW_SYMBOLS(8) * fft_size(256) = 2048`.
When chunk size equals the cap, `rx_streaming()`'s SEEKING-state trim
("keep only the last `cap` samples" after concatenating the new chunk)
discards the *entire* previous chunk on every call, leaving **zero
overlap** between consecutive search windows. Any preamble straddling a
chunk boundary is then structurally unrecoverable.

Confirmed empirically in pure simulation (no real RF at all, a script in
the scratchpad dir, since deleted with the session): scanning frame-start
offsets across one full 2048-sample period,
- `STREAM_CHUNK=2048`: 6/64 offsets missed (9.4%), all landing in the
  last ~192 samples before a chunk boundary.
- `STREAM_CHUNK=1024`: 0/32 missed.

Never caught by the existing pytest suite because no test ever used
chunk_size == cap -- `tests/test_ofdm_streaming*.py` and
`tests/test_mac_rs_viterbi_matrix.py` use chunk sizes 37, 64, 90, 97,
128, 256, 400, 1024, never 2048.

**Fix:** `STREAM_CHUNK` changed to `1024` in
`abhi/pluto_rx_standalone_v2.py`, with a comment explaining why. (Note:
`debug/pluto_rx_pingpong_test.py` has the identical bug and was
identified but **not** fixed this session -- still open if that script
is used again.)

### 1c. libiio shutdown segfault
Every run of `pluto_rx_standalone_v2.py` printed its summary cleanly,
then segfaulted (exit 139) during interpreter teardown. Root cause,
matching the exact pattern already documented in
`debug/pluto_rx_pingpong_test.py`'s own commit message: the reader
thread's closure keeps `rx` (the `adi.Pluto` handle) reachable in a way
plain refcounting doesn't clear, so the interpreter's own cyclic-GC pass
at exit can destroy the libiio `Buffer` object after its parent
`Context` is already gone -- a use-after-free in libiio's C code.

**Fix:** explicit teardown before the summary print --
`del reader_thread; chunk_queue = None; rx = None; gc.collect()` --
same proven pattern already used by the pingpong script. Confirmed fixed
(clean exit code 0 on every run since).

---

## 2. DONE: root-causing the *remaining* loss after both fixes

After 1a-1c, loss dropped but was still substantial (~20-30% at
2.425GHz). Ruled out, with direct measurement, before finding the real
cause:

- **Reader/queue backpressure**: `chunks_dropped` was 0.00% on every
  run except one (915MHz, see §4) -- not the cause.
- **Hardware-level DMA overflow**: SSH'd into the Pluto's own embedded
  Linux (`ssh root@192.168.3.1`, needed a `sudo dhclient`-then-static-IP
  fix on the host's `eth1` first) and checked `dmesg` for
  overflow/underrun/drop messages during a test window -- **clean, no
  hits**. Rules out a silent hardware-level sample discontinuity between
  `rx.rx()` calls.
- **`rx.rx()` call-to-call gap**: added timestamp instrumentation
  (later removed) -- real but tiny, median ~230-280us against a 25,000us
  per-buffer budget at 4Msps. Did not correlate with the loss.

**Actual cause: real RF interference, not code.** Added a temporary
near-miss probe (recompute the SEEKING-state sync metric even when it
doesn't cross threshold) and a temporary header-decode-failure debug
print (both since fully reverted). Findings, at 2.425GHz with 100 real
packets sent:
- **19,531-20,994 false sync triggers per 30s window** (~650-700/sec)
  against ~80-90 real decoded frames in the same window.
- Header-decode failures were all `ValueError: decoded mod_scheme code
  X is not a known scheme`, with X scattered across the full 0-255
  range (241, 250, 14, 68, 9, 122, ...) -- consistent with the header
  bits being noise, not a real header.
- **The metric distributions overlap heavily**: true-positive
  (successful decode) trigger metrics ranged 0.501-0.976 (median
  0.964) across 81 real frames; false near-miss metrics reached as high
  as **0.999**, with 481 exceeding 0.95 and 3,517 exceeding the weakest
  real frame's own metric (0.501). **No single `sync_threshold` value
  cleanly separates real frames from false triggers** in this
  environment.
- Every false trigger occupies the receiver's state machine in
  `WAITING_HEADER` for a window; a real preamble arriving during that
  window is structurally missed (the SEEKING-state search doesn't run
  again until the false lead resolves). With real packets every 100ms
  but false triggers every ~1.5ms, there's ample statistical
  opportunity for collision.
- User confirmed the physical cause: **at least 20 mesh WiFi devices
  operating 10-20cm from the hardware.**
- **Architectural insight (open, not fixed this session):**
  `spectracuda/sync/schmidl_cox.py`'s detector is pure self-correlation
  (`r[n]` vs `r[n+L]`, "do two halves look alike"), with no
  cross-correlation against a sequence specific to this system. 802.11
  WiFi's own preamble (repeated short training symbols) has exactly
  this same repeated-halves structure, so the detector likely cannot
  distinguish this system's own preamble from a nearby WiFi preamble --
  a structural vulnerability to any OFDM-based interferer sharing the
  same repeat-period, not just to generic noise. This is why frequency
  retuning helped (fewer WiFi preambles physically on-air at that
  channel) rather than because the receiver became more selective. See
  §5 for the proposed fix (not yet implemented).

All temporary diagnostics (`OFDM_DEBUG_HEADER_FAIL` env-gated print and
`_stream_trigger_metric` capture in `ofdm.py`; the near-miss probe,
`trigger_metric` field, and near-miss summary line in
`pluto_rx_standalone_v2.py`) were **removed** once root-caused.
`ofdm.py` verified clean against `git diff`. The RX script now just
prints `[decoded] #N crc_valid=... evm=... rssi_db=...` per frame plus
the original summary block.

**Across every test in this entire session, at every frequency and all
three modulations (QPSK/QAM16/QAM64), zero CRC failures were ever
observed on a frame that reached decode.** The FEC/equalizer/demapper
chain has a clean track record throughout -- all measured loss is at
the sync/capture stage, not decode-quality.

---

## 3. DONE: 2.4GHz frequency + TX-power sweep (QPSK, rate=4Msps unless noted)

*Pre-`STREAM_CHUNK` fix (§1b), for reference -- 120s windows, before the
1024 change:*

| Sent | Decoded | CRC-valid |
|---|---|---|
| ~20 | 14 | 14/14 |
| 50 | 33 | 31/33 |
| 50 | 39 | 38/39 |

*Post-`STREAM_CHUNK=1024` fix, all subsequent numbers:*

| Frequency | TX power | Sent | Decoded | CRC-valid | Capture rate |
|---|---|---|---|---|---|
| 2.425 GHz (WiFi center) | -10dBm | 50 | 38 | 38/38 | 76% |
| 2.425 GHz | -10dBm | 100 | 81 | 80/81 | 81% |
| 2.425 GHz | 0dBm | 100 | 82 | -- | 82% |
| 2.425 GHz | 0dBm | 200 | 177 | 175/177 | 88.5% |
| 2.425 GHz | 0dBm | 500 | 409 | 409/409 | 81.8% |
| 2.425 GHz | 0dBm | 500 | 425 | -- | 85% |
| 2.495 GHz (just above ISM top edge) | -10dBm | 500 | 275 | -- | 55% |
| 2.495 GHz | 0dBm | 100 | 97 | -- | 97% |
| 2.402 GHz (ISM bottom edge) | 0dBm | 500 | 457 | -- | 91.4% |
| 2.402 GHz | 0dBm | 500 | 460 | -- | 92% |
| 2.483 GHz (ISM top edge) | -10dBm | 500 | 474 | 474/474 | 94.8% |
| 2.483 GHz | 0dBm | 500 | 485 | -- | 97% |
| 2.525 GHz (just above ISM top edge) | -10dBm | 100 | 93 | 93/93 | 93% |
| 2.525 GHz | -10dBm | 500 | 487 | 487/487 | 97.4% |
| 2.525 GHz | 0dBm | 200 | 198 | 198/198 | 99% |
| 915 MHz | -10dBm | 500 | 421 | 421/421 | 84.2% (**63,293 chunks dropped, 17.94%** -- CPU overwhelmed by 141,985 false triggers/90s, a genuinely different failure mode: real backpressure, only time it's ever occurred) |

**Conclusions:**
- Band **center (2.425/2.495GHz-ish middle) consistently underperforms
  both edges** (2.402/2.483/2.525GHz) -- 20+ point spread, reproducible
  across many runs.
- **TX power matters, but is secondary to frequency**: 0dBm vs -10dBm
  helped hugely at a marginal frequency (2.495GHz: 55%→97%) but
  barely moved 2.425GHz (stayed 82-89% either way).
- **915MHz is not simply "further from WiFi = quieter"** -- it was
  the single worst result of the whole 2.4GHz-adjacent sweep, with a
  qualitatively different failure mode (real CPU/queue backpressure from
  an even higher false-trigger rate than 2.425GHz itself). Likely a
  different interferer (cellular GSM-900/LTE-800/900, or lab equipment)
  dominates that band in this specific room.

## 4. DONE: modulation-order sweep (QPSK / QAM16 / QAM64), 500 pkts each

| Frequency | QPSK | QAM16 | QAM64 |
|---|---|---|---|
| 2.402 GHz | 91.4-92% | 90% (450/450 CRC) | 82.6% (413/413 CRC) |
| 2.483 GHz | 94.8-97% | 89.8% (449/449 CRC) | 66% (330/500, CRC count unconfirmed) |

**Conclusion:** capture rate degrades with modulation order everywhere,
but the *rate* of degradation is frequency-dependent -- 2.402GHz loses
~9 points QPSK→QAM64, 2.483GHz loses ~30 points over the same step.
2.483GHz is the better *QPSK* frequency but the *worse* QAM64 one --
"best frequency" is not a single fixed answer independent of modulation
choice in this environment.

`Ofdm(modem=...)` valid strings, for reference (from
`spectracuda/modem/mapper.py`): `"bpsk"`, `"qpsk"`, `"qam16"`,
`"qam64"`, `"qam256"`.

## 5. DONE: 5GHz ISM band access + characterization

Stock Pluto firmware caps `rx_lo`/`tx_lo` at 3.8GHz (verified directly:
3.80GHz OK, 3.85GHz+ `[Errno 22] Invalid argument`). **Unlocked the full
70MHz-6GHz range** on both Plutos via the standard community trick
(the physical AD9363 die is register-compatible with the wider-range
AD9364; overriding the reported chip identity unlocks the range the
firmware would otherwise restrict):

```bash
ssh root@192.168.3.1   # default password: analog, if unchanged
fw_setenv attr_name compatible
fw_setenv attr_val ad9364   # NOT ad9361 -- Pluto is 1T1R, ad9364 is the
                             # 1T1R wideband part; ad9361 is 2T2R and
                             # would mismatch the physical channel count
reboot
```

`ad9364` (not `ad9361`) confirmed correct by channel-count reasoning
above; verified working after the fact (6.0GHz tunes successfully
post-reboot).

**Side-effect encountered:** after the Pluto's reboot, its USB
RNDIS network interface (`eth1` on the Pi host) re-enumerated at the
USB/kernel level (confirmed via `dmesg`) but came back with **no IP
address** -- ping failed even though `lsusb` showed the device present.
Fixed on the host side (no `sudo` password available in this session,
user ran it):
```bash
sudo ip addr add 192.168.3.10/24 dev eth1
sudo ip link set eth1 up
```
(DHCP via `dhclient` was tried first but abandoned as unnecessary
complexity in favor of just statically assigning the known subnet.)

Post-unlock tuning-range verification (all succeeded): 3.8, 4.0, 4.5,
5.0, 5.725, 5.8, 5.875, 6.0 GHz.

### 5GHz sweep results (QAM64, 500 pkts unless noted, rate=4Msps)

| Frequency | Position | Capture rate |
|---|---|---|
| 5.785 GHz | Center of WiFi channel 157 (busy) | 77.6%, 83.6%, 89.4% (3 runs -- wide spread, consistent with time-varying WiFi load) |
| 5.840 GHz | Edge, just above ch165 | 92% |
| 5.850 GHz | Edge | 95% (500 sent, CRC count not confirmed) |
| **5.860 GHz** | Edge | **95.6%** |
| **5.870 GHz** | Edge | **95.6%** |

Same edge-beats-center pattern as 2.4GHz, now confirmed independently
in a second band. Biggest single jump in this whole session (66%
→ 95.6%, same code, same QAM64 modulation) came purely from frequency
selection, zero pipeline changes -- reinforces §2's conclusion that the
dominant remaining loss source is RF environment, not software.

**Working recommendation for this environment:** 5.860 or 5.870 GHz for
QAM64; plateaued between them (identical 95.6%), so either is fine.
5.860GHz picked as the working choice (more margin below the 5.875GHz
ISM ceiling).

---

## OPEN / not done this session

1. **Sync-detector WiFi vulnerability (§2's architectural finding)**
   not yet fixed. Proposed fix: add a cross-correlation confirmation
   stage against this system's own specific known preamble sequence
   after the coarse Schmidl-Cox self-correlation hit, before committing
   to `WAITING_HEADER` -- would reject WiFi's own (structurally similar
   but sequence-different) preambles while still accepting genuine
   frames, independent of frequency. This is the real fix needed if the
   deployed system must operate *inside* actively-used WiFi channels
   rather than seeking out a quiet edge -- user explicitly flagged this
   requirement ("if we develop this solution it will not work in ISM
   bands esp the WiFi bands"). Not started.
2. `debug/pluto_rx_pingpong_test.py` has the identical `STREAM_CHUNK`
   boundary bug as §1b -- confirmed present, not fixed (out of scope,
   different file than the one actively used this session).
3. Shortening the false-trigger dwell time in `WAITING_HEADER` (an
   alternative/complementary mitigation to §1 above) was discussed but
   not implemented or measured.
4. CRC-valid counts not confirmed for: 2.483GHz QAM64 (330/500),
   5.850GHz QAM64 (475/500) -- both reported as decoded counts only.
5. Full QPSK/QAM16 matrix at 5GHz not run -- only QAM64 was tested in
   the 5GHz band this session.
6. Adaptive-MCS end-to-end run (`examples/drone_tui/`, via
   `Mac.set_tx_scheme()`) was scoped out mid-session in favor of the
   5GHz detour -- not yet attempted with the two standalone Pi5+Pluto
   rigs. Requires switching from the simplex TX-only/RX-only debug
   scripts to the full-duplex `ground_unit.py`/`air_unit.py` pair (each
   needs both TX and RX on its own Pluto, at two different
   frequencies for the two directions) -- command reference:
   ```
   python examples/drone_tui/ground_unit.py --transport pluto --uri ip:192.168.3.1 --tx-freq F1 --rx-freq F2
   python examples/drone_tui/air_unit.py    --transport pluto --uri <air's own Pluto URI> --tx-freq F2 --rx-freq F1
   ```
