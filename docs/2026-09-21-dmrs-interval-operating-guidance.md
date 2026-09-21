# DMRS interval: operating guidance per sample rate

**Decision recorded 2026-09-21. Simulation-backed, hardware-unvalidated.**
Supersedes nothing; it is the operating-point choice that follows from
`docs/2026-09-21-dmrs-differential-doppler-characterization.md` and the
runs under `debug/dmrs_doppler_*/`.

All figures assume the 320-sample symbol (`fft_size=256`, `cp_len=64`),
16QAM, `rs_m8` + block interleaver + `conv_v27`, 8 pilots, 2 training
symbols, `timing_advance=2`, 5575 B payload.

## The setting

| rate | mode | `dmrs_interval` | nominal | **true dT** | overhead |
|---|---|---|---:|---:|---:|
| 10 MSps | default | **16** | 512 us | 544 us | 5.6% |
| 10 MSps | high mobility / bad channel | **8** | 256 us | 288 us | 10.5% |
| 20 MSps | default | **32** | 512 us | 528 us | 2.5% |
| 20 MSps | high mobility / bad channel | **16** | 256 us | 272 us | 5.6% |

The interval counts DATA symbols, not time. A 320-sample symbol is
32.0 us at 10 MSps and 16.0 us at 20 MSps, so the same `iv` is a
different refresh period at each rate -- which is why the table is
per-rate and why 512 us is `iv=16` at one rate and `iv=32` at the other.
True dT is `(iv+1) x slot`, because a DMRS occupies a slot of its own;
overheads are measured from real 5575 B frames, not `1/(iv+1)`.

## Why these four

Differential Doppler `delta_f` tolerated, 16QAM, echo a=0.2:

| rate / iv | 15 dB | 25 dB |
|---|---|---|
| 10 MSps iv=16 | 500/500 to 300 Hz; 487/500 at 400; 61/500 at 500 | 200/200 to 500 Hz |
| 10 MSps iv=8 | 500/500 to 500 Hz (end of sweep) | 100/100 to 1000 Hz |
| 20 MSps iv=32 | 300/300 to 300 Hz; 295/300 at 400; 165/300 at 500 | not swept |
| 20 MSps iv=16 | 300/300 to **600 Hz** (end of sweep) | not swept |

The defaults carry the 300 Hz engineering test point with room to spare
and reach 400 Hz at ~98%. **400 Hz is the limit of the default, not its
operating point** -- at 400 Hz the RS load is already 2.5-4.0 bytes per
codeword against a 16-byte budget with a fraction of codewords over, and
one sweep point further the frames are gone. The high-mobility settings
show no failure anywhere in the swept range at either rate.

Two results underpin the table and are worth not re-deriving:

- **Aging follows `delta_f x dT` in TIME**, not the sample rate and not
  `iv`. Matched-time pairs agree to within 0.006 mean EVM in every cell.
  That is what lets 20 MSps buy the same tracking for less overhead.
- **Physical delay spread does not enter.** A 100 ns echo (2 samples at
  20 MSps) behaves like a 50 ns one (1 sample) to within 0.003 EVM,
  because the stale-H error `2a|sin(pi*delta_f*dT)|` carries no delay
  term -- delay rotates the error across subcarriers without changing
  its size.

## Blocker: `iv=8` is not on the wire

`framing/dmrs.py` fixes the 2-bit `dmrs_period` field at

```python
DMRS_PERIOD_CODES = {0: 0, 1: 16, 2: 32, 3: 64}
```

so of the four settings above:

| setting | encodable today? |
|---|---|
| 10 MSps default, `iv=16` | yes (code 1) |
| 10 MSps high mobility, `iv=8` | **NO** |
| 20 MSps default, `iv=32` | yes (code 2) |
| 20 MSps high mobility, `iv=16` | yes (code 1) |

**The 20 MSps pair works with the shipped wire format unchanged. The
10 MSps pair does not** -- its high-mobility setting has no code.

The cheap fix, if 10 MSps needs the high-mobility mode, is to re-point
code 3 from `64` to `8`: `iv=64` is dead weight by measurement (1024 us
at 10 MSps already fails at 300 Hz, and 2048 us is worse), and the field
keeps its width. That breaks interoperability with any already-deployed
build, so it is a decision rather than a patch, and it has not been made.

This is a further argument for 20 MSps, alongside the two already on
record: 20 MSps with `fft=256`/`cp=64` is exactly the 802.11ax numerology
(78.125 kHz subcarrier spacing, 12.8 us symbol, 3.2 us CP = 11ax's long
guard interval), and it halves frame airtime.

## Soft-decision Viterbi changes part of this picture

Soft-decision Viterbi has been experimentally demonstrated **in
simulation** to substantially improve resistance to frequency-selective
multipath. In characterized 20 MSps / 16QAM / 15 dB cases, several
channels that produced complete packet loss with hard decisions recovered
completely or nearly completely with soft decisions. Extremely broad or
deep fades remain problematic. The current software implementation is
unoptimized and incurs substantial CPU cost, so soft decoding remains
optional pending optimization and a quantized-LLR / FPGA cost evaluation.

`Ofdm(soft_decision=True)`, **off by default** — which is the correct
engineering choice until that optimization and quantization work is
finished. See
`docs/2026-09-21-multipath-severity-characterization.md` §7.

One consequence worth noting for the table above: the a=0.4 / 500 ns /
300 Hz case previously needed `iv=16` to survive and is recovered by soft
decision at `iv=32`. If soft decision is eventually enabled, some of the
high-mobility interval choices may be revisitable at lower DMRS overhead
— but that has not been characterized across the interval matrix and the
table above stands unchanged for hard-decision operation.

## What this guidance does not establish

- **No hardware validation.** Every number is simulation against a single
  deterministic rotating tap -- no fading statistics, no delay profile,
  no angle-of-arrival geometry. The Pi/Pluto mobility run named in
  `docs/spectracuda-phy-specification.md` remains the completion gate.
- **The `delta_f` design point is an assumption.** Nothing here measures
  what differential spread a real 5.8 GHz UAV link sees. The table says
  what each interval tolerates, not what the channel will demand.
- **One echo amplitude (a=0.2, Rician K=14 dB) and one payload.** A
  weaker LOS component was not swept at these intervals.
- **`delta_f` is never converted to km/h.** UAV velocity,
  absolute/common Doppler (absorbed by CFO+CPE -- verified to 2000 Hz)
  and differential multipath Doppler are three separate quantities and
  are kept separate deliberately.

## Reproduce

```
examples/dmrs_doppler_study.py          --part sweep     # 25 dB matrix
examples/dmrs_doppler_postfix_study.py                   # post-timing-fix
examples/dmrs_doppler_15db_study.py                      # 15 dB
examples/dmrs_doppler_rate_study.py                      # 10 vs 20 MSps
examples/dmrs_doppler_delay_study.py                     # fixed 100 ns echo
```

Raw JSON and rendered tables under `debug/dmrs_doppler_*/`.
