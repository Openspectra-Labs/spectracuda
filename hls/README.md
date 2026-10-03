# hls/ — spectracuda's OFDM chain for Vitis HLS

Approach **B** from [`docs/vitis-hls-ofdm-ip-plan.md`](../docs/vitis-hls-ofdm-ip-plan.md):
the HLS blocks implement **spectracuda's** frame format, so spectracuda's
Python — RF-validated between two Plutos — is the golden model.

This is deliberately *not* a continuation of `~/work/ofdm-hls/`. That
project is a different radio (Zadoff-Chu preamble, its own header, no
training symbol) with its own Python reference, and it has never been
RF-tested. Its FEC blocks are reusable (same K=7, 0171/0133); its
framing is not.

## Layout

```
gen/emit.py     Python: emits the CONFIG headers (constants, subcarrier
                tables, ROMs) from a constructed Ofdm object. Never emits
                DSP source code.
gen/golden.py   Python: dumps golden IQ + expected results from the real
                spectracuda pipeline. Never reimplements an algorithm --
                it calls the same blocks the library uses.
src/            hand-written HLS C++ (fixed; does not change per config)
src/generated/  emitted headers -- do not edit, regenerate
tb/             C-sim testbenches (plain C++, no Vitis needed to run)
golden/         generated vectors (snr<NN>_s<NN>/ per case)
tcl/            Vitis HLS synthesis scripts
```

## Running

```bash
make headers       # regenerate config headers from an Ofdm object
make golden        # regenerate vectors from spectracuda
make csim          # double-precision build: proves the ALGORITHM
make csim-fixed    # ap_fixed build: proves it survives quantization
make sweep         # word-length sweep across the full case grid
```

`csim` compiles with plain `g++` against Vitis' HLS headers
(`/home/abhi/work/xilinx/2025.2/Vitis/include`) — seconds per iteration,
versus minutes to launch `vitis_hls`. Synthesis uses the same sources.

## Status

| component | state |
|---|---|
| generator (`gen/emit.py`) | **working** -- 4 headers + validation, 17 tests |
| `sc_sync` (Schmidl-Cox detect) | **C-sim passing**, float and fixed, against Python |
| CFO estimate/correct | not started |
| CP strip + FFT | not started |
| channel est / equalize | not started |
| demap | not started |
| conv/Viterbi | reuse from ofdm-hls, pending bit-order check |
| Reed-Solomon | absent — still the open build-vs-buy item |

Nothing has been through `vitis_hls` synthesis yet. No resource numbers
exist, and none should be quoted until they do.

## The bug that justifies generating rather than hand-writing

`ap_fixed<16,1>` spans **[-1, +1)** -- so **1.0 is not representable in
it**. BPSK pilot values are exactly +1.0, some training values reach 1.0,
and `TRAIN_SCALE = 1/N` is exactly 1.0 when `N=1`. All three would have
silently saturated to 0.99997, with nothing in any log to say so.

Hence three types, not one:

| type | for | why |
|---|---|---|
| `sample_t` = `ap_fixed<16,1>` | signal datapath | normalized below 1.0 |
| `coef_t` = `ap_fixed<16,2>` | ROM constants | must hold exactly +-1.0 |
| `scale_t` = `ap_fixed<18,2>` | derived scalars | `1/N` is 1.0 at N=1 |

`gen/emit.py` checks every emitted constant against its type's range and
**raises `NotRepresentable` at generate time** rather than letting
`ap_fixed` saturate it. A hand-written table gets this wrong silently; a
generated one cannot.

## What the generator refuses

`Ofdm(...)` is a waveform specification, so the generator validates it
before emitting. It carries only the **HLS-specific delta** -- Python's
own constructor already rejects incoherent waveforms, so an invalid
config cannot reach here. What Python allows but fabric cannot express:

| rejected | why |
|---|---|
| `fft_size` not a power of 2 | numpy.fft handles any N and spectracuda has no such check; no FFT core can be built |
| strategy without an HLS block | e.g. `equalizer="zf"`, `sync="zadoff_chu"` |
| `fec1` other than `"none"` | Reed-Solomon is not in fabric |
| `cp_len=0` | no multipath guard; RX slot arithmetic degenerates |

All problems are reported together, not one per run.

## Two design notes worth not re-deriving

**Schmidl-Cox is cheaper in fabric than a matched filter.** It correlates
the signal against a delayed copy of *itself*, using the preamble's
two-identical-halves structure and never its content. So there is no
known-template ROM — unlike `ofdm-hls`'s ZC `sync_detect`, which needs a
200-entry complex one. `sc_sync.cpp` also compares candidates by
cross-multiplication rather than dividing, so the peak search needs no
divider; one divide runs per frame, at the end.

**The port source is the numba kernel, not the numpy path.**
`spectracuda/sync/_numba_schmidl_cox.py` is already the single-pass
sliding-window form (O(1) per sample, no prefix-sum arrays) — which is
what fabric wants. `schmidl_cox.py`'s ten full-array passes are a
batch-machine idiom with no hardware equivalent.

## Measured: word length for `sc_sync`

Sweep across 4 SNRs x 3 seeds straddling spectracuda's own 6-9 dB
detection cliff. Pass bar is agreement with **Python**, not with the true
frame start — Python itself lands ~4 samples early at 6 dB on some seeds,
and the block is required to reproduce that.

| sample width | cases matching Python | worst offset |
|---|---|---|
| 4 | 7 / 12 | 6 samples |
| 6 | 10 / 12 | 4 samples |
| **8** | **12 / 12** | 0 |
| 10-18 | 12 / 12 | 0 |

**8 bits is the floor; keep `ap_fixed<16,1>` anyway.** Narrowing buys
nothing on an Artix-7: the DSP48E1 is 25x18, so any operand up to 18 bits
costs exactly one DSP slice, and the 2L+1 sample delay line fits in one
BRAM36 at either width. 16 also matches `ofdm-hls`'s existing `sample_t`.

Measure at the cliff or not at all: a sweep run only at 20 dB reports
that 4 bits is fine, because there the channel noise is 30 dB above the
quantization noise and the experiment cannot see the thing it is meant
to measure.

## Known gap: batch argmax vs streaming detection

`sc_sync` currently takes a whole buffer and returns the global argmax,
matching Python exactly. Real fabric cannot wait for a buffer — it needs
threshold-then-peak-hold on a free-running stream (what `ofdm-hls`'s
`sync_detect` does, and what `Ofdm.rx_streaming()` does on the Python
side). That is a separate change with its own failure modes and is not
mixed into this one: prove the arithmetic first, then the state machine.
