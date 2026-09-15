# spectracuda

## Design a custom OFDM link in about 15 lines

Every real OFDM radio link needs the same stack: a synchronizer that finds
the frame in a stream of noise, a CFO estimator that undoes the clock
mismatch between two independent radios, channel estimation and
equalization that undo the actual RF channel, a modem, forward error
correction, and — if the link needs to move more than one packet reliably —
a MAC layer above all of it handling segmentation, acknowledgment, and
retries. Building that stack from scratch, or hand-wiring a general-purpose
SDR toolkit's separate pieces together, is normally a multi-week
undertaking. spectracuda collapses it to this:

```python
from spectracuda.pipeline import Ofdm
from spectracuda.mac import MacLink
import numpy as np

ofdm = Ofdm(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    modem="qpsk", fec="conv_v27", crc="crc32",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
)

link = MacLink(ofdm, mode="um")
link.bind()

payload = np.random.default_rng(0).integers(0, 2, size=4000).astype("uint8")
delivered = link.send(payload)   # segmentation, PHY tx/rx, reassembly -- one call
```

That's a complete link — real Schmidl-Cox synchronization and CFO
correction, LS channel estimation, MMSE equalization, QPSK modulation,
convolutional FEC, CRC, and MAC-layer framing with an actual bind handshake
— not a loopback demo with the hard parts stubbed out. Every string above
(`sync=`, `cfo=`, `channel_estimator=`, `equalizer=`, `modem=`, `fec=`,
`mode=`) is a real, independently swappable strategy, not a fixed pipeline
with tunable parameters: swap `sync="schmidl_cox"` for `sync="zadoff_chu"`,
`fec="conv_v27"` for one of twelve 802.11n LDPC variants or `fec="rs_m8"`,
`mode="um"` for `"tm"` or acknowledged-retransmission `"am"` — the rest of
the link doesn't change. The same object graph runs on a plain CPU
(`backend="numpy"`, no GPU required) or a Jetson/CUDA GPU (`backend="cupy"`)
with no code change beyond that one string, and this exact link design has
been run and validated over the air on real PlutoSDR hardware, not only in
simulation — see {doc}`hardware-validation` for the actual numbers.

## From an OFDM idea to real RF without MATLAB

spectracuda is intended to shorten the complete custom-radio development
loop, not only the simulation step. A link can be designed, impaired, and
tested on a normal development machine, then run with the same PHY and frame
format on a Raspberry Pi 5 connected to an ADALM-PLUTO. The project includes
the pieces that are otherwise often assembled across a MATLAB model, generated
code, and separate radio scripts:

- configurable OFDM framing, pilots, cyclic prefix, modulation, CRC, FEC,
  synchronization, CFO correction, channel estimation, and equalization;
- reproducible AWGN, multipath, CFO, fixed-point, and packet-loss experiments;
- batch and arbitrary-chunk streaming receive APIs;
- TM/UM/AM-style MAC behavior, segmentation, binding, status reporting, and
  retransmission;
- NumPy reference paths plus Numba, native C/SIMD, CuPy, and experimental
  FPGA/HLS implementation paths;
- runnable PlutoSDR transports and two-node examples rather than a
  simulation-only hardware placeholder.

MATLAB remains useful for teams that already depend on its toolboxes, but it
is not required anywhere in the spectracuda workflow. The Python source is the
executable reference model, the test oracle for optimized implementations,
and the code that runs on the radio host. That removes a translation boundary:
an algorithm does not have to be recreated in a second environment before it
can be exercised over the air.

This workflow has been used on two physically separate Raspberry Pi 5 +
ADALM-PLUTO nodes over real RF. The recorded tests cover QPSK, 16-QAM, and
64-QAM, concatenated Reed--Solomon + convolutional FEC, LS channel estimation,
MMSE equalization, streaming frame capture, and frequency sweeps in real
interference. Those results, including the observed failures and current
limits, are reported in {doc}`hardware-validation`.

GPU-accelerated, liquid-dsp-inspired SDR PHY (+ MAC) framework for NVIDIA
Jetson (Orin Nano first, NX/AGX Orin and desktop CUDA GPUs as additional
targets).

Not a CUDA port of liquid-dsp. liquid-dsp never exposed swappable
`channel_estimator=`/`equalizer=`/`cfo=` strategies -- those are algorithms
buried inside monolithic C objects, not interchangeable components.
spectracuda extracts the algorithms liquid-dsp *does* have a reference
implementation for, ports the ones that are self-contained (CRC, the
Zadoff-Chu-adjacent `qdetector` idea, the `interleaver` algorithm),
re-derives the ones that only exist wrapped around an external C library
(Viterbi, Reed-Solomon), and designs the rest from standard references
where liquid-dsp has no precedent at all (LS/MMSE channel estimation,
ZF/MMSE equalization, LDPC, the MAC layer) -- everything batch-first,
GPU-first, with a NumPy fallback so it runs anywhere.

Source: [github.com/iottrends/spectracuda](https://github.com/iottrends/spectracuda)

## Just the PHY, for finer control

`Ofdm` is a complete, standalone tx+rx object on its own — reach for it
directly (skip `MacLink`/`Mac` entirely) when a raw one-frame-at-a-time
link, without MAC-layer segmentation or a bind handshake, is all you need:

```python
from spectracuda.pipeline import Ofdm
import numpy as np

ofdm = Ofdm(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    modem="qpsk", fec="conv_v27", crc="crc32",
    sync="schmidl_cox", cfo="schmidl_cox",
    channel_estimator="ls", equalizer="mmse",
)

bits = np.random.default_rng(0).integers(0, 2, size=(1, 64)).astype("uint8")
tx_iq = ofdm.generate_frame(bits)      # -> (1, n_samples) complex64
result = ofdm.rx_process(tx_iq)        # -> dict, stable key set

result["frame_found"]   # True
result["bits"]          # decoded payload bits
result["crc_valid"]     # per-item bool array
result["evm"], result["rssi_db"], result["cfo_estimate"], result["header"]
```

See {doc}`book/ch01_ofdm_object` for the full walkthrough, {doc}`mac` for
the MAC layer behind the `MacLink` example above, {doc}`installation` to
actually get this running, or the
[README](https://github.com/iottrends/spectracuda#readme) for every
swappable strategy.

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation
```

```{toctree}
:maxdepth: 1
:caption: Real-World Validation

hardware-validation
```

```{toctree}
:maxdepth: 2
:caption: The OFDM Field Guide — Part I, the PHY chain

book/ch01_ofdm_object
book/ch02_synchronization
book/ch03_cfo
book/ch04_channel_estimation
book/ch05_modem
book/ch06_fec_ldpc
book/ch07_framing
book/ch08_streaming
```

```{toctree}
:maxdepth: 2
:caption: The OFDM Field Guide — Part II, the MAC layer

book/ch09_mac_modes
book/ch10_two_radios
book/ch11_binding_and_quality
book/ch12_bidirectional_am
book/roadmap
```

```{toctree}
:maxdepth: 1
:caption: Reference

architecture
comparison
fec
fec-c-lib-acceleration
hexagon-fec-offload-plan
mac
ldpc
liquid-dsp-api-inventory
todo
```
