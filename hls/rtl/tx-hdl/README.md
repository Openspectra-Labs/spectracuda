# Standalone frozen-format TX HDL

Initial implementation: **TX_BIT 125 MHz → asynchronous stream → TX_FD
100 MHz → TX_TD 100 MHz**. TX processing hardware and IFFT are independent
of RX. All work and generated artifacts live in this directory.

Implemented:

- `src/tx_bit_domain.v`: two committed byte-packet banks (next frame captured
  while the current one is encoded), frozen PHY header,
  outer byte interleaver, rate-1/2 K=7 convolutional coding, six tail bits,
  symbol padding and metadata-bearing coded groups.
- `src/tx_stream_cdc.v`: 32-entry asynchronous coded-group FIFO, registered
  read side.
- `src/tx_freq_domain.v`: mapper and two resource-grid banks, training,
  header/data placement, fixed pilots and nulls.
- `src/tx_time_domain.v`, `src/tx_ifft_engine.v`: buffered inverse transform,
  quantization, CP, preamble and sample scheduling.
- `src/tx_top.v`: domain instances and explicit connections.

See [INTERFACES.md](INTERFACES.md) for the actual contract. The initial profile
is the RX regression format at Python `ad0a396`: CRC none, RS none, byte outer
interleaver, conv_v27, QPSK/16QAM/64QAM payload, uncoded BPSK header. Payload
CRC/RS and MAC remain host responsibilities; other profiles are not supported.
Protected header, inner interleaver2 and DMRS are later format changes.

From this directory, run:

```sh
../../../.venv/bin/python -B run_bit_tests.py
../../../.venv/bin/python -B run_freq_tests.py
../../../.venv/bin/python -B run_top_tests.py   # normal, forced underrun, bad symbol
../../../.venv/bin/python -B run_loopback.py    # TX RTL -> RX RTL (../src), bytes back exact
./run_top_xsim.sh                               # same 3 top scenarios on the REAL xfft netlist
```

The runners verify the pinned worktree in `../build/spectracuda_ref`, generate
local vectors and save revision hashes, compiler logs and results under local
`build/<run-id>/`. They require NumPy and Verilator. Top-level simulation uses
`tb/tx_xfft_256_model.v`, a floating inverse-DFT model, **not** the vendor's
bit-accurate FFT model. IQ comparisons permit one LSB against this model.

`impl_tx.tcl` generates and implements the dedicated vendor core and TX top.
Standalone synthesis and routing have now completed; see
[the measured resources/timing](reports/VIVADO_MEASUREMENT.md). Real-IFFT simulation,
CDC constraints and TX→RTL RX loopback are done. Before hardware use: the
AD9361 transport, TX→Python RX, and longer continuous maximum-load runs.
Frames are not gapless (TD waits for the next frame's first two symbols).
CDC: the FIFO-memory -> read-register path is bounded by a post-synthesis
max-delay (`tx_cdc_post_synth.xdc`, applied after `synth_design`).
