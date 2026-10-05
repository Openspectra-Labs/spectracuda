# TX HDL -- v3 frame format

**TX_BIT 125 MHz → asynchronous stream → TX_FD 100 MHz → TX_TD 100 MHz**.
Format = the pinned Python reference `hls/rtl/golden_ref_v3.py` (protected
header, interleaver2, DMRS, hard decision); the RX in `../src` decodes the
same format.

- `src/tx_bit_domain.v`: two packet banks; v3 header (14 info bytes + crc16
  -> conv_v27 + tail -> scramble -> 268 of 432 slots over 2 BPSK symbols);
  outer byte interleaver, conv_v27 + tail, padding filler; interleaver2 per
  symbol (ping-pong bit buffers, 2 bits/clock column-major read); DMRS
  tokens (stype 4) after every 16/32/64 data symbols, trailing one
  suppressed (spectracuda framing/dmrs.py).
- `src/tx_stream_cdc.v`: 32-entry asynchronous coded-group FIFO, registered
  read side.
- `src/tx_freq_domain.v`: mapper and two resource-grid banks; TRAIN and DMRS
  tokens both produce the training grid; pilots and nulls.
- `src/tx_time_domain.v`, `src/tx_ifft_engine.v`: IFFT (registered output),
  quantization, CP, preamble, three time banks, frame-cut recovery.
- `src/tx_top.v`: domain instances and explicit connections.

See [INTERFACES.md](INTERFACES.md) for the contract. Payload CRC/RS and
MAC remain host responsibilities; C2 regions are not supported.

From this directory, run:

```sh
../../../.venv/bin/python -B run_tx_v3.py      # 8 frames vs pinned Python (all ROMs regenerated),
                                               # + forced underrun + bad symbol, + Python RX on RTL output
../../../.venv/bin/python -B run_freq_tests.py # FD alone, incl. bad-symbol injection
../../../.venv/bin/python -B run_loopback.py   # TX RTL -> RX RTL (../src), bytes back exact
./run_top_xsim.sh                              # newest run_tx_v3 fixtures on the REAL xfft netlist
```

The runners verify the pinned worktree (`../golden_ref_v3.py`), generate
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
