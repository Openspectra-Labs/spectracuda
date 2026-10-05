# fpga/ — the Verilog OFDM modem

Hand-written Verilog RX and TX for the spectracuda waveform, target XC7A50T
(Artix-7) on the Spectra M.2 SDR. The Python package (`spectracuda/`) is the
golden model: every block is checked against it. RS / CRC / MAC stay on the
host. Current state and the comparison with OpenOFDM:
[docs/2026-10-05-hdl-rx-tx-status.md](docs/2026-10-05-hdl-rx-tx-status.md).

## Layout

| Folder | Contents |
|---|---|
| `rtl/rx/` | Receiver: `rx_top` = time domain (sync, CFO, FFT) -> frequency domain (channel estimate, MMSE, CPE, soft demapper) -> bit domain (Viterbi, deinterleave) |
| `rtl/tx/` | Transmitter: `tx_top` = bit domain (header, interleaver, conv code) -> frequency domain (mapper, grid) -> time domain (IFFT, CP, preamble) |
| `rtl/common/` | Shared: Viterbi decoders (hard, overlapped, soft), CDC FIFOs, sync FIFO |
| `rtl/vendor/` | Small helper blocks taken from OpenOFDM |
| `rtl/generated/` | Tables and constants generated from Python (`gen/`); `generated/tx/` for TX |
| `tb/rx/`, `tb/tx/` | Testbenches (Verilator unless noted) |
| `sim/`, `sim/tx/` | Test runners (Python / shell) |
| `gen/` | Table generators: `python -m fpga.gen.emit_rtl`, `emit_soft_table` |
| `vivado/`, `vivado/tx/` | Synthesis / implementation tcl and constraints (`.xdc`) |
| `fixtures/frames/` | Captured full-frame interface dumps for the stage testbenches |
| `fixtures/blocks/` | Per-block golden vectors |
| `docs/` | `rundown.md` (start here), RX architecture, status, TX docs |
| `refs/` | Pinned Python reference worktrees (ignored; `sim/setup_ref.sh`) |
| `build/` | All generated output (ignored) |

All runners work from any directory; they locate `fpga/` themselves.

## Toolchain

| Tool | Path |
|---|---|
| Vivado 2025.2 | `/home/abhi/work/xilinx/2025.2/Vivado/bin/vivado` (not on PATH) |
| xsim | `xvlog`, `xelab`, `xsim` in `/home/abhi/work/xilinx/2025.2/Vivado/bin/`; glbl: `…/Vivado/data/verilog/src/glbl.v` |
| Verilator 5.020 | `/usr/bin/verilator` (default simulator) |
| Python | `.venv/bin/python` at the repo root |

Part XC7A50T CSG325-1. More detail (licensing, xsim with IP netlists) in the
repo-root `CLAUDE.md`.

## Common commands (from the repo root)

```sh
fpga/sim/setup_ref.sh                          # once: pinned Python reference
.venv/bin/python fpga/sim/run_frame.py --bits 2000 --modem qam64   # one frame, RX vs Python
fpga/sim/rate_matrix.sh                        # RX regression across sample rates
.venv/bin/python fpga/sim/run_fd_stage.py      # FD stage suite
.venv/bin/python fpga/sim/run_bd_stage.py      # BD stage suite
.venv/bin/python fpga/sim/run_demap_soft.py    # soft demapper vs Python table model
.venv/bin/python fpga/sim/tx/run_bit_tests.py  # TX bit domain
.venv/bin/python fpga/sim/tx/run_top_tests.py  # TX full chain (+ underrun / bad symbol)
.venv/bin/python fpga/sim/tx/run_loopback.py   # TX RTL -> RX RTL
/home/abhi/work/xilinx/2025.2/Vivado/bin/vivado -mode batch -nojournal -nolog -source fpga/vivado/impl_rx_top.tcl   # RX P&R
```
