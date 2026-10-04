# TX routed measurement — UPDATED 2026-10-04 (after review fixes)

Final RTL (2 packet banks, 3 time banks, frame-cut recovery, registered CDC
read, post-synth FIFO max-delay). XC7A50T-1, OOC, Vivado 2025.2, real xfft.

| Block | LUT | FF | BRAM | DSP |
|---|---:|---:|---:|---:|
| TX total | 4446 (13.6%) | 4575 (7.0%) | 5 tiles (2 RAMB36 + 6 RAMB18) | 9 |
| BIT | 643 | 477 | 2 RAMB36 | 0 |
| CDC | 66 | 83 | 0 | 0 |
| FD | 585 | 119 | 1 RAMB18 | 0 |
| TD incl. IFFT | 3155 | 3892 | 5 RAMB18 | 9 |

Routed setup/hold: clk_bit 125 MHz +0.372/+0.040 ns; clk_sample 100 MHz
+0.035/+0.046 ns (worst: xfft output -> /128 round -> clamp -> timemem write,
12 levels; one register on the IFFT output would add margin). FIFO crossing
clk_bit->clk_sample: 38 endpoints, +6.375 ns, report_cdc 0 unsafe.

Verification on this RTL: real-xfft netlist xsim (run_top_xsim.sh) 3/3 at
40 MSPS incl. forced underrun and bad-symbol recovery, within 1 LSB on 15648
samples; TX RTL -> RX RTL loopback 5/5 bytes exact.

---
Original measurement (superseded):

# TX routed measurement — 2026-10-04

Vivado 2025.2, XC7A50T CSG325 -1, standalone out-of-context tx_top.
The actual standard Xilinx xfft v9.1 is included, configured as a 256-point
unscaled inverse transform through the TX wrapper. No behavioral FFT stub is
included in synthesis. No RX source was changed.

| Resource | Routed TX total | Device utilization |
|---|---:|---:|
| LUT | 3938 | 12.08% |
| FF | 4386 | 6.73% |
| Distributed RAM LUT | 845 | Included in LUT total |
| Shift register LUT | 534 | Included in LUT total |
| BRAM36-equivalent tiles | 4 | 5.33% |
| DSP48 | 9 | 7.50% |

Physical RAM is 1 RAMB36 + 6 RAMB18. BIT: 477 LUT/367 FF/1 RAMB36;
CDC: 65 LUT/50 FF; FD: 580 LUT/118 FF/1 RAMB18;
TD including IFFT: 2818 LUT/3847 FF/5 RAMB18/9 DSP.
Top reset synchronizers add 4 FF. Arithmetic sum of hierarchy LUT values can
differ slightly from reported total due to sharing/combining.

| Clock | Frequency | Routed setup WNS | Routed hold WHS |
|---|---:|---:|---:|
| BIT | 125 MHz | +0.503 ns | +0.085 ns |
| FD/TD | 100 MHz | +0.090 ns | +0.039 ns |

All constrained internal setup/hold paths pass in this standalone run.
This is not board-level timing signoff: I/O delays, external transport and
actual top-level clock placement are absent. Vivado warns HD.CLK_SRC is unset.

## Findings and limits

- FD mapped-symbol memory and TD time-sample memory map to LUTRAM despite
  block RAM attributes. Their current access/reset/output structure needs
  adjustment if BRAM mapping is required. These costs are included above.
- CDC report flags custom FIFO memory crossings and a multi-bit synchronizer;
  CDC signoff is pending. Gray-pointer max-delay and bus-skew constraints are
  present and bus-skew checks pass (+6.887 ns / +8.759 ns slack). The FIFO
  memory publication protocol still requires review; a positive timing summary
  alone does not establish CDC correctness.
- DRC has warnings, including RAM asynchronous-control checks. No implementation
  errors occurred. The pipeline requires whole-domain reset; partial reset is
  unsupported. RAM-control/reset findings need review before board integration.
- This includes current host-profile TX only: no AD9361 transport, payload
  RS/CRC RTL, protected header, inner interleaver2 or DMRS.
- Vendor IFFT synthesized and routed; its functional simulation remains pending.

Raw evidence is in ../build/vivado_tx/: impl_util.txt, impl_util_hier.txt,
impl_timing.txt, cdc.txt, drc.txt and tx_routed.dcp. Main log is
../build/tx_vivado.log. Reproduce using impl_tx.tcl from this directory;
inspect_tx.tcl adds detailed CDC and bus-skew reports from the routed checkpoint.
