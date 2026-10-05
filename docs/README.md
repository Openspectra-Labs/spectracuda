# docs/ — what is where

The Python modem (`spectracuda/`) is the golden model; the FPGA modem lives
in `fpga/` with its own docs in `fpga/docs/`. Files directly in `docs/` are
the published documentation site (Sphinx, `index.md`); `reports/` and
`plans/` are the working record. This README is not part of the site.

## Published site (`docs/`)

| File | What it is |
|---|---|
| [index.md](index.md) | Site landing page and table of contents |
| [installation.md](installation.md) | Installing spectracuda |
| [architecture.md](architecture.md) | Software architecture: blocks, backends, pipeline |
| [spectracuda-phy-specification.md](spectracuda-phy-specification.md) | PHY specification and validation record |
| [mac.md](mac.md) | MAC layer: TM / UM / AM modes |
| [fec.md](fec.md) | Forward error correction overview |
| [ldpc.md](ldpc.md) | LDPC scheme: implementation plan |
| [fec-c-lib-acceleration.md](fec-c-lib-acceleration.md) | Native C FEC acceleration |
| [hardware-validation.md](hardware-validation.md) | Real-RF validation results |
| [comparison.md](comparison.md) | How spectracuda compares |
| [liquid-dsp-api-inventory.md](liquid-dsp-api-inventory.md) | liquid-dsp public API inventory |
| [todo.md](todo.md) | Gaps to close |
| [book/](book/) | The OFDM Field Guide (chapters of the site) |

## Reports (`docs/reports/`) — dated findings, newest last

| File | Topic |
|---|---|
| [2026-08-27-neon-viterbi-and-rx-throughput.md](reports/2026-08-27-neon-viterbi-and-rx-throughput.md) | NEON Viterbi kernel + RX throughput |
| [2026-09-06-rx-packet-loss-and-ism-band-characterization.md](reports/2026-09-06-rx-packet-loss-and-ism-band-characterization.md) | RX packet loss + 2.4/5 GHz ISM band |
| [2026-09-09-fast-viterbi-kernel.md](reports/2026-09-09-fast-viterbi-kernel.md) | "fast" Viterbi decoder |
| [2026-09-09-numba-cfo-kernel.md](reports/2026-09-09-numba-cfo-kernel.md) | Numba CFO estimate/correct kernels |
| [2026-09-09-numba-sync-kernel.md](reports/2026-09-09-numba-sync-kernel.md) | Numba sync/CFO kernel |
| [2026-09-09-rx-streaming-partial-preamble-fix.md](reports/2026-09-09-rx-streaming-partial-preamble-fix.md) | rx_streaming() frame-loss fix |
| [2026-09-18-rssi-measurement-window-fix.md](reports/2026-09-18-rssi-measurement-window-fix.md) | RSSI measurement window on real Pluto |
| [2026-09-20-dmrs-static-channel-cost.md](reports/2026-09-20-dmrs-static-channel-cost.md) | DMRS static-channel cost (simulation artifact) |
| [2026-09-20-lambda-m2-sdr-spectracuda-engineering-journey.md](reports/2026-09-20-lambda-m2-sdr-spectracuda-engineering-journey.md) | M.2 SDR to UAV link: engineering journey |
| [2026-09-21-dmrs-differential-doppler-characterization.md](reports/2026-09-21-dmrs-differential-doppler-characterization.md) | DMRS interval vs differential Doppler |
| [2026-09-21-dmrs-interval-operating-guidance.md](reports/2026-09-21-dmrs-interval-operating-guidance.md) | DMRS interval per sample rate |
| [2026-09-21-multipath-severity-characterization.md](reports/2026-09-21-multipath-severity-characterization.md) | Multipath severity |

## Plans and reviews (`docs/plans/`)

| File | Topic |
|---|---|
| [2026-09-10-ldpc-numba-minsum-kernel-plan.md](plans/2026-09-10-ldpc-numba-minsum-kernel-plan.md) | LDPC Numba min-sum decoder: plan and outcome |
| [2026-09-20-critical-c2-region-plan.md](plans/2026-09-20-critical-c2-region-plan.md) | Protected C2 region (MAVLink) |
| [2026-09-20-dmrs-periodic-channel-refresh-plan.md](plans/2026-09-20-dmrs-periodic-channel-refresh-plan.md) | Periodic DMRS |
| [flexlink-spec-review.md](plans/flexlink-spec-review.md) | FlexLink PHY spec review |
| [hexagon-fec-offload-plan.md](plans/hexagon-fec-offload-plan.md) | Hexagon DSP FEC offload |
| [pluto-radio-api-plan.md](plans/pluto-radio-api-plan.md) | Simple PlutoSDR API |

## Elsewhere

- **FPGA:** [../fpga/README.md](../fpga/README.md) and `fpga/docs/`
  (rundown, RX architecture, status vs OpenOFDM, TX docs).
- **Archived Vitis-HLS flow:** `archive/hls-cpp/` (incl. its plan).
