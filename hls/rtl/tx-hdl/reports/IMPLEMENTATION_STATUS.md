# Initial implementation verification — 2026-10-04

Repository starting HEAD: fb551ae. Reference: pinned Python ad0a396.
Each results.json records exact source hashes and reference revision.

| Check | Result | Evidence |
|---|---|---|
| Standalone BIT | 36/36 exact group comparisons | ../build/af28f710c8ae/results.json |
| FD through asynchronous FIFO | 7680 exact bins in each of two stall modes | ../build/9b2b5df08537/results.json |
| Full TX, 125/100 MHz, 40 MSPS enables | Six frames, 15648 samples; no clipping or status errors | ../build/7103cfd144da/results.json |

BIT spans QPSK/16QAM/64QAM, six payload lengths (8 through 16384 bits),
with/without output stalls. FD covers training/header and all constellation
labels, changing frame identities and heavy stalls. Full-chain exercises mixed
lengths/modulations and at most two outstanding frame identities, with byte
input gaps. Source compilation treats Verilator warnings as fatal.

The full-chain IFFT is a simulation-only floating inverse DFT of quantized
bins. IQ tolerance is one LSB against that oracle; this is not vendor IFFT
bit accuracy. These results establish functional progress, not FPGA readiness.

## Remaining gates

1. Generate dedicated Xilinx IFFT and use its simulation model; confirm inverse
   configuration, result layout, latency and numerical error against reference.
2. Add hardware clock/CDC constraints, inspect CDC, and measure resources and
   whole-TX timing at BIT125/FD100/TD100; no such tools have been run here.
3. Complete AD9361-facing sample transport and explicit status crossings.
4. Add malformed descriptor/stream, FIFO reset/full, underrun and maximum-load
   deadline tests, including maximum supported frames and continuous traffic.
5. Perform frozen Python RX and RTL RX loopbacks before radio operation.
6. Review ownership/profile extension before enabling payload CRC/RS, protected
   header, interleaver2, puncturing or DMRS. These features are absent now.

Initial BIT has one committed packet bank. Inter-frame gaps are permitted;
gapless continuous transmission and all-host-profile compatibility are not
claimed. Existing RX modules were not modified.

Update: standalone Vivado synthesis/routing now completed; see VIVADO_MEASUREMENT.md.
Functional vendor-model simulation and board/CDC signoff remain pending.
