#!/bin/bash
# Full-chain TX on the REAL xfft core (Vivado-generated structural sim
# netlist, unisims) in xsim: vendor latency, truncation and backpressure.
# Uses the fixtures of the newest run_top_tests.py build. Needs the IP
# generated once (impl_tx.tcl or ip/create_ifft.tcl).
#   ./run_top_xsim.sh            -> normal, underrun_frame2, bad_symbol_frame3
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)  # fpga/
VIV=${VIVADO:-/home/abhi/work/xilinx/2025.2/Vivado}
NET=$HERE/build/ip/tx_xfft_256/tx_xfft_256_sim_netlist.v
FIX=$(ls -td "$HERE"/build/*/ | while read d; do [ -f "$d/expected.txt" ] && echo "$d" && break; done)
[ -f "$NET" ] || { echo "missing $NET"; exit 2; }
[ -n "$FIX" ] || { echo "run run_top_tests.py first"; exit 2; }
OUT=$HERE/build/xsim_$(date +%s)_$$; mkdir -p "$OUT"; cd "$OUT"
ln -s "$HERE/reference" reference
"$VIV/bin/xvlog" -sv "$HERE"/src/*.v "$HERE/tb/tx_top_tb.v" "$NET" "$VIV/data/verilog/src/glbl.v" > comp.log 2>&1 || { echo "xvlog failed $OUT"; exit 1; }
"$VIV/bin/xelab" -L unisims_ver tx_top_tb glbl -s top -timescale 1ns/1ps > elab.log 2>&1 || { echo "xelab failed $OUT"; exit 1; }
rc=0
for run in "normal:" "underrun_frame2:-testplusarg abort_frame=2" "bad_symbol_frame3:-testplusarg bad_frame=3"; do
  name=${run%%:*}; extra=${run#*:}
  "$VIV/bin/xsim" top -R -testplusarg input=$FIX/input.txt -testplusarg expected=$FIX/expected.txt $extra > sim_$name.log 2>&1
  line=$(grep -m1 -E "^PASS frames=|Fatal|FATAL" sim_$name.log)
  case "$line" in PASS*) echo "$name PASS $line";; *) echo "$name FAIL $line"; rc=1;; esac
done
echo "xsim results: $OUT"; exit $rc
