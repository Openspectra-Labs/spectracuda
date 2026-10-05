#!/bin/bash
# Run a Vitis HLS tcl script with the environment this machine needs.
#
# Reuses ~/work/ofdm-hls/lib_compat: Ubuntu 22+ dropped libncurses.so.5
# and libtinfo.so.5, which Vitis 2025.2 still links against. That shim
# directory is the only reason this project depends on ofdm-hls at all.
#
#   ./run_hls.sh tcl/synth_sc_sync.tcl
set -euo pipefail
XILINX_ROOT="${XILINX_ROOT:-/home/abhi/work/xilinx/2025.2}"
COMPAT="${HLS_LIB_COMPAT:-/home/abhi/work/ofdm-hls/lib_compat}"
export RDI_BINROOT="$XILINX_ROOT/Vitis/bin"
export RDI_APPROOT="$XILINX_ROOT/Vitis"
export TCL_LIBRARY="$XILINX_ROOT/tps/tcl/tcl8.6"
export XILINX_VIVADO="$XILINX_ROOT/Vivado"
export XILINX_VITIS="$XILINX_ROOT/Vitis"
export XILINX_HLS="$XILINX_ROOT/Vitis"
export PATH="$XILINX_ROOT/Vivado/bin:$XILINX_ROOT/Vitis/bin:$PATH"
export LD_LIBRARY_PATH="$COMPAT:$XILINX_ROOT/Vitis/lib/lnx64.o:$XILINX_ROOT/Vivado/lib/lnx64.o:${LD_LIBRARY_PATH:-}"
exec "$XILINX_ROOT/Vitis/bin/loader" -exec vitis_hls -f "$@"
