# ============================================================
# synth_sc_sync.tcl -- Schmidl & Cox preamble detector
#   top   : sc_sync
#   part  : xc7a50tcsg325-1  (matches ~/work/ofdm-hls's target)
#   clock : 10 ns = 100 MHz, the OFDM system clock
#
# SC_SYNC_USE_FIXED is mandatory here. The default (double) build exists
# only so C-sim can prove the algorithm against Python without
# quantization in the way -- synthesizing it would infer double-precision
# floating-point cores and tell us nothing useful about the real block.
# ============================================================
open_project -reset sc_sync_proj
set_top sc_sync
add_files src/sc_sync.cpp -cflags "-DSC_SYNC_USE_FIXED -Isrc"
add_files -tb tb/sc_sync_tb.cpp -cflags "-DSC_SYNC_USE_FIXED -Isrc"
open_solution sol1 -reset
set_part xc7a50tcsg325-1
create_clock -period 10
config_compile -pipeline_loops 0
csynth_design
puts "\n=== sc_sync Synthesis Report ==="
set rpt [open sc_sync_proj/sol1/syn/report/sc_sync_csynth.rpt r]
puts [read $rpt]
close $rpt
close_project
exit
