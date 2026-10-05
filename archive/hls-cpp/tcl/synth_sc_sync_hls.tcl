# Vivado out-of-context synthesis of the VITIS-HLS-GENERATED sc_sync RTL,
# so the HLS-vs-hand-written comparison is post-synth on BOTH sides.
#
# Until now the only HLS number on record was the csynth ESTIMATE
# (8081 LUT, hls/vitis_sc_sync_synth.log) measured against a real Vivado
# post-synth number for the hand-written Verilog (596 LUT,
# hls/rtl/build/rtl_utilization.txt). Those are different stages, and a
# csynth estimate is routinely 2x off in either direction, so the ratio
# between them was never a measurement.
#
# Same part and same 10 ns as synth_sc_sync.tcl, deliberately -- the
# point is comparability, so nothing here may differ except the source.
#
# The HLS top's clock port is `ap_clk`, NOT `clk`. hls/rtl/sc_sync.xdc
# constrains [get_ports clk], which on this design matches nothing and
# would leave the whole thing unconstrained while still reporting a
# confident-looking WNS. Hence a separate constraint written here.
#
# Run with cwd = the impl/verilog directory: the generated BRAM model
# does $readmemh("./sc_sync_buf_i_RAM_2P_BRAM_1R1W.dat"), a path relative
# to the process, so elaborating from anywhere else silently loses the
# memory init.
set part xc7a50tcsg325-1
set hls  /home/abhi/work/spectracuda/hls/sc_sync_proj/sol1/impl/verilog
set out  /home/abhi/work/spectracuda/hls/rtl/build

create_project -force -in_memory -part $part
add_files [glob $hls/*.v]
set_property top sc_sync [current_fileset]

set xdc "$out/sc_sync_hls.xdc"
set fd [open $xdc w]
puts $fd "create_clock -name ap_clk -period 10.000 \[get_ports ap_clk\]"
close $fd
read_xdc $xdc

synth_design -top sc_sync -part $part -mode out_of_context

# Did the clock actually attach? A create_clock that matched nothing
# leaves get_clocks empty and every path unconstrained.
set nclk [llength [get_clocks -quiet]]
puts "=== CLOCKS FOUND: $nclk -> [get_clocks -quiet] ==="

report_utilization              -file "$out/sc_sync_hls_util.txt"
report_utilization -hierarchical -file "$out/sc_sync_hls_util_hier.txt"
report_timing_summary           -file "$out/sc_sync_hls_timing.txt"

puts "=== CELL COUNTS ==="
puts "  total leaf cells : [llength [get_cells -hier -filter {IS_PRIMITIVE==1}]]"
puts "  black boxes      : [llength [get_cells -hier -filter {IS_BLACKBOX==1}]]"
puts "=== DONE ==="
exit
