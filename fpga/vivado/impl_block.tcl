# Out-of-context synth + place + route of ONE block at a given clock period,
# for per-domain timing (e.g. rx_bit_domain at 200 MHz while TD/FD stay at
# 100 MHz). Same part as every other measurement.
#
#   vivado -mode batch -source impl_block.tcl -tclargs <top> <period_ns> <src.v> [more.v ...]
#
# Reports: build/<top>_<period>ns_{timing,util,util_hier}.txt
set part   xc7a50tcsg325-1
set root   [file normalize [file join [file dirname [info script]] ..]]  ;# fpga/
set top    [lindex $argv 0]
set period [lindex $argv 1]
set srcs   [lrange $argv 2 end]
create_project -force -in_memory -part $part
foreach f $srcs { add_files "$root/$f" }
set_property include_dirs [list "$root/rtl" "$root/rtl/rx" "$root/rtl/common" "$root/rtl/generated"] [current_fileset]
foreach f [get_files -quiet {*grid_extract.v *header_decode.v *frame_sync.v *pilot_cpe.v *rx_top.v *rx_time_domain.v *rx_freq_domain.v *sync_fifo_fwft.v *rx_header.v *rx_bit_decoder.v *rx_bit_domain.v *cdc_*.v}] {
    set_property file_type SystemVerilog $f
}
set xdc [file join $root build ${top}_${period}ns.xdc]
set fh [open $xdc w]
puts $fh "create_clock -name clk -period $period \[get_ports clk\]"
close $fh
read_xdc $xdc
set_property top $top [current_fileset]
# Optional parameter overrides and report tag, from the environment:
#   GENERICS="PM_W=10 NORM=0"   TAG=soft_mod10
set gen {}
if {[info exists ::env(GENERICS)] && $::env(GENERICS) ne ""} {
    foreach g $::env(GENERICS) { lappend gen -generic $g }
}
synth_design -top $top -part $part -mode out_of_context {*}$gen
opt_design
place_design
phys_opt_design
route_design
set tag "${top}_${period}ns"
if {[info exists ::env(TAG)] && $::env(TAG) ne ""} { set tag "${top}_$::env(TAG)_${period}ns" }
report_timing_summary -max_paths 20 -file "$root/build/${tag}_timing.txt"
# one line per failing endpoint (worst path to each), to see WHAT fails
report_timing -max_paths 5000 -nworst 1 -slack_lesser_than 0 -path_type summary \
    -file "$root/build/${tag}_endpoints.txt"
report_utilization -file "$root/build/${tag}_util.txt"
report_utilization -hierarchical -file "$root/build/${tag}_util_hier.txt"
puts "=== done $tag ==="
exit
