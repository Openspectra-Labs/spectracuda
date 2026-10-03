# Out-of-context synthesis of sc_sync_rtl alone, same part and clock as
# the HLS build of the identical function (hls/tcl/synth_sc_sync.tcl), so
# the LUT/DSP/BRAM numbers are directly comparable.
#
# The clock lives in sc_sync.xdc, not in a create_clock call here: that
# command needs an open design, and constraints must be present DURING
# synthesis to influence it, not applied afterwards.
set part xc7a50tcsg325-1
set root [file normalize [file dirname [info script]]]

create_project -force -in_memory -part $part
add_files "$root/src/sc_sync_rtl.v"
read_xdc "$root/sc_sync.xdc"
set_property top sc_sync_rtl [current_fileset]

synth_design -top sc_sync_rtl -part $part -mode out_of_context
report_utilization -file "$root/build/rtl_utilization.txt"
report_timing_summary -file "$root/build/rtl_timing.txt"
puts "=== done ==="
exit
