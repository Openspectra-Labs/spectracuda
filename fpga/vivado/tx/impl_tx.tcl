# Standalone TX resource/timing measurement. No board I/O or RX changes.
set root [file normalize [file join [file dirname [info script]] .. ..]]  ;# fpga/
cd $root
set out "$root/build/vivado_tx"
file mkdir $out
file mkdir "$root/build/ip"
set part xc7a50tcsg325-1
create_project -in_memory -part $part
set_property target_language Verilog [current_project]
set xci "$root/build/ip/tx_xfft_256/tx_xfft_256.xci"
if {[file exists $xci]} {
 read_ip $xci
} else {
 create_ip -name xfft -vendor xilinx.com -library ip -version 9.1 -module_name tx_xfft_256 -dir "$root/build/ip"
 set_property -dict [list CONFIG.transform_length {256} CONFIG.implementation_options {pipelined_streaming_io} CONFIG.data_format {fixed_point} CONFIG.input_width {16} CONFIG.phase_factor_width {16} CONFIG.scaling_options {unscaled} CONFIG.rounding_modes {truncation} CONFIG.output_ordering {natural_order} CONFIG.throttle_scheme {nonrealtime} CONFIG.aresetn {true} CONFIG.target_clock_frequency {100}] [get_ips tx_xfft_256]
}
generate_target {synthesis simulation} [get_ips tx_xfft_256]
synth_ip [get_ips tx_xfft_256]
foreach f {tx_top tx_bit_domain tx_stream_cdc tx_freq_domain tx_ifft_engine tx_time_domain} {
 add_files "$root/rtl/tx/$f.v"
}
set_property file_type SystemVerilog [get_files "$root/src/*.v"]
read_xdc "$root/vivado/tx/tx_clocks.xdc"
synth_design -top tx_top -part $part -mode out_of_context
read_xdc -unmanaged "$root/vivado/tx/tx_cdc_post_synth.xdc"
write_checkpoint -force "$out/tx_synth.dcp"
report_utilization -file "$out/synth_util.txt"
report_utilization -hierarchical -file "$out/synth_util_hier.txt"
report_timing_summary -file "$out/synth_timing.txt"
opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force "$out/tx_routed.dcp"
report_utilization -file "$out/impl_util.txt"
report_utilization -hierarchical -file "$out/impl_util_hier.txt"
report_timing_summary -file "$out/impl_timing.txt"
report_exceptions -file "$out/exceptions.txt"
report_bus_skew -file "$out/bus_skew.txt"
report_clock_interaction -file "$out/clock_interaction.txt"
report_cdc -file "$out/cdc.txt"
report_drc -file "$out/drc.txt"
puts "TX IMPLEMENTATION COMPLETE: $out"
exit
