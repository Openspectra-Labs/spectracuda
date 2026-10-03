# Out-of-context synthesis of cp_fft (CP strip + Xilinx FFT core),
# same part and clock as every other block here so the numbers compare.
#
# Unlike the other blocks this one instantiates an IP, so the .xci must be
# read in and its synthesis target generated before synth_design; adding
# only the .v would leave xfft_256 as a black box and undercount by the
# entire cost of the core, which is most of the block.
set part xc7a50tcsg325-1
set root [file normalize [file dirname [info script]]]

create_project -in_memory -part $part
set_property target_language Verilog [current_project]

read_ip "$root/build/ip/xfft_256/xfft_256.xci"
generate_target synthesis [get_ips xfft_256]
synth_ip [get_ips xfft_256]

add_files "$root/src/cp_fft.v"
read_xdc "$root/sc_sync.xdc"
set_property top cp_fft [current_fileset]

synth_design -top cp_fft -part $part -mode out_of_context
report_utilization -file "$root/build/cp_fft_util.txt"
report_utilization -hierarchical -file "$root/build/cp_fft_util_hier.txt"
report_timing_summary -file "$root/build/cp_fft_timing.txt"
puts "=== CP_FFT SYNTH DONE ==="
exit
