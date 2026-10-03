# Simulate cp_fft (CP strip + Xilinx FFT) under Vivado xsim.
#
# xsim rather than Verilator for this block ALONE: the generated FFT core
# is VHDL (build/ip/xfft_256/sim/xfft_256.vhd) and Verilator does not read
# VHDL. Every other block in hls/rtl stays on Verilator for the fast loop;
# only the vendor core forces the slower simulator.
#
# launch_simulation is used rather than hand-driving xvhdl/xvlog/xelab so
# Vivado compiles the xfft_v9_1 support library itself -- doing that by
# hand is where this flow usually breaks.
set part xc7a50tcsg325-1
set root [file normalize [file dirname [info script]]]

create_project -force fft_sim "$root/build/fft_sim" -part $part
set_property target_language Verilog [current_project]

read_ip "$root/build/ip/xfft_256/xfft_256.xci"
generate_target simulation [get_ips xfft_256]

add_files "$root/src/cp_fft.v"
add_files -fileset sim_1 "$root/tb/cp_fft_tb.v"
set_property include_dirs [list "$root"] [get_filesets sim_1]
set_property top cp_fft_tb [get_filesets sim_1]
set_property -name {xsim.simulate.runtime} -value {all} \
    -objects [get_filesets sim_1]

# The testbench reads and writes relative paths, so run where they live.
set_property -name {xsim.simulate.xsim.more_options} -value {} \
    -objects [get_filesets sim_1]

launch_simulation -simset sim_1 -mode behavioral
puts "=== SIM DONE ==="
exit
