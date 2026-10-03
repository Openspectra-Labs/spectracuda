# Generate the Xilinx FFT LogiCORE for spectracuda's 256-point transform.
#
# CONFIG MATCHES openofdm (reference/openofdm/verilog/Xilinx/zynq/xfft/
# xfft_v9.xci), which is proven in openwifi, with transform_length raised
# from their 64 (802.11) to our 256:
#
#     unscaled + natural_order + 16-bit input + truncation
#
# UNSCALED is the load-bearing choice, and it is not the obvious one.
# ~/work/ofdm-hls uses scaled with scale_sch=0xAA and argues at length
# that 0x55 would overflow. That reasoning is correct FOR A SCALED CORE,
# which must discard bits every stage. Unscaled removes the question:
# the core widens the datapath instead, so no schedule can be wrong.
#
# It also makes verification direct -- numpy's forward FFT is unscaled
# too (spectracuda/ofdm/fft.py), so RTL output correlates with the golden
# model with no scale factor to reconcile. openofdm narrows to 16 bits
# downstream instead (sync_long.v takes fft_out_re[22:7]).
#
# Non-project (in-memory) flow: synth_ip is unsupported in project mode
# and warns.
set part xc7a50tcsg325-1
set root [file normalize [file dirname [info script]]]
file mkdir "$root/build/ip"

create_project -in_memory -part $part
set_property target_language Verilog [current_project]

create_ip -name xfft -vendor xilinx.com -library ip -version 9.1 \
    -module_name xfft_256 -dir "$root/build/ip"
set_property -dict [list \
    CONFIG.transform_length {256} \
    CONFIG.implementation_options {pipelined_streaming_io} \
    CONFIG.data_format {fixed_point} \
    CONFIG.input_width {16} \
    CONFIG.scaling_options {unscaled} \
    CONFIG.rounding_modes {truncation} \
    CONFIG.output_ordering {natural_order} \
    CONFIG.target_clock_frequency {100} \
] [get_ips xfft_256]

generate_target {synthesis simulation} [get_ips xfft_256]
synth_ip [get_ips xfft_256]

puts "=== XFFT_256 CONFIG ==="
foreach p {C_INPUT_WIDTH C_OUTPUT_WIDTH C_NFFT_MAX C_HAS_SCALING C_ARCH} {
    if {![catch {set v [get_property CONFIG.$p [get_ips xfft_256]]}]} {
        puts "  $p = $v"
    }
}
puts "=== XFFT_256 DONE ==="
exit
