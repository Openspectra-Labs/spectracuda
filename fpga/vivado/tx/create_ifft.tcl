# Run manually in Vivado from fpga/. Creates TX-local IP; no RX mutation.
set tx_root [file normalize [file join [file dirname [info script]] .. ..]]
create_project -in_memory -part xc7a50tcsg325-1
create_ip -name xfft -vendor xilinx.com -library ip -version 9.1 \
    -module_name tx_xfft_256 -dir [file join $tx_root build ip]
set_property -dict [list CONFIG.transform_length {256} \
    CONFIG.implementation_options {pipelined_streaming_io} \
    CONFIG.data_format {fixed_point} CONFIG.input_width {16} \
    CONFIG.phase_factor_width {16} CONFIG.scaling_options {unscaled} \
    CONFIG.rounding_modes {truncation} CONFIG.output_ordering {natural_order} \
    CONFIG.throttle_scheme {nonrealtime} CONFIG.aresetn {true}] [get_ips tx_xfft_256]
generate_target all [get_ips tx_xfft_256]
# Generation script only: synthesis, routing, timing and CDC signoff are separate gates.
