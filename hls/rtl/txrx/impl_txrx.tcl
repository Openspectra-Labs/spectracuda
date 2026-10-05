# Out-of-context place & route of the COMPLETE modem (RX + TX, v3 format)
# on XC7A50T, clk_sample 100 MHz / clk_bit 125 MHz. Not a bitstream: no
# pins; proves fit + timing before the Spectra M.2 integration.
set part xc7a50tcsg325-1
set here [file normalize [file dirname [info script]]]
set rtl  [file normalize "$here/.."]
set tx   "$rtl/tx-hdl"
set out  "$here/build"
file mkdir $out
create_project -in_memory -part $part
set_property target_language Verilog [current_project]
read_ip "$rtl/build/ip/xfft_256/xfft_256.xci"
read_ip "$tx/build/ip/tx_xfft_256/tx_xfft_256.xci"
generate_target synthesis [get_ips]
synth_ip [get_ips]
foreach f {rx_top rx_time_domain rx_freq_domain sync_fifo_fwft rx_bit_domain
           sc_sync_rtl frame_sync cfo_estimate cfo_correct cordic_rot cordic_vec cp_fft
           grid_extract ls_chanest mmse_eq pilot_cpe header_decode_v3 demapper demapper_soft llr_scale viterbi_dec
           viterbi_dec_ovl viterbi_dec_soft cdc_async_fifo cdc_bundle cdc_reset_sync il2_deint
           deinterleaver} {
    add_files "$rtl/src/$f.v"
}
foreach f {tx_top tx_bit_domain tx_stream_cdc tx_freq_domain tx_ifft_engine tx_time_domain} {
    add_files "$tx/src/$f.v"
}
add_files "$here/txrx_top.v"
set_property file_type SystemVerilog [get_files -filter {NAME !~ *build/ip*} *.v]
set_property include_dirs [list "$rtl/src" "$rtl/src/generated"] [current_fileset]
read_xdc "$here/txrx.xdc"
# TX ROMs are referenced relative to tx-hdl
cd $tx
synth_design -top txrx_top -part $part -mode out_of_context
write_checkpoint -force "$out/txrx_synth.dcp"
report_utilization -hierarchical -hierarchical_depth 2 -file "$out/synth_util_hier.txt"
opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force "$out/txrx_routed.dcp"
report_utilization -file "$out/impl_util.txt"
report_utilization -hierarchical -hierarchical_depth 2 -file "$out/impl_util_hier.txt"
report_timing_summary -file "$out/impl_timing.txt"
report_cdc -file "$out/cdc.txt"
report_drc -file "$out/drc.txt"
puts "=== TXRX IMPL DONE ==="
exit
