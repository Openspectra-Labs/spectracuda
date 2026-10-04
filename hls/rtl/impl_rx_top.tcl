# Out-of-context PLACE & ROUTE of the whole receiver.
#
# Self-contained (re-runs synthesis) rather than reading a checkpoint,
# so the implemented result cannot silently come from a stale synth.
#
# out_of_context means no I/O buffers are inserted: this proves the
# design FITS the fabric and CLOSES TIMING at 100 MHz, which is what we
# need before integrating it. It is NOT a bitstream -- rx_top has no
# pins. The bitstream is built in the Spectra M.2 project once this is
# packaged as IP and wired to the AD9361 and PCIe/USB blocks.
set part xc7a50tcsg325-1
set root [file normalize [file dirname [info script]]]

create_project -in_memory -part $part
set_property target_language Verilog [current_project]

read_ip "$root/build/ip/xfft_256/xfft_256.xci"
generate_target synthesis [get_ips xfft_256]
synth_ip [get_ips xfft_256]

foreach f {rx_top rx_time_domain rx_freq_domain sync_fifo_fwft rx_bit_domain
           sc_sync_rtl frame_sync cfo_estimate cfo_correct
           cordic_rot cordic_vec cp_fft grid_extract ls_chanest
           mmse_eq pilot_cpe header_decode demapper viterbi_dec viterbi_dec_ovl viterbi_dec_ovl
           deinterleaver} {
    add_files "$root/src/$f.v"
}
set_property include_dirs [list "$root/src" "$root/src/generated"] [current_fileset]

foreach f [get_files -quiet {*grid_extract.v *header_decode.v *frame_sync.v *pilot_cpe.v *rx_top.v *rx_time_domain.v *rx_freq_domain.v *sync_fifo_fwft.v *rx_bit_domain.v}] {
    set_property file_type SystemVerilog $f
}

read_xdc "$root/sc_sync.xdc"
set_property top rx_top [current_fileset]

synth_design -top rx_top -part $part -mode out_of_context
write_checkpoint -force "$root/build/rx_top_synth.dcp"

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force "$root/build/rx_top_routed.dcp"
report_utilization           -file "$root/build/rx_top_impl_util.txt"
report_utilization -hierarchical -file "$root/build/rx_top_impl_util_hier.txt"
report_timing_summary        -file "$root/build/rx_top_impl_timing.txt"
report_drc                   -file "$root/build/rx_top_impl_drc.txt"
puts "=== RX_TOP IMPL DONE ==="
exit
