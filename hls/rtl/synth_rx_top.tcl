# Out-of-context synthesis of the whole receiver.
#
# Like synth_cp_fft.tcl this reads the FFT IP in and generates its
# synthesis target first -- adding only cp_fft.v would leave xfft_256 a
# black box and undercount the design by most of the FFT's cost.
#
# tb/stubs/xfft_256.v is NEVER added here. It is a behavioural DFT for
# Verilator only; synthesizing it would be meaningless.
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
           mmse_eq pilot_cpe header_decode demapper viterbi_dec viterbi_dec_ovl viterbi_dec_soft cdc_async_fifo cdc_bundle cdc_reset_sync viterbi_dec_ovl
           deinterleaver} {
    add_files "$root/src/$f.v"
}
set_property include_dirs [list "$root/src" "$root/src/generated"] [current_fileset]
# The five newest blocks use SystemVerilog sized casts to keep
# width-safe comparisons readable. Vivado parses .v as Verilog-2001 by
# default and rejects that syntax, so they are marked explicitly.
# Verilator accepted them without this, which is why it only surfaced
# at synthesis.
foreach f [get_files -quiet {*grid_extract.v *header_decode.v *frame_sync.v *pilot_cpe.v *rx_top.v *rx_time_domain.v *rx_freq_domain.v *sync_fifo_fwft.v *rx_bit_domain.v}] {
    set_property file_type SystemVerilog $f
}

read_xdc "$root/sc_sync.xdc"
read_xdc "$root/rx_top_cdc.xdc"
set_property top rx_top [current_fileset]

synth_design -top rx_top -part $part -mode out_of_context
report_utilization -file "$root/build/rx_top_util.txt"
report_utilization -hierarchical -file "$root/build/rx_top_util_hier.txt"
report_timing_summary -file "$root/build/rx_top_timing.txt"
puts "=== RX_TOP SYNTH DONE ==="
exit
