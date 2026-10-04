# Out-of-context synthesis of one module, same part and clock as every
# other measurement in this project so the numbers are comparable.
#
#   vivado -mode batch -source synth_any.tcl -tclargs <top> <src.v> [more.v ...]
#
# The clock comes from an XDC via read_xdc: create_clock before
# synth_design fails with "No open design", and constraints have to be
# present DURING synthesis to influence it rather than applied after.
set part xc7a50tcsg325-1
set root [file normalize [file dirname [info script]]]
set top  [lindex $argv 0]
set srcs [lrange $argv 1 end]

create_project -force -in_memory -part $part
foreach f $srcs { add_files "$root/$f" }
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
set_property top $top [current_fileset]

synth_design -top $top -part $part -mode out_of_context
report_utilization -file "$root/build/${top}_util.txt"
report_utilization -hierarchical -file "$root/build/${top}_util_hier.txt"
report_timing_summary -file "$root/build/${top}_timing.txt"
puts "=== done $top ==="
exit
