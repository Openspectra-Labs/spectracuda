set root [file normalize [file join [file dirname [info script]] .. ..]]  ;# fpga/
open_checkpoint "$root/build/vivado_tx/tx_routed.dcp"
report_bus_skew -file "$root/build/vivado_tx/bus_skew.txt"
report_cdc -details -file "$root/build/vivado_tx/cdc_details.txt"
exit
