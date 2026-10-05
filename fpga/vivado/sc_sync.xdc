# 100 MHz, matching `create_clock -period 10` in the HLS synthesis script
# (hls/tcl/synth_sc_sync.tcl) so the two builds are compared at the same
# clock as well as the same part.
create_clock -name clk -period 10.000 [get_ports clk]
