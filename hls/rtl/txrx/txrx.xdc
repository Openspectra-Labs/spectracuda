# Complete modem: two clocks, asynchronous to each other.
create_clock -name clk_sample -period 10.000 [get_ports clk_sample]
create_clock -name clk_bit    -period 8.000  [get_ports clk_bit]
# Every crossing is a Gray-pointer FIFO, a req/ack bundle or a reset
# synchronizer. Bound each to one DESTINATION period (datapath only) rather
# than cutting it with clock groups, which would leave them unconstrained.
set_max_delay -datapath_only 8.000  -from [get_clocks clk_sample] -to [get_clocks clk_bit]
set_max_delay -datapath_only 10.000 -from [get_clocks clk_bit]    -to [get_clocks clk_sample]
# TX coded-group FIFO Gray pointers: bounded skew between bits
set_bus_skew 8.000  -from [get_cells -hier -regexp {.*crossing/wg_reg\[[0-9]+\]}] -to [get_cells -hier -regexp {.*crossing/wg1_reg\[[0-9]+\]}]
set_bus_skew 10.000 -from [get_cells -hier -regexp {.*crossing/rg_reg\[[0-9]+\]}] -to [get_cells -hier -regexp {.*crossing/rg1_reg\[[0-9]+\]}]
set_false_path -from [get_ports arst]
