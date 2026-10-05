create_clock -name clk_bit -period 8.000 [get_ports clk_bit]
create_clock -name clk_sample -period 10.000 [get_ports clk_sample]
# Explicit Gray constraints, rather than blanket asynchronous clock groups
# which would override datapath-only max delays. Patterns are hierarchy-
# independent (.*crossing/...) so they still match when tx_top is embedded.
set_max_delay -datapath_only 8.000 -from [get_cells -hier -regexp {.*crossing/wg_reg\[[0-9]+\]}] -to [get_cells -hier -regexp {.*crossing/wg1_reg\[[0-9]+\]}]
set_bus_skew 8.000 -from [get_cells -hier -regexp {.*crossing/wg_reg\[[0-9]+\]}] -to [get_cells -hier -regexp {.*crossing/wg1_reg\[[0-9]+\]}]
set_max_delay -datapath_only 10.000 -from [get_cells -hier -regexp {.*crossing/rg_reg\[[0-9]+\]}] -to [get_cells -hier -regexp {.*crossing/rg1_reg\[[0-9]+\]}]
set_bus_skew 10.000 -from [get_cells -hier -regexp {.*crossing/rg_reg\[[0-9]+\]}] -to [get_cells -hier -regexp {.*crossing/rg1_reg\[[0-9]+\]}]
set_false_path -from [get_ports arst]
