# Bit domain on its own clock (rx_top.clk_bd), asynchronous to clk (TD/FD).
# Read after sc_sync.xdc, which defines clk at 100 MHz.
create_clock -name clk_bd -period 8.000 [get_ports clk_bd]
set_clock_groups -asynchronous -group [get_clocks clk] -group [get_clocks clk_bd]
# Gray pointers and the C1 req/ack: bound the skew so a synchronizer never
# sees a pointer more than one step old (max-delay = one destination period).
set_max_delay -datapath_only 8.000 -from [get_clocks clk]    -to [get_clocks clk_bd]
set_max_delay -datapath_only 10.000 -from [get_clocks clk_bd] -to [get_clocks clk]
