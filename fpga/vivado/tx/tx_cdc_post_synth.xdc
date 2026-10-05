# Apply AFTER synthesis (impl only): the FIFO memory primitives exist only
# once mapped. FIFO data is published through the synchronized Gray write
# pointer (2 clk_sample flops), so the write-port -> read-side path must be
# bounded by one clk_sample period. A max-delay (NOT a false path) keeps it
# timed. Fails loudly instead of silently matching nothing.
set tx_fifo_memory [get_cells -hier -regexp -filter {IS_PRIMITIVE} {.*crossing/mem_reg.*}]
if {![llength $tx_fifo_memory]} { error "tx_cdc_post_synth.xdc: no crossing/mem_reg primitives found" }
set_max_delay -datapath_only 10.000 -from $tx_fifo_memory -to [get_clocks clk_sample]
