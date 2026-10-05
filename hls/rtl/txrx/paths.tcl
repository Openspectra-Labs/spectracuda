open_checkpoint build/txrx_routed.dcp
foreach clk {clk_bit clk_sample} {
  foreach p [get_timing_paths -group $clk -max_paths 600 -nworst 1 -unique_pins] {
    set s [get_property STARTPOINT_PIN $p]; set e [get_property ENDPOINT_PIN $p]
    puts "PATH $clk [format %.3f [get_property SLACK $p]] lv=[get_property LOGIC_LEVELS $p] [regsub -all {\[[0-9]+\]} [get_property PARENT_CELL [get_cells -of $s]] {}] -> [regsub -all {\[[0-9]+\]} $e {}]"
  }
}
exit
