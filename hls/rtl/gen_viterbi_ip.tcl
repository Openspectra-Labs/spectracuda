# Xilinx Viterbi Decoder v9.1, configured for spectracuda's conv_v27:
# rate 1/2, K=7, G=(0o171, 0o133), unpunctured, HARD decision.
#
# Deltas from openofdm's own viterbi_v7_0.xco, and why:
#   Coding      Hard, not Soft_Coding/3-bit -- demapper.v emits hard bits
#               and spectracuda's ConvolutionalCode is a hard-decision
#               decoder, so this is what the golden model does. Soft is
#               worth ~2 dB and is a later change on BOTH sides at once.
#   Puncturing  None, not External -- 802.11 punctures to 2/3 and 3/4,
#               spectracuda's conv_v27 does not. No erase input.
#   Best_State  off, BER_Symbol_Count off -- openofdm wires those to its
#               side-channel monitor; nothing here consumes them.
#
# Code order: Xilinx Convolution0_Code0/Code1 map to the two output bits
# in order. Python's ConvolutionalCode emits G1=0o171 first, then
# G2=0o133 (fec/viterbi.py:60), so Code0=171 and Code1=133 -- the
# OPPOSITE order from openofdm's xco, which lists code0=133 first.
# Getting this backwards decodes to garbage, so it is checked against
# Python in tb/viterbi_tb.v rather than assumed.
set part xc7a50tcsg325-1
set root [file normalize [file dirname [info script]]]

create_project -force -in_memory -part $part
set_property IP_REPO_PATHS "" [current_project]

create_ip -vlnv xilinx.com:ip:viterbi:9.1 -module_name sc_viterbi -dir "$root/build/ip"
set_property -dict [list \
    CONFIG.Architecture             {Parallel} \
    CONFIG.Constraint_Length        {7} \
    CONFIG.Convolution_Code_0_Radix {Octal} \
    CONFIG.Convolution0_Code0       {171} \
    CONFIG.Convolution0_Code1       {133} \
    CONFIG.Coding                   {Hard_Coding} \
    CONFIG.Output_Rate0             {2} \
    CONFIG.Puncturing               {None} \
    CONFIG.Traceback_Length         {42} \
    CONFIG.Best_State               {false} \
    CONFIG.BER_Symbol_Count         {false} \
] [get_ips sc_viterbi]

generate_target {instantiation_template synthesis simulation} [get_ips sc_viterbi]
synth_ip [get_ips sc_viterbi]

puts "=== sc_viterbi generated ==="
foreach p [lsort [list_property [get_ips sc_viterbi]]] {
    if {[string match CONFIG.* $p]} { puts "  $p = [get_property $p [get_ips sc_viterbi]]" }
}
exit
