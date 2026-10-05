`timescale 1ns/1ps
// Full standalone TX wiring. Common asynchronous reset; synchronous local
// releases. cfg/data are in clk_bit; sample_ce and IQ are in clk_sample.
module tx_top(
 input wire clk_bit,clk_sample,arst,sample_ce,
 input wire cfg_valid,output wire cfg_ready,input wire[15:0] cfg_payload_bits,
 input wire[2:0] cfg_mod,input wire[63:0] cfg_user,input wire[1:0] cfg_fseq,
 input wire in_valid,output wire in_ready,input wire[7:0] in_byte,input wire in_last,
 output wire out_valid,output wire signed[15:0] out_i,out_q,
 output wire out_active,out_frame_start,out_frame_end,output wire[1:0] out_fseq,
 output wire bit_done,
 // st_bit_error is clk_bit; all other status outputs are clk_sample.
 output wire st_bit_error,st_freq_error,st_sequence_error,st_underrun,st_clipped,st_ifft_error,
 // frames cut by an underrun or a bad symbol (TX recovers by itself)
 output wire [15:0] st_abort_count
);
 (* ASYNC_REG="TRUE" *) reg[1:0] br,sr;
 always @(posedge clk_bit or posedge arst) if(arst) br<=3;else br<={br[0],1'b0};
 always @(posedge clk_sample or posedge arst) if(arst) sr<=3;else sr<={sr[0],1'b0};
 wire bv,ready,cvalid,cready;wire[31:0] bitem,citem;
 tx_bit_domain bit_domain(.clk_bit(clk_bit),.rst(br[1]),
   .cfg_valid(cfg_valid),.cfg_ready(cfg_ready),.cfg_payload_bits(cfg_payload_bits),
   .cfg_mod(cfg_mod),.cfg_user(cfg_user),.cfg_fseq(cfg_fseq),.in_valid(in_valid),
   .in_ready(in_ready),.in_byte(in_byte),.in_last(in_last),
   .out_valid(bv),.out_ready(ready),.out_bits(bitem[5:0]),.out_n(bitem[8:6]),
   .out_sc(bitem[16:9]),.out_sym_idx(bitem[24:17]),.out_stype(bitem[27:25]),
   .out_fseq(bitem[29:28]),.out_frame_start(bitem[30]),.out_frame_end(bitem[31]),
   .frame_done(bit_done),.st_error(st_bit_error));
 tx_stream_cdc #(.WIDTH(32)) crossing(.wclk(clk_bit),.rclk(clk_sample),.arst(arst),
   .w_valid(bv),.w_ready(ready),.w_data(bitem),.r_valid(cvalid),.r_ready(cready),.r_data(citem));
 wire fv,fr;wire signed[15:0] re,im;wire[7:0] bin,sym;wire[2:0] st;
 wire[1:0] fq;wire fs,fe,fbad;
 tx_freq_domain freq_domain(.clk(clk_sample),.rst(sr[1]),.in_valid(cvalid),.in_ready(cready),
   .in_bits(citem[5:0]),.in_n(citem[8:6]),.in_sc(citem[16:9]),.in_sym_idx(citem[24:17]),
   .in_stype(citem[27:25]),.in_fseq(citem[29:28]),.in_frame_start(citem[30]),.in_frame_end(citem[31]),
   .out_valid(fv),.out_ready(fr),.out_re(re),.out_im(im),.out_bin(bin),.out_sym_idx(sym),
   .out_stype(st),.out_fseq(fq),.out_frame_start(fs),.out_frame_end(fe),.out_bad(fbad),.st_error(st_freq_error));
 tx_time_domain time_domain(.clk(clk_sample),.rst(sr[1]),.sample_ce(sample_ce),
   .in_valid(fv),.in_ready(fr),.in_re(re),.in_im(im),.in_bin(bin),.in_sym_idx(sym),
   .in_stype(st),.in_fseq(fq),.in_frame_start(fs),.in_frame_end(fe),.in_bad(fbad),
   .out_valid(out_valid),.out_i(out_i),.out_q(out_q),.out_active(out_active),
   .out_frame_start(out_frame_start),.out_frame_end(out_frame_end),.out_fseq(out_fseq),
   .st_sequence_error(st_sequence_error),.st_underrun(st_underrun),.st_clipped(st_clipped),.st_abort_count(st_abort_count),.st_ifft_error(st_ifft_error));
endmodule
