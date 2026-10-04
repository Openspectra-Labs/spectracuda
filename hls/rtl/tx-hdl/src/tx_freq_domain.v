// T1 -> T2. clk=100MHz. Two independent symbol banks overlap capture
// and grid output. All constellations/grid sequences come from pinned Python.
// Frequency components are signed 16-bit, with 14 fractional bits.
`timescale 1ns/1ps
module tx_freq_domain #(
 parameter MAP_FILE="reference/mapper.mem",
 parameter GRID_FILE="reference/grid_type.mem",
 parameter BIN_FILE="reference/data_bins.mem",
 parameter TRAIN_FILE="reference/training.mem"
)(
 input wire clk,rst,
 input wire in_valid,output wire in_ready,
 input wire [5:0] in_bits,input wire [2:0] in_n,
 input wire [7:0] in_sc,in_sym_idx,input wire [2:0] in_stype,
 input wire [1:0] in_fseq,input wire in_frame_start,in_frame_end,
 output reg out_valid,input wire out_ready,
 output reg signed [15:0] out_re,out_im,
 output reg [7:0] out_bin,out_sym_idx,output reg [2:0] out_stype,
 output reg [1:0] out_fseq,output reg out_frame_start,out_frame_end,
 // out_bad: this symbol's T1 input broke the protocol (wrong sc order,
 // metadata change mid-symbol, illegal n/stype). Held for all 256 bins so
 // TD drops the whole frame instead of radiating it.
 output reg out_bad,
 output reg st_error
);
 (* rom_style="distributed" *) reg [31:0] mapper[0:255];
 (* rom_style="distributed" *) reg [1:0] kind[0:255];
 (* rom_style="distributed" *) reg [7:0] data_bin[0:215];
 reg [31:0] training[0:255];
 (* ram_style="block" *) reg [31:0] mapped[0:511];
 initial begin
   $readmemh(MAP_FILE,mapper);$readmemh(GRID_FILE,kind);
   $readmemh(BIN_FILE,data_bin);$readmemh(TRAIN_FILE,training);
 end
 reg wrbank,rdbank;
 reg [1:0] full;
 reg [7:0] expect_sc;
 reg [7:0] sym[0:1];reg [2:0] stype[0:1];reg [1:0] fq[0:1];
 reg fs[0:1],fe[0:1],bad[0:1];
 reg [2:0] nbits[0:1];
 reg emitting;reg [7:0] next_bin;
 wire [1:0] map_group=in_n==1?0:(in_n==2?1:(in_n==4?2:3));
 wire legal_n=in_n==1||in_n==2||in_n==4||in_n==6;
 assign in_ready=!rst&&!full[wrbank];
 wire accept=in_valid&&in_ready;
 wire item_err=in_sc!=expect_sc || (in_stype!=0&&in_stype!=1&&in_stype!=3) ||
          (in_stype==0 ? (in_n!=0||in_sc!=0) : !legal_n) ||
          (in_stype==1&&in_n!=1) ||
          (in_frame_end&&(in_stype!=3||in_sc!=215));
 wire meta_err=expect_sc!=0 && (sym[wrbank]!=in_sym_idx || stype[wrbank]!=in_stype ||
          fq[wrbank]!=in_fseq || nbits[wrbank]!=in_n || in_frame_start);
 wire load=!out_valid||out_ready;
 wire [31:0] bin_value=stype[rdbank]==0 ? training[next_bin] :
     (kind[next_bin]==0 ? 32'd0 :
      (kind[next_bin]==2 ? 32'h40000000 : mapped[{rdbank,next_bin}]));
 always @(posedge clk) begin
   if(rst) begin
     wrbank<=0;rdbank<=0;full<=0;expect_sc<=0;
     emitting<=0;next_bin<=0;out_valid<=0;st_error<=0;
     out_re<=0;out_im<=0;out_bin<=0;out_sym_idx<=0;out_stype<=0;
     out_fseq<=0;out_frame_start<=0;out_frame_end<=0;out_bad<=0;bad[0]<=0;bad[1]<=0;
     sym[0]<=0;sym[1]<=0;stype[0]<=0;stype[1]<=0;
     fq[0]<=0;fq[1]<=0;fs[0]<=0;fs[1]<=0;fe[0]<=0;fe[1]<=0;
     nbits[0]<=0;nbits[1]<=0;
   end else begin
     if(out_valid&&out_ready) out_valid<=0;
     if(accept) begin
       if(item_err||meta_err) st_error<=1;
       if(expect_sc==0) begin
         sym[wrbank]<=in_sym_idx;stype[wrbank]<=in_stype;
         fq[wrbank]<=in_fseq;fs[wrbank]<=in_frame_start;
         nbits[wrbank]<=in_n;bad[wrbank]<=item_err;
       end else if(item_err||meta_err) bad[wrbank]<=1;
       if(in_stype!=0&&in_sc<=215)
         mapped[{wrbank,data_bin[in_sc]}]<=mapper[{map_group,in_bits}];
       if(in_stype==0||in_sc==215) begin
         full[wrbank]<=1;fe[wrbank]<=in_frame_end;wrbank<=!wrbank;expect_sc<=0;
       end else expect_sc<=expect_sc+1;
     end
     if(!emitting&&full[rdbank]) begin emitting<=1;next_bin<=0;end
     if(emitting&&load) begin
       out_valid<=1;{out_re,out_im}<=bin_value;out_bin<=next_bin;
       out_sym_idx<=sym[rdbank];out_stype<=stype[rdbank];out_fseq<=fq[rdbank];
       out_frame_start<=fs[rdbank]&&next_bin==0;
       out_frame_end<=fe[rdbank]&&next_bin==255;out_bad<=bad[rdbank];
       if(next_bin==255) begin
         emitting<=0;full[rdbank]<=0;rdbank<=!rdbank;
       end else next_bin<=next_bin+1;
     end
   end
 end
endmodule
