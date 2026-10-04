// 100MHz T2 -> Q15 time samples. Dedicated IFFT; two frequency banks and
// THREE reserved time banks: with two, the real xfft core's latency
// (256 in + pipeline + 256 out) exceeded the 720-clock symbol budget at
// 40 MSPS/100 MHz and underran (measured on the vendor netlist). sample_ce is a radio-transport demand strobe;
// idle output is zero. Once a frame starts, demand must remain uninterrupted.
// Frame launch requires both initial time banks ready.
// Recovery: if the next symbol is not ready in time (underrun) or a symbol
// arrives marked bad (FD in_bad or a T2 sequence error here), the frame is
// cut: one zero sample with out_frame_end=1/out_active=0, then the rest of
// that frame's symbols are drained without being sent and TX returns to
// waiting for the next frame. No reset is needed; st_abort_count counts.
`timescale 1ns/1ps
module tx_time_domain #(parameter PREAMBLE_FILE="reference/preamble.mem")(
 input wire clk,rst,sample_ce,
 input wire in_valid,output wire in_ready,
 input wire signed [15:0] in_re,in_im,input wire [7:0] in_bin,in_sym_idx,
 input wire [2:0] in_stype,input wire [1:0] in_fseq,
 input wire in_frame_start,in_frame_end,in_bad,
 output reg out_valid,output reg signed [15:0] out_i,out_q,
 output reg out_active,out_frame_start,out_frame_end,
 output reg [1:0] out_fseq,
 output reg st_sequence_error,st_underrun,st_clipped,
 output reg [15:0] st_abort_count,
 output wire st_ifft_error
);
 (* ram_style="block" *) reg [31:0] freqmem[0:511];
 (* ram_style="block" *) reg [31:0] timemem[0:767];
 reg [31:0] preamble[0:255];initial $readmemh(PREAMBLE_FILE,preamble);
 reg fw,fr;
 reg [1:0] tw,tr,mw; // time banks 0..2
 reg [1:0] ffull;
 reg [2:0] treserved,tfull;
 reg [1:0] ffq[0:1],tfq[0:2];
 reg ffs[0:1],ffe[0:1],fbad[0:1],tfs[0:2],tfe[0:2],tbad[0:2];
 function [1:0] inc3(input [1:0] b); inc3 = b==2 ? 2'd0 : b+2'd1; endfunction
 reg [7:0] fsym[0:1],expect_bin;
 reg [2:0] fst[0:1];
 reg feeding,svalid;
 reg [8:0] feed_count;
 reg [31:0] sample_input;
 wire sready,mvalid,mlast;
 wire signed[24:0] mre,mim;
 reg [7:0] mindex;
 assign in_ready=!rst&&!ffull[fw];
 // Round nearest, ties away from zero, divide inverse sum by 256, then
 // Q14 -> Q15: net /128. Clamp final sample, no wraparound.
 function signed [25:0] rounded(input signed [24:0] x);
   reg signed[25:0] ext;
   begin ext={x[24],x}; rounded=ext>=0 ? ((ext+26'sd64)>>>7) : -(((-ext)+26'sd64)>>>7);end
 endfunction
 function [15:0] clamp(input signed[25:0] x);
   begin clamp=x>32767 ? 16'h7fff : (x < -32768 ? 16'h8000 : x[15:0]);end
 endfunction
 wire signed[25:0] ri=rounded(mre),rq=rounded(mim);
 wire seq_err=in_bin!=expect_bin || (in_frame_end&&in_bin!=255) ||
     (in_bin!=0&&(ffq[fw]!=in_fseq||fsym[fw]!=in_sym_idx||fst[fw]!=in_stype||in_frame_start));
 wire mready=treserved[mw]&&!tfull[mw];
 tx_ifft_engine ifft(.clk(clk),.rst(rst),.s_valid(svalid),.s_ready(sready),
   .s_re(sample_input[31:16]),.s_im(sample_input[15:0]),.s_last(feed_count==256),
   .m_valid(mvalid),.m_ready(mready),.m_re(mre),.m_im(mim),.m_last(mlast),.st_error(st_ifft_error));
 localparam WAIT_FRAME=0,PRE=1,SYMBOL=2,NEXT_SYMBOL=3,DRAIN=4;
 reg [2:0] txstate;
 reg [8:0] sample_index;
 wire [7:0] time_index=sample_index<32 ? 8'(sample_index+9'd224) : 8'(sample_index-9'd32);
 always @(posedge clk) begin
   if(rst) begin
     fw<=0;fr<=0;tw<=0;tr<=0;mw<=0;ffull<=0;treserved<=0;tfull<=0;
     feeding<=0;svalid<=0;feed_count<=0;sample_input<=0;mindex<=0;expect_bin<=0;
     txstate<=WAIT_FRAME;sample_index<=0;out_valid<=0;out_i<=0;out_q<=0;
     out_active<=0;out_frame_start<=0;out_frame_end<=0;out_fseq<=0;
     st_sequence_error<=0;st_underrun<=0;st_clipped<=0;st_abort_count<=0;
     fbad[0]<=0;fbad[1]<=0;tbad[0]<=0;tbad[1]<=0;tbad[2]<=0;
     ffq[0]<=0;ffq[1]<=0;tfq[0]<=0;tfq[1]<=0;
     ffs[0]<=0;ffs[1]<=0;ffe[0]<=0;ffe[1]<=0;
     tfs[0]<=0;tfs[1]<=0;tfs[2]<=0;tfe[0]<=0;tfe[1]<=0;tfe[2]<=0;tfq[2]<=0;
     fsym[0]<=0;fsym[1]<=0;fst[0]<=0;fst[1]<=0;
   end else begin
     out_valid<=sample_ce;out_frame_start<=0;out_frame_end<=0;
     if(in_valid&&in_ready) begin
       if(seq_err) st_sequence_error<=1;
       if(in_bin==0) begin
         ffq[fw]<=in_fseq;ffs[fw]<=in_frame_start;fsym[fw]<=in_sym_idx;fst[fw]<=in_stype;
         fbad[fw]<=seq_err||in_bad;
       end else if(seq_err||in_bad) fbad[fw]<=1;
       freqmem[{fw,in_bin}]<={in_re,in_im};expect_bin<=in_bin+1;
       if(in_bin==255) begin ffull[fw]<=1;ffe[fw]<=in_frame_end;fw<=!fw;end
     end
     if(!feeding&&ffull[fr]&&!treserved[tw]) begin
       feeding<=1;feed_count<=0;
       treserved[tw]<=1;tfq[tw]<=ffq[fr];tfs[tw]<=ffs[fr];tfe[tw]<=ffe[fr];tbad[tw]<=fbad[fr];tw<=inc3(tw);
     end
     // Registered frequency RAM read; advances only when the core accepts.
     if(feeding&&(!svalid||sready)) begin
       if(feed_count<256) begin
         sample_input<=freqmem[{fr,feed_count[7:0]}];svalid<=1;feed_count<=feed_count+1;
       end else begin
         svalid<=0;feeding<=0;ffull[fr]<=0;fr<=!fr;
       end
     end
     if(mvalid&&mready) begin
       timemem[{mw,mindex}]<={clamp(ri),clamp(rq)};
       if(ri>32767||ri < -32768||rq>32767||rq < -32768) st_clipped<=1;
       if(mlast!=(mindex==255)) st_sequence_error<=1;
       if(mindex==255) begin tfull[mw]<=1;mw<=inc3(mw);end
       mindex<=mindex+1;
     end
     // Drain: retire the cut frame's remaining symbols without sending them
     // (clock rate, not sample rate). A frame-start symbol ends the drain.
     if(txstate==DRAIN&&tfull[tr]) begin
       if(tfs[tr]) txstate<=WAIT_FRAME;
       else begin
         treserved[tr]<=0;tfull[tr]<=0;tr<=inc3(tr);
         if(tfe[tr]) txstate<=WAIT_FRAME;
       end
     end
     if(sample_ce) begin
       case(txstate)
       WAIT_FRAME: begin
         out_active<=0;out_i<=0;out_q<=0;
         if(tfull[tr]&&(!tfs[tr]||tbad[tr])) begin
           // orphan or bad frame-start symbol: drop the frame
           if(tfs[tr]) st_abort_count<=st_abort_count+1;
           treserved[tr]<=0;tfull[tr]<=0;tr<=inc3(tr);
           if(!tfe[tr]) txstate<=DRAIN;
         end else if(tfull[tr]&&tfull[inc3(tr)]&&tfs[tr]) begin
           {out_i,out_q}<=preamble[0];out_active<=1;out_frame_start<=1;
           out_fseq<=tfq[tr];sample_index<=1;txstate<=PRE;
         end
       end
       PRE: begin
         {out_i,out_q}<=preamble[sample_index[7:0]];out_active<=1;
         if(sample_index==255) begin sample_index<=0;txstate<=SYMBOL;end
         else sample_index<=sample_index+1;
       end
       SYMBOL: begin
         {out_i,out_q}<=timemem[{tr,time_index}];out_active<=1;
         if(sample_index==287) begin
           treserved[tr]<=0;tfull[tr]<=0;tr<=inc3(tr);sample_index<=0;
           if(tfe[tr]) begin txstate<=WAIT_FRAME;out_frame_end<=1;end
           else txstate<=NEXT_SYMBOL;
         end else sample_index<=sample_index+1;
       end
       NEXT_SYMBOL: begin
         if(!tfull[tr]||tfq[tr]!=out_fseq||tfs[tr]||tbad[tr]) begin
           // cut the frame: close it on the radio side, drain the rest
           if(!tfull[tr]) st_underrun<=1;
           st_abort_count<=st_abort_count+1;
           out_i<=0;out_q<=0;out_active<=0;out_frame_end<=1;txstate<=DRAIN;
         end else begin
           {out_i,out_q}<=timemem[{tr,8'd224}];out_active<=1;sample_index<=1;txstate<=SYMBOL;
         end
       end
       default: begin out_i<=0;out_q<=0;out_active<=0;end
       endcase
     end
   end
 end
endmodule
