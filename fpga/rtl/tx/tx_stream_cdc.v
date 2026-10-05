// Atomic T1 crossing. DEPTH >=4 and power of two. Reset must assert for
// BOTH domains together; release is locally synchronized. Gray pointer
// crossings require source-period max-delay/bus-skew constraints in hardware.
// The read side registers the RAM output (r_data is a flop in rclk), so the
// only clk_bit -> clk_sample data endpoints are those WIDTH flops; nothing
// downstream sees the RAM's asynchronous read path.
`timescale 1ns/1ps
module tx_stream_cdc #(parameter WIDTH=32, DEPTH=32)(
 input wire wclk,rclk,arst,
 input wire w_valid, output wire w_ready,input wire [WIDTH-1:0] w_data,
 output wire r_valid,input wire r_ready,output wire [WIDTH-1:0] r_data
);
 localparam AW=$clog2(DEPTH);
 initial if(DEPTH<4 || (DEPTH&(DEPTH-1))!=0) $fatal(1,"CDC depth must be power of two >=4");
 (* ASYNC_REG="TRUE" *) reg [1:0] wr_reset,rd_reset;
 always @(posedge wclk or posedge arst)
   if(arst) wr_reset<=3; else wr_reset<={wr_reset[0],1'b0};
 always @(posedge rclk or posedge arst)
   if(arst) rd_reset<=3; else rd_reset<={rd_reset[0],1'b0};
 wire wrst=wr_reset[1],rrst=rd_reset[1];
 (* ram_style="distributed" *) reg [WIDTH-1:0] mem[0:DEPTH-1];
 reg [AW:0] wb,wg,rb,rg;
 (* ASYNC_REG="TRUE" *) reg [AW:0] rg1,rg2,wg1,wg2;
 wire full=wg=={~rg2[AW:AW-1],rg2[AW-2:0]};
 assign w_ready=!wrst&&!full;
 reg ov;reg [WIDTH-1:0] od;
 wire pop=!rrst&&(rg!=wg2)&&(!ov||r_ready);
 assign r_valid=ov;
 assign r_data=od;
 wire [AW:0] wn=wb+1,rn=rb+1;
 always @(posedge wclk) begin
   if(wrst) begin wb<=0;wg<=0;rg1<=0;rg2<=0;end
   else begin
     rg1<=rg;rg2<=rg1;
     if(w_valid&&w_ready) begin mem[wb[AW-1:0]]<=w_data;wb<=wn;wg<=wn^(wn>>1);end
   end
 end
 always @(posedge rclk) begin
   if(rrst) begin rb<=0;rg<=0;wg1<=0;wg2<=0;ov<=0;end
   else begin
     wg1<=wg;wg2<=wg1;
     if(pop) begin od<=mem[rb[AW-1:0]];ov<=1;rb<=rn;rg<=rn^(rn>>1);end
     else if(r_ready) ov<=0;
   end
 end
endmodule
