// Private vendor-core wrapper. Generate tx_xfft_256 using ip/create_ifft.tcl.
// Input Q14, unscaled inverse result retains Q14 and 25 significant bits.
`timescale 1ns/1ps
module tx_ifft_engine(
 input wire clk,rst,
 input wire s_valid,output wire s_ready,
 input wire signed [15:0] s_re,s_im,input wire s_last,
 output wire m_valid, input wire m_ready,
 output wire signed [24:0] m_re,m_im,output wire m_last,
 output reg st_error
);
 reg configured;
 wire config_ready;
 wire [63:0] data;
 wire unexpected,missing,halt;
 always @(posedge clk) begin
   if(rst) begin configured<=0;st_error<=0;end
   else begin
     if(!configured&&config_ready) configured<=1;
     if(unexpected||missing||halt) st_error<=1;
   end
 end
 wire core_ready;
 assign s_ready=configured&&core_ready&&!rst;
 assign m_re=data[24:0];assign m_im=data[56:32];
 tx_xfft_256 core(
   .aclk(clk),.aresetn(!rst),
   .s_axis_config_tdata(8'h00),.s_axis_config_tvalid(!configured&&!rst),
   .s_axis_config_tready(config_ready),
   .s_axis_data_tdata({s_im,s_re}),.s_axis_data_tvalid(s_valid&&configured&&!rst),
   .s_axis_data_tready(core_ready),.s_axis_data_tlast(s_last),
   .m_axis_data_tdata(data),.m_axis_data_tvalid(m_valid),
   .m_axis_data_tready(m_ready),.m_axis_data_tlast(m_last),
   .event_frame_started(),.event_tlast_unexpected(unexpected),
   .event_tlast_missing(missing),.event_status_channel_halt(),
   .event_data_in_channel_halt(),.event_data_out_channel_halt(halt)
 );
endmodule
