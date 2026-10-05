// SIMULATION ONLY: floating direct inverse DFT on quantized inputs.
// Verifies framing, sign, normalization and scheduler plumbing; does NOT
// reproduce vendor twiddle truncation, latency, routing or achievable timing.
`timescale 1ns/1ps
module tx_xfft_256(
 input wire aclk,aresetn,
 input wire[7:0] s_axis_config_tdata,input wire s_axis_config_tvalid,
 output wire s_axis_config_tready,
 input wire[31:0] s_axis_data_tdata,input wire s_axis_data_tvalid,
 output wire s_axis_data_tready,input wire s_axis_data_tlast,
 output reg[63:0] m_axis_data_tdata,output reg m_axis_data_tvalid,
 input wire m_axis_data_tready,output reg m_axis_data_tlast,
 output wire event_frame_started,event_tlast_unexpected,event_tlast_missing,
 output wire event_status_channel_halt,event_data_in_channel_halt,event_data_out_channel_halt
);
 integer ni,oi,k,n,phase;
 reg[31:0] samples[0:255];reg[63:0] results[0:255];
 real theta,rr,ii,vr,vi;
 integer xr,xi,yr,yi;
 reg configured;
 assign s_axis_config_tready=aresetn;
 assign s_axis_data_tready=aresetn&&configured&&phase==0;
 assign event_frame_started=0;assign event_tlast_unexpected=0;
 assign event_tlast_missing=0;assign event_status_channel_halt=0;
 assign event_data_in_channel_halt=0;assign event_data_out_channel_halt=0;
 always @(posedge aclk) begin
   if(!aresetn) begin ni=0;oi=0;phase=0;configured=0;m_axis_data_tvalid<=0;m_axis_data_tlast<=0;m_axis_data_tdata<=0;end
   else begin
     if(s_axis_config_tvalid) begin
       if(s_axis_config_tdata!=0) $fatal(1,"TX must use inverse direction");configured=1;
     end
     if(s_axis_data_tvalid&&s_axis_data_tready) begin
       if(s_axis_data_tlast!=(ni==255)) $fatal(1,"transform framing");
       samples[ni]=s_axis_data_tdata;
       if(ni==255) begin
         for(n=0;n<256;n=n+1) begin
           rr=0;ii=0;
           for(k=0;k<256;k=k+1) begin
             xr=int'($signed(samples[k][15:0]));xi=int'($signed(samples[k][31:16]));
             theta=6.283185307179586*n*k/256.0;
             rr=rr+xr*$cos(theta)-xi*$sin(theta);
             ii=ii+xr*$sin(theta)+xi*$cos(theta);
           end
           yr=rr>=0 ? $rtoi(rr+0.5) : $rtoi(rr-0.5);
           yi=ii>=0 ? $rtoi(ii+0.5) : $rtoi(ii-0.5);
           results[n]={32'(yi),32'(yr)};
         end
         ni=0;oi=0;phase=1;
       end else ni=ni+1;
     end
     if(phase==1&&(!m_axis_data_tvalid||m_axis_data_tready)) begin
       if(oi<256) begin
         m_axis_data_tdata<=results[oi];m_axis_data_tvalid<=1;m_axis_data_tlast<=oi==255;oi=oi+1;
       end else begin m_axis_data_tvalid<=0;m_axis_data_tlast<=0;phase=0;end
     end
   end
 end
endmodule
