`timescale 1ns/1ps
module tx_top_tb;
 reg clkbit=0,clk=0;always #4 clkbit=~clkbit;always #5 clk=~clk;
 reg arst=1,ce=0,cv=0,iv=0,last=0;
 reg[15:0] len=0;reg[2:0] modcode=0;reg[47:0] userword=0;reg[1:0] frame=0,dmrscode=0;
 reg[7:0] byte_in=0;
 wire cr,ir,valid,active,fs,fe,bdone,be,ferr,seqerr,under,clipped,ifterr;
 wire signed[15:0] re,im;wire[1:0] fq;wire[15:0] aborts;
 // +abort_frame=K: demand one sample per clock (100 MSPS) during frame K
 // to force an underrun; +bad_frame=K: mark symbol 3 of frame K bad at the
 // FD->TD boundary. Either must cut exactly that frame and recover: every
 // other frame bit-exact, no reset.
 // No +expected (loopback mode): nothing compared here, every radiated
 // sample goes to +dump=FILE as "i q" and the run ends when all frames end.
 integer has_exp=1,dumpfd=0;
 integer abort_frame=-1,bad_frame=-1,fastce=0,started=0,aborts_seen=0,fdstarts=0,forcing=0;
 integer cycle=0,fd,ef,rc,a,b,c,d,nb,i,count=0,total=0,frames=0,ended=0;
 reg[1023:0] input_file,expected_file;
 reg signed[15:0] ei[0:150000],eq[0:150000];reg[1:0] eframe[0:150000];
 reg efs[0:150000],efe[0:150000];
 tx_top dut(.clk_bit(clkbit),.clk_sample(clk),.arst(arst),.sample_ce(ce),
   .cfg_valid(cv),.cfg_ready(cr),.cfg_payload_bits(len),.cfg_mod(modcode),.cfg_dmrs(dmrscode),
   .cfg_user(userword),.cfg_fseq(frame),.in_valid(iv),.in_ready(ir),.in_byte(byte_in),.in_last(last),
   .out_valid(valid),.out_i(re),.out_q(im),.out_active(active),.out_frame_start(fs),.out_frame_end(fe),
   .out_fseq(fq),.bit_done(bdone),.st_bit_error(be),.st_freq_error(ferr),
   .st_sequence_error(seqerr),.st_underrun(under),.st_clipped(clipped),.st_abort_count(aborts),.st_ifft_error(ifterr));
 always @(negedge clk) begin
   cycle=cycle+1;ce=!arst&&(fastce!=0||cycle%5==0||cycle%5==2); // 40MSPS demand
   if(cycle>1500000) $fatal(1,"timeout samples=%0d/%0d frames=%0d ended=%0d BIT=%0d FDfull=%b TDfull=%b reserved=%b feeding=%b svalid=%b",count,total,frames,ended,dut.bit_domain.estate,dut.freq_domain.full,dut.time_domain.tfull,dut.time_domain.treserved,dut.time_domain.feeding,dut.time_domain.svalid);
 end
 always @(negedge clk) if(!arst) begin
   if(be||ferr||seqerr||(under&&abort_frame<0)||ifterr) $fatal(1,"status be=%b fd=%b seq=%b underrun=%b ifft=%b",be,ferr,seqerr,under,ifterr);
   if(valid&&fe&&!active) begin
     // frame cut: skip the rest of its expected samples
     aborts_seen=aborts_seen+1;ended=ended+1;fastce=0;
     while(count<total&&!efs[count]) count=count+1;
   end
   if(valid&&active&&fs) begin started=started+1;if(started==abort_frame+1) fastce=1;end
   if(dut.freq_domain.out_valid&&dut.fr&&dut.freq_domain.out_frame_start) fdstarts=fdstarts+1;
   if(forcing==0&&fdstarts==bad_frame+1&&dut.freq_domain.out_valid&&dut.freq_domain.out_sym_idx==3) begin
     force dut.fbad=1'b1;forcing=1;
   end else if(forcing==1&&dut.freq_domain.out_sym_idx!=3) begin release dut.fbad;forcing=2;end
   if(valid&&active&&dumpfd!=0) $fwrite(dumpfd,"%0d %0d\n",re,im);
   if(valid&&active&&has_exp==0) begin count=count+1;if(fe) ended=ended+1;end
   else if(valid&&active) begin
     if(count>=total) $fatal(1,"extra sample");
     // One final-Q15 LSB tolerance for floating DFT oracle rounding.
     if(int'(re)-int'(ei[count])>1||int'(re)-int'(ei[count]) < -1||
        int'(im)-int'(eq[count])>1||int'(im)-int'(eq[count]) < -1||
        fq!=eframe[count]||fs!=efs[count]||fe!=efe[count])
       $fatal(1,"sample %0d got %0d,%0d want %0d,%0d fq=%0d fs=%b fe=%b",count,re,im,ei[count],eq[count],fq,fs,fe);
     count=count+1;if(fe) ended=ended+1;
   end
 end
 initial begin
   void'($value$plusargs("abort_frame=%d",abort_frame));void'($value$plusargs("bad_frame=%d",bad_frame));
   if(!$value$plusargs("input=%s",input_file)) $fatal(1,"fixtures");
   if($value$plusargs("dump=%s",expected_file)) dumpfd=$fopen(expected_file,"w");
   if(!$value$plusargs("expected=%s",expected_file)) has_exp=0;
   if(has_exp!=0) ef=$fopen(expected_file,"r");
   while(has_exp!=0&&!$feof(ef)) begin
     rc=$fscanf(ef,"%d %d %d %d %d\n",a,b,c,d,i);
     if(rc==5) begin ei[total]=16'(a);eq[total]=16'(b);eframe[total]=2'(c);efs[total]=d[0];efe[total]=i[0];total=total+1;end
   end
   if(has_exp!=0) $fclose(ef);
   fd=$fopen(input_file,"r");
   repeat(12) @(negedge clkbit);arst=0;
   repeat(5) @(negedge clkbit);
   while(!$feof(fd)) begin
     rc=$fscanf(fd,"%d %d %h %d %d %d\n",a,b,userword,c,nb,d);
     if(rc==6) begin
       @(negedge clkbit);
       while(!cr) @(negedge clkbit);
       cv=1;len=16'(a);modcode=3'(b);frame=2'(c);dmrscode=2'(d);@(negedge clkbit);cv=0;
       for(i=0;i<nb;i=i+1) begin
         rc=$fscanf(fd,"%d\n",a);if(rc!=1) $fatal(1,"payload");
         while(!ir) @(negedge clkbit);
         iv=1;byte_in=8'(a);last=i==nb-1;@(negedge clkbit);
         // Capture is committed before launch; deliberately intermittent host.
         iv=0;repeat(i%3) @(negedge clkbit);
       end
       iv=0;last=0;frames=frames+1;
       // At most two outstanding frames; do not reuse unretired fseq.
       if(frames%2==0) wait(ended==frames);
     end
   end
   $fclose(fd);
   if(has_exp!=0) wait(count==total); else wait(ended==frames);
   if(dumpfd!=0) $fclose(dumpfd);repeat(100) @(negedge clk);
   if(int'(aborts)!=aborts_seen||aborts_seen!=(abort_frame>=0 ? 1:0)+(bad_frame>=0 ? 1:0)||(abort_frame>=0&&!under))
     $fatal(1,"abort accounting: count=%0d seen=%0d underrun=%b",aborts,aborts_seen,under);
   $display("PASS frames=%0d samples=%0d cycles=%0d clipped=%b aborts=%0d",frames,count,cycle,clipped,aborts);$finish;
 end
endmodule
