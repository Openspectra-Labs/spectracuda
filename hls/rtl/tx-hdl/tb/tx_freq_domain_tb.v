`timescale 1ns/1ps
module tx_freq_domain_tb;
 reg bitclk=0,clk=0;always #4 bitclk=~bitclk;always #5 clk=~clk;
 reg rst=1,iv=0;reg [31:0] input_item=0;
 wire ir,cv,cr;wire [31:0] crossed;
 tx_stream_cdc #(.WIDTH(32)) cdc(.wclk(bitclk),.rclk(clk),.arst(rst),.w_valid(iv),.w_ready(ir),
   .w_data(input_item),.r_valid(cv),.r_ready(cr),.r_data(crossed));
 wire v,fs,fe,error,bad;wire signed[15:0] re,im;
 wire [7:0] bin,sym;wire[2:0] st;wire[1:0] fq;
 // +corrupt=K: flip sc bit0 of the K-th DATA item (sc 1..200); FD must flag
 // exactly that symbol: out_bad on all its 256 bins, every other bin exact.
 integer corrupt=-1,ndata=0,badbins=0,corrupted=0;
 integer cyc=0,stall=0,fd,rc,a,b,c,d,e,f,g,h,count=0,total=0;
 wire ready=stall==0||(cyc%17<9);
 tx_freq_domain dut(.clk(clk),.rst(rst),.in_valid(cv),.in_ready(cr),
   .in_bits(crossed[5:0]),.in_n(crossed[8:6]),.in_sc(crossed[16:9]),
   .in_sym_idx(crossed[24:17]),.in_stype(crossed[27:25]),.in_fseq(crossed[29:28]),
   .in_frame_start(crossed[30]),.in_frame_end(crossed[31]),
   .out_valid(v),.out_ready(ready),.out_re(re),.out_im(im),.out_bin(bin),
   .out_sym_idx(sym),.out_stype(st),.out_fseq(fq),.out_frame_start(fs),
   .out_frame_end(fe),.out_bad(bad),.st_error(error));
 reg[1023:0] file_in,file_out;
 reg[54:0] expected[0:100000];
 wire[54:0] result={fe,fs,fq,st,sym,bin,re,im};
 reg held=0;reg[54:0] old_result;
 always @(posedge clk) begin
   cyc<=cyc+1;
   if(!rst) begin
     if(held&&(!v||result!==old_result)) $fatal(1,"unstable grid output");
     held<=v&&!ready;old_result<=result;
     if(error&&corrupt<0) $fatal(1,"FD sequence error");
     if(v&&ready&&bad) begin badbins=badbins+1;count<=count+1;end
     else if(v&&ready) begin
       if(count>=total||result!==expected[count]) $fatal(1,"bin %0d got %h want %h",count,result,expected[count]);
       count<=count+1;
     end
     if(cyc>2000000) $fatal(1,"timeout");
   end
 end
 initial begin
   if(!$value$plusargs("input=%s",file_in)||!$value$plusargs("output=%s",file_out)) $fatal(1,"files required");
   void'($value$plusargs("stall=%d",stall));void'($value$plusargs("corrupt=%d",corrupt));
   fd=$fopen(file_out,"r");
   while(!$feof(fd)) begin
     rc=$fscanf(fd,"%d %d %d %d %d %d %d %d\n",a,b,c,d,e,f,g,h);
     if(rc==8) begin expected[total]={h[0],g[0],a[1:0],b[2:0],c[7:0],d[7:0],e[15:0],f[15:0]};total=total+1;end
   end
   $fclose(fd);fd=$fopen(file_in,"r");
   repeat(8) @(negedge bitclk);rst=0;
   repeat(4) @(negedge bitclk);
   while(!$feof(fd)) begin
     rc=$fscanf(fd,"%d %d %d %d %d %d %d %d\n",a,b,c,d,e,f,g,h);
     if(rc==8) begin
       if(b==3&&d>=1&&d<=200) begin
         if(ndata==corrupt) begin d=d^1;corrupted=1;end
         ndata=ndata+1;
       end
       iv=1;input_item={h[0],g[0],a[1:0],b[2:0],c[7:0],d[7:0],e[2:0],f[5:0]};
       while(!ir) @(negedge bitclk);
       @(negedge bitclk);
     end
   end
   iv=0;$fclose(fd);wait(count==total);repeat(100) @(negedge clk);
   if(v||cv) $fatal(1,"extra bins");
   if(corrupt<0 ? badbins!=0 : (badbins!=256||!error||corrupted==0)) $fatal(1,"bad-symbol check: badbins=%0d error=%b",badbins,error);
   $display("PASS bins=%0d cycles=%0d badbins=%0d",count,cyc,badbins);$finish;
 end
endmodule
