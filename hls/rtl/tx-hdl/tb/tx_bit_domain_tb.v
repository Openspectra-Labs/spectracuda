`timescale 1ns/1ps
module tx_bit_domain_tb;
    reg clk=0; always #4 clk=~clk; // 125 MHz
    reg rst=1, cfg_valid=0, in_valid=0, in_last=0;
    reg [15:0] length_bits=0;
    reg [2:0] mod_code=0;
    reg [63:0] user_data=0;
    reg [1:0] frame=0;
    reg [7:0] byte_in=0;
    wire cfg_ready,in_ready,valid,fs,fe,done,error;
    wire [5:0] bits_out; wire [2:0] n,stype;
    wire [7:0] sc,sym; wire [1:0] fq;
    integer cycle=0, count=0, expected_count=0, fd,rc;
    integer a,b,c,d,e,f,g,h,bytes,i,stall=0;
    reg [1023:0] name;
    wire ready = stall==0 || (cycle%11!=3 && cycle%11!=4 && cycle%11!=5);
    tx_bit_domain dut(.clk_bit(clk),.rst(rst),.cfg_valid(cfg_valid),.cfg_ready(cfg_ready),
      .cfg_payload_bits(length_bits),.cfg_mod(mod_code),.cfg_user(user_data),.cfg_fseq(frame),
      .in_valid(in_valid),.in_ready(in_ready),.in_byte(byte_in),.in_last(in_last),
      .out_valid(valid),.out_ready(ready),.out_bits(bits_out),.out_n(n),.out_sc(sc),
      .out_sym_idx(sym),.out_stype(stype),.out_fseq(fq),.out_frame_start(fs),
      .out_frame_end(fe),.frame_done(done),.st_error(error));
    reg [31:0] expected[0:30000];
    reg held=0; reg [31:0] previous;
    wire [31:0] item={fs,fe,fq,stype,sym,sc,n,bits_out};
    always @(posedge clk) begin
        cycle<=cycle+1;
        if (!rst) begin
            if (held && (!valid || item!==previous)) $fatal(1,"unstable stalled output");
            held<=valid&&!ready; previous<=item;
            if (valid&&ready) begin
                if (count>=expected_count || item!==expected[count])
                    $fatal(1,"item %0d got %h expected %h",count,item,expected[count]);
                count<=count+1;
            end
            if (error) $fatal(1,"unexpected protocol error");
            if(cycle>2000000) $fatal(1,"timeout");
        end
    end
    initial begin
        if (!$value$plusargs("case=%s",name)) $fatal(1,"need case");
        void'($value$plusargs("stall=%d",stall));
        fd=$fopen(name,"r");
        rc=$fscanf(fd,"%d %d %h %d %d\n",a,b,user_data,c,bytes);
        if(rc!=5) $fatal(1,"bad config");
        length_bits=16'(a);mod_code=3'(b);frame=2'(c);
        // Payload bytes precede expected records in the fixture.
        repeat(5) @(negedge clk); rst=0; cfg_valid=1;
        @(negedge clk);cfg_valid=0;
        for(i=0;i<bytes;i=i+1) begin
            rc=$fscanf(fd,"%d\n",a); if(rc!=1) $fatal(1,"bad byte");
            in_valid=1;byte_in=8'(a);in_last=i==bytes-1;
            @(negedge clk);
        end
        in_valid=0;in_last=0;
        while(!$feof(fd)) begin
            rc=$fscanf(fd,"%d %d %d %d %d %d %d %d\n",a,b,c,d,e,f,g,h);
            if(rc==8) begin
                expected[expected_count]={g[0],h[0],a[1:0],b[2:0],c[7:0],d[7:0],e[2:0],f[5:0]};
                expected_count=expected_count+1;
            end
        end
        $fclose(fd);
        wait(done); @(negedge clk);
        if(count!=expected_count) $fatal(1,"missing output");
        $display("PASS items=%0d cycles=%0d",count,cycle);$finish;
    end
endmodule
