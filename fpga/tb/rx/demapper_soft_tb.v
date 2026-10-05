`timescale 1ns / 1ps
// Feeds +in=FILE lines "mod y_re y_im kcnt" (mod: 0 QPSK, 1 16QAM, 2 64QAM)
// through demapper_soft and prints "n l0 l1 l2 l3 l4 l5" per item.
module demapper_soft_tb;
    reg clk = 0, rst = 1; always #5 clk = ~clk;
    reg signed [17:0] yr = 0, yi = 0; reg [2:0] k = 0; reg [1:0] md = 0; reg v = 0;
    wire [23:0] llr; wire [3:0] n; wire ov; wire [0:0] om;
    demapper_soft #(.W(18), .META_W(1)) dut(.clk(clk), .rst(rst), .y_re(yr), .y_im(yi), .kcnt(k),
        .mod_scheme(md), .in_valid(v), .in_meta(1'b0), .llr(llr), .n_bits(n), .out_valid(ov), .out_meta(om));
    integer fd, r, a, b, c, d, nin = 0, nout = 0;
    reg [1023:0] fn;
    always @(posedge clk) if (ov) begin
        $display("O %0d %0d %0d %0d %0d %0d %0d", n, $signed(llr[3:0]), $signed(llr[7:4]), $signed(llr[11:8]),
                 $signed(llr[15:12]), $signed(llr[19:16]), $signed(llr[23:20]));
        nout = nout + 1;
    end
    initial begin
        if (!$value$plusargs("in=%s", fn)) $fatal(1, "need +in");
        fd = $fopen(fn, "r"); repeat (3) @(negedge clk); rst = 0;
        while ($fscanf(fd, "%d %d %d %d\n", a, b, c, d) == 4) begin
            md = a; yr = b; yi = c; k = d; v = 1; @(negedge clk); nin = nin + 1;
        end
        v = 0; repeat (20) @(negedge clk);
        if (nout != nin) $fatal(1, "out %0d of %0d", nout, nin);
        $finish;
    end
endmodule
