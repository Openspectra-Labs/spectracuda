// Feeds one encoded frame through viterbi_dec and dumps the decoded bits.
// Scored against spectracuda's own decoder in check_viterbi.py.
`timescale 1ns / 1ps
`include "build/vit_tb_params.vh"

module viterbi_tb;
    localparam integer NSYM  = `VIT_NSYM;
    localparam integer NBITS = `VIT_NBITS;

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg  [1:0] sym = 2'b00;
    reg        in_valid = 0, last = 0, start = 0;
    wire       in_ready, out_bit, out_valid, frame_done;

    viterbi_dec dut (
        .clk(clk), .rst(rst), .start(start),
        .sym(sym), .in_valid(in_valid), .in_ready(in_ready), .last(last),
        .out_bit(out_bit), .out_valid(out_valid), .frame_done(frame_done)
    );

    reg [3:0] stim [0:NSYM-1];
    integer fd, i, got = 0;

    always @(posedge clk)
        if (!rst && out_valid) begin
            $fwrite(fd, "%0d\n", out_bit);
            got = got + 1;
        end

    initial begin
        $readmemh(`VIT_STIM_PATH, stim);
        fd = $fopen(`VIT_OUT_PATH, "w");
        if (fd == 0) begin $display("FAIL: cannot open output"); $finish; end

        repeat (4) @(posedge clk); rst = 0; @(posedge clk);
        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;

        i = 0;
        while (i < NSYM) begin
            @(negedge clk);
            if (in_ready) begin
                sym      = stim[i][1:0];
                in_valid = 1'b1;
                last     = (i == NSYM - 1);
                i        = i + 1;
            end else begin
                in_valid = 1'b0;
                last     = 1'b0;
            end
        end
        @(negedge clk); in_valid = 1'b0; last = 1'b0;

        // Let the final traceback drain.
        i = 0;
        while (i < 20000 && !frame_done) begin @(posedge clk); i = i + 1; end
        repeat (200) @(posedge clk);

        $fclose(fd);
        $display("viterbi_tb: fed %0d symbols, got %0d bits (expect >= %0d)",
                 NSYM, got, NBITS);
        $finish;
    end
endmodule
