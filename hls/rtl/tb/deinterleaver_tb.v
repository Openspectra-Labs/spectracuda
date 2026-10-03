// Block de-interleaver: feeds an interleaved block produced by
// spectracuda's own BlockInterleaver and dumps the de-interleaved bytes.
// Bit-exact scoring -- a permutation is exact or it is wrong.
//
// The producer here honours `in_ready`, which is the point: the address
// generator stalls on the grid's virtual padding cells, and a testbench
// that ignored the handshake would silently feed data into those slots.
`timescale 1ns / 1ps
`include "build/deint_tb_params.vh"

module deinterleaver_tb;
    localparam integer UNIT_W = 8, N = `DI_N, CNT_W = 13;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg  [CNT_W-1:0] n_units = `DI_N, rows = `DI_ROWS, cols = `DI_COLS;
    reg              start = 0;
    reg  [UNIT_W-1:0] in_unit = 0;
    reg              in_valid = 0;
    wire             in_ready;
    wire [UNIT_W-1:0] out_unit;
    wire             out_valid, out_last;

    deinterleaver #(.UNIT_W(UNIT_W), .MAX_UNITS(4096), .CNT_W(CNT_W)) dut (
        .clk(clk), .rst(rst),
        .n_units(n_units), .rows(rows), .cols(cols), .start(start),
        .in_unit(in_unit), .in_valid(in_valid), .in_ready(in_ready),
        .out_unit(out_unit), .out_valid(out_valid), .out_last(out_last)
    );

    reg [7:0] stim [0:N-1];
    integer fd, i, got = 0;

    always @(posedge clk)
        if (!rst && out_valid) begin
            $fwrite(fd, "%0d\n", out_unit);
            got = got + 1;
        end

    initial begin
        $readmemh(`DI_STIM_PATH, stim);
        fd = $fopen(`DI_OUT_PATH, "w");
        if (fd == 0) begin $display("FAIL: cannot open output"); $finish; end

        repeat (4) @(posedge clk); rst = 0; @(posedge clk);
        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;

        i = 0;
        while (i < N) begin
            @(negedge clk);
            in_unit  = stim[i];
            in_valid = 1'b1;
            @(posedge clk);
            // Only advance when the block actually took the byte.
            if (in_ready) i = i + 1;
        end
        @(negedge clk); in_valid = 1'b0;

        repeat (N + 64) @(posedge clk);
        $fclose(fd);
        $display("deinterleaver_tb: fed %0d, got %0d", N, got);
        $finish;
    end
endmodule
