// Demapper: feeds noisy constellation symbols and dumps the hard bits.
// Scored bit-exactly against Modem.demodulate() in check_dm.py -- this is
// an integer decision, so "close" is not a thing: either the boundary is
// in the same place as Python's or it is not.
`timescale 1ns / 1ps
`include "build/dm_tb_params.vh"

module demapper_tb;
    localparam integer W = 18, N = `DM_N;
    // A macro cannot be bit-selected directly -- `DM_SCHEME[1:0]`
    // is a syntax error. Give it a sized localparam first.
    localparam [1:0] SCHEME = `DM_SCHEME;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg  signed [W-1:0] y_re = 0, y_im = 0;
    reg                 in_valid = 0;
    wire [5:0]          bits;
    wire [3:0]          n_bits;
    wire                out_valid;

    demapper #(.W(W)) dut (
        .clk(clk), .rst(rst), .y_re(y_re), .y_im(y_im),
        .mod_scheme(SCHEME), .in_valid(in_valid),
        .bits(bits), .n_bits(n_bits), .out_valid(out_valid)
    );

    reg [35:0] stim [0:N-1];
    integer fd, i, j, got = 0;

    always @(posedge clk)
        if (!rst && out_valid) begin
            // MSB-first, only the n_bits that are live for this scheme.
            for (j = 0; j < `DM_BPS; j = j + 1)
                $fwrite(fd, "%0d", bits[5 - j]);
            $fwrite(fd, "\n");
            got = got + 1;
        end

    initial begin
        $readmemh(`DM_STIM_PATH, stim);
        fd = $fopen(`DM_OUT_PATH, "w");
        if (fd == 0) begin $display("FAIL: cannot open output"); $finish; end
        repeat (4) @(posedge clk); rst = 0; @(posedge clk);
        for (i = 0; i < N; i = i + 1) begin
            @(negedge clk);
            y_re = stim[i][35:18]; y_im = stim[i][17:0];
            in_valid = 1'b1;
        end
        @(negedge clk); in_valid = 1'b0;
        repeat (12) @(posedge clk);
        $fclose(fd);
        $display("demapper_tb: fed %0d, got %0d", N, got);
        $finish;
    end
endmodule
