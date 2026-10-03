// LS channel estimator: feeds the received reference bins from a real
// multipath frame and dumps the interpolated estimate for every
// subcarrier. Scoring is in check_ce.py against spectracuda's own
// LSChannelEstimator.
`timescale 1ns / 1ps
`include "build/ce_params.vh"

module ls_chanest_tb;
    localparam integer IN_W = 20, H_W = 18;
    localparam integer N_IN = `CE_N_IN, N_BINS = `CE_N_BINS;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg  signed [IN_W-1:0] pilot_re = 0, pilot_im = 0;
    reg                    pilot_valid = 0;
    wire signed [H_W-1:0]  h_re, h_im;
    wire                   h_valid, h_last;

    ls_chanest #(.IN_W(IN_W), .H_W(H_W)) dut (
        .clk(clk), .rst(rst),
        .pilot_re(pilot_re), .pilot_im(pilot_im), .pilot_valid(pilot_valid),
        .h_re(h_re), .h_im(h_im), .h_valid(h_valid), .h_last(h_last)
    );

    reg [39:0] stim [0:N_IN-1];
    integer fd, i, got = 0, saw_last = 0;

    always @(posedge clk) begin
        if (!rst && h_valid) begin
            $fwrite(fd, "%0d %0d\n", $signed(h_re), $signed(h_im));
            got = got + 1;
            if (h_last) saw_last = 1;
        end
    end

    initial begin
        $readmemh(`CE_STIM_PATH, stim);
        fd = $fopen(`CE_OUT_PATH, "w");
        if (fd == 0) begin $display("FAIL: cannot open output"); $finish; end

        repeat (4) @(posedge clk); rst = 0; @(posedge clk);

        for (i = 0; i < N_IN; i = i + 1) begin
            @(negedge clk);
            pilot_re = stim[i][39:20];
            pilot_im = stim[i][19:0];
            pilot_valid = 1'b1;
        end
        @(negedge clk); pilot_valid = 1'b0;

        repeat (N_BINS + 32) @(posedge clk);
        $fclose(fd);
        $display("ls_chanest_tb: fed %0d refs, got %0d bins, saw_last=%0d",
                 N_IN, got, saw_last);
        $finish;
    end
endmodule
