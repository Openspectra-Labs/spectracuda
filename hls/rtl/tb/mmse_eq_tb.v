// MMSE equalizer: feeds real (rx, H) pairs from a multipath frame and
// dumps the equalized symbols. Scoring in check_eq.py against
// spectracuda's own MMSEEqualizer.
`timescale 1ns / 1ps
`include "build/eq_tb_params.vh"

module mmse_eq_tb;
    localparam integer W = 18, N = `EQ_N;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg  signed [W-1:0] rx_re = 0, rx_im = 0, h_re = 0, h_im = 0;
    reg                 in_valid = 0;
    wire signed [W-1:0] y_re, y_im;
    wire                y_valid;

    // Metadata check (step 3a): tag every input with its sequence number;
    // each output must carry the next number, in order.
    reg  [15:0] in_meta = 16'd0;
    wire [15:0] y_meta;
    integer exp_m = 0;
    always @(posedge clk) begin
        if (in_valid) in_meta <= in_meta + 1'b1;
        if (y_valid) begin
            if (y_meta != exp_m[15:0])
                $fatal(1, "mmse_eq: y_meta=%0d, want %0d", y_meta, exp_m);
            exp_m = exp_m + 1;
        end
    end

    mmse_eq #(.W(W), .META_W(16)) dut (
        .clk(clk), .rst(rst),
        .rx_re(rx_re), .rx_im(rx_im), .h_re(h_re), .h_im(h_im),
        .in_valid(in_valid), .in_meta(in_meta),
        .y_re(y_re), .y_im(y_im), .y_valid(y_valid), .y_meta(y_meta)
    );

    reg [71:0] stim [0:N-1];
    integer fd, i, got = 0;

    always @(posedge clk)
        if (!rst && y_valid) begin
            $fwrite(fd, "%0d %0d\n", $signed(y_re), $signed(y_im));
            got = got + 1;
        end

    initial begin
        $readmemh(`EQ_STIM_PATH, stim);
        fd = $fopen(`EQ_OUT_PATH, "w");
        if (fd == 0) begin $display("FAIL: cannot open output"); $finish; end
        repeat (4) @(posedge clk); rst = 0; @(posedge clk);
        for (i = 0; i < N; i = i + 1) begin
            @(negedge clk);
            rx_re = stim[i][71:54]; rx_im = stim[i][53:36];
            h_re  = stim[i][35:18]; h_im  = stim[i][17:0];
            in_valid = 1'b1;
        end
        @(negedge clk); in_valid = 1'b0;
        repeat (24) @(posedge clk);
        $fclose(fd);
        $display("mmse_eq_tb: fed %0d, got %0d", N, got);
        $finish;
    end
endmodule
