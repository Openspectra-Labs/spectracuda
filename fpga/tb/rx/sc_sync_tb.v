// Feeds sc_sync_rtl the SAME golden capture the HLS testbench uses --
// produced by spectracuda's own pipeline via fpga/gen/golden.py, then
// converted to int16 hex by fpga/sim/gen_stimulus.py.
//
// The block emits P and R, not the metric, so this testbench does not
// judge pass/fail: it dumps one (p_re, p_im, r_sum) triple per candidate
// and fpga/sim/check.py computes |P|^2/R^2, takes the argmax, and compares
// against what spectracuda's SchmidlCoxSync returned for that capture.
// Keeping the scoring in Python means the RTL and the golden model are
// never compared through two different implementations of the metric.
`timescale 1ns / 1ps
`include "build/stim_params.vh"

module sc_sync_tb;

    localparam integer SAMPLE_W = 16;
    localparam integer LAG      = 128;
    localparam integer ACC_W    = 48;
    localparam integer MAX_SAMP = 65536;

    reg clk = 0;
    reg rst = 1;
    always #5 clk = ~clk;          // 100 MHz

    reg  signed [SAMPLE_W-1:0] in_i = 0, in_q = 0;
    reg                        in_valid = 0;
    wire signed [ACC_W-1:0]    p_re, p_im, r_sum;
    wire                       out_valid;

    sc_sync_rtl #(.SAMPLE_W(SAMPLE_W), .LAG(LAG), .ACC_W(ACC_W)) dut (
        .clk(clk), .rst(rst),
        .in_i(in_i), .in_q(in_q), .in_valid(in_valid),
        .p_re(p_re), .p_im(p_im), .r_sum(r_sum), .out_valid(out_valid)
    );

    reg [31:0] stim [0:MAX_SAMP-1];
    integer n_samples;
    integer fd_out;
    integer i;

    // Count candidates as they come out, so the index written next to each
    // triple is directly comparable to Python's start_index.
    integer d_count = 0;

    always @(posedge clk) begin
        if (!rst && out_valid) begin
            $fwrite(fd_out, "%0d %0d %0d %0d\n",
                    d_count, $signed(p_re), $signed(p_im), $signed(r_sum));
            d_count = d_count + 1;
        end
    end

    initial begin
        // Sample count comes from the generated define, not from scanning
        // the array for untouched entries: Verilator is 2-state by default,
        // so an unwritten entry reads as 0 -- indistinguishable from the
        // 200 genuinely-silent samples the golden capture opens with.
        for (i = 0; i < MAX_SAMP; i = i + 1) stim[i] = 32'h0;
        $readmemh("build/stimulus.hex", stim);
        n_samples = `N_SAMPLES;

        fd_out = $fopen("build/rtl_metric.txt", "w");
        if (fd_out == 0) begin
            $display("FAIL: cannot open build/rtl_metric.txt");
            $finish;
        end

        repeat (4) @(posedge clk);
        rst = 0;
        @(posedge clk);

        for (i = 0; i < n_samples; i = i + 1) begin
            @(negedge clk);
            in_i     = stim[i][31:16];
            in_q     = stim[i][15:0];
            in_valid = 1'b1;
        end
        @(negedge clk);
        in_valid = 1'b0;

        repeat (10) @(posedge clk);
        $fclose(fd_out);
        $display("sc_sync_tb: fed %0d samples, emitted %0d candidates",
                 n_samples, d_count);
        $finish;
    end
endmodule
