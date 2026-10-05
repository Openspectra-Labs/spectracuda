// Integration test for the CFO pair: cfo_estimate -> cfo_correct.
//
// Feeds P from spectracuda at the detected peak, then streams the same
// received samples spectracuda's correct() was given, and dumps what the
// hardware produced. Scoring is in check_cfo.py against
// cfo_corrected.txt -- the reference is always the Python block, never a
// second Verilog implementation of the same formula.
`timescale 1ns / 1ps
`include "build/cfo_params.vh"

module cfo_tb;
    localparam integer SAMPLE_W = 16, ANGLE_W = 16, STAGES = 16;
    localparam integer ACC_W = 48, DATA_W = 20, LOG2_LAG = 7;
    localparam integer NMAX = `CFO_N_SAMPLES;
    localparam integer N    = `CFO_N_ACTUAL;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg  signed [ACC_W-1:0]   p_re = 0, p_im = 0;
    reg                       p_valid = 0;
    wire signed [ANGLE_W-1:0] angle_turns, d_phase;
    wire                      est_valid;

    cfo_estimate #(.ACC_W(ACC_W), .DATA_W(DATA_W), .ANGLE_W(ANGLE_W),
                   .STAGES(STAGES), .LOG2_LAG(LOG2_LAG)) est (
        .clk(clk), .rst(rst), .p_re(p_re), .p_im(p_im), .p_valid(p_valid),
        .angle_turns(angle_turns), .d_phase(d_phase), .out_valid(est_valid)
    );

    // d_phase is registered one cycle after est_valid, so the load pulse
    // trails it by one.
    reg est_valid_d = 0;
    always @(posedge clk) est_valid_d <= est_valid;

    reg  signed [SAMPLE_W-1:0] in_i = 0, in_q = 0;
    reg                        in_valid = 0;
    wire signed [SAMPLE_W-1:0] out_i, out_q;
    wire                       out_valid;

    cfo_correct #(.SAMPLE_W(SAMPLE_W), .ANGLE_W(ANGLE_W), .STAGES(STAGES),
                  .LOG2_LAG(LOG2_LAG)) cor (
        .clk(clk), .rst(rst),
        .d_phase(d_phase), .d_phase_valid(est_valid_d),
        .in_i(in_i), .in_q(in_q), .in_valid(in_valid),
        .out_i(out_i), .out_q(out_q), .out_valid(out_valid)
    );

    reg [31:0] stim [0:NMAX-1];
    reg [47:0] pval [0:1];
    integer fd, fa, i, got = 0;

    always @(posedge clk) begin
        if (!rst && out_valid) begin
            $fwrite(fd, "%0d %0d\n", $signed(out_i), $signed(out_q));
            got = got + 1;
        end
        if (!rst && est_valid)
            $fwrite(fa, "%0d %0d\n", $signed(angle_turns), $signed(d_phase));
    end

    initial begin
        $readmemh("build/cfo_stimulus.hex", stim);
        $readmemh("build/cfo_p.hex", pval);
        fd = $fopen("build/cfo_out.txt", "w");
        fa = $fopen("build/cfo_angle.txt", "w");
        if (fd == 0 || fa == 0) begin $display("FAIL: cannot open outputs"); $finish; end

        repeat (4) @(posedge clk); rst = 0; @(posedge clk);

        // One-shot: hand P over, then wait for the estimate to fall out
        // and load before any sample is presented.
        @(negedge clk);
        p_re = pval[0]; p_im = pval[1]; p_valid = 1'b1;
        @(negedge clk);
        p_valid = 1'b0;
        repeat (STAGES + 8) @(posedge clk);

        for (i = 0; i < N; i = i + 1) begin
            @(negedge clk);
            in_i = stim[i][31:16]; in_q = stim[i][15:0]; in_valid = 1'b1;
        end
        @(negedge clk); in_valid = 1'b0;
        repeat (STAGES + 8) @(posedge clk);

        $fclose(fd); $fclose(fa);
        $display("cfo_tb: fed %0d samples, got %0d corrected", N, got);
        $finish;
    end
endmodule
