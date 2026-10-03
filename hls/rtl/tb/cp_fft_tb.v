// CP-strip + FFT test. Feeds one full slot (cyclic prefix included) and
// dumps the frequency-domain output.
//
// The transform itself is a vendor core and is taken as correct; what is
// under test is the wrapper -- whether the CP is stripped at the right
// offset, whether tlast frames the transform correctly, and whether the
// output convention matches numpy's. Scoring is in check_fft.py against
// spectracuda.
//
// Runs under Vivado xsim rather than Verilator: the generated FFT core is
// VHDL, which Verilator does not read. Every other block in hls/rtl stays
// on Verilator for the fast loop; only this one needs the slower
// simulator, and only because of the vendor core.
`timescale 1ns / 1ps
`include "build/fft_params.vh"

module cp_fft_tb;
    localparam integer SAMPLE_W = 16;
    localparam integer OUT_W    = 32;
    localparam integer N_IN     = `FFT_N_IN;
    localparam integer N_OUT    = `FFT_N_OUT;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg  signed [SAMPLE_W-1:0] in_i = 0, in_q = 0;
    reg                        in_valid = 0, sof = 0;
    wire signed [OUT_W-1:0]    out_re, out_im;
    wire                       out_valid, out_last, overflow;

    cp_fft #(.SAMPLE_W(SAMPLE_W), .FFT_SIZE(`FFT_SIZE),
             .CP_LEN(`FFT_CP_LEN), .OUT_W(OUT_W)) dut (
        .clk(clk), .rst(rst),
        .in_i(in_i), .in_q(in_q), .in_valid(in_valid), .sof(sof), .in_stype(2'd0),
        .out_re(out_re), .out_im(out_im),
        .out_valid(out_valid), .out_last(out_last), .out_sof(), .out_stype(), .overflow(overflow)
    );

    reg [31:0] stim [0:N_IN-1];
    integer fd, i, got = 0;

    always @(posedge clk) begin
        if (!rst && out_valid) begin
            $fwrite(fd, "%0d %0d\n", $signed(out_re), $signed(out_im));
            got = got + 1;
        end
    end

    initial begin
        $readmemh(`FFT_STIM_PATH, stim);
        fd = $fopen(`FFT_OUT_PATH, "w");
        if (fd == 0) begin $display("FAIL: cannot open fft_out.txt"); $finish; end

        repeat (8) @(posedge clk);
        rst = 0;
        // Let the core accept its config word before any data.
        repeat (16) @(posedge clk);

        for (i = 0; i < N_IN; i = i + 1) begin
            @(negedge clk);
            in_i     = stim[i][31:16];
            in_q     = stim[i][15:0];
            in_valid = 1'b1;
            sof      = (i == 0);
        end
        @(negedge clk);
        in_valid = 1'b0; sof = 1'b0;

        // Pipelined-streaming latency for N=256 is a few hundred cycles;
        // wait generously rather than guess it exactly.
        repeat (3000) @(posedge clk);

        $fclose(fd);
        $display("cp_fft_tb: fed %0d samples, got %0d bins, overflow=%0d",
                 N_IN, got, overflow);
        if (overflow) $display("FAIL: core deasserted tready and a sample was dropped");
        $finish;
    end
endmodule
