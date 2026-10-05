// ============================================================
// cfo_estimate.v -- Schmidl & Cox CFO estimate from P
//
// spectracuda defines `cfo = angle(P) / pi`
// (spectracuda/cfo/schmidl_cox.py), where P is the preamble
// self-correlation that sc_sync_rtl ALREADY computes. So this block adds
// no correlation datapath -- it takes p_re/p_im straight from the
// detector at its peak and turns them into a phase increment.
//
// Output is the per-sample phase increment, not the cfo value, because
// nothing downstream wants the cfo value. spectracuda's correct() applies
//   rx[n] * exp(-j*2*pi*cfo*n/fft_size),
// and with phase held as a fraction of a turn that is
//   d_phase = -(angle(P)/2pi) / (fft_size/2) = -angle_turns / L.
//
// THAT DIVISION MUST NOT BE A TRUNCATING SHIFT. Computing
// `-(angle_turns >>> log2(L))` here looks free and is wrong: it discards
// up to one LSB of increment, in the SAME direction every sample, and a
// phase accumulator integrates that bias. Measured on a real frame:
// angle 1786 shifted to -13 instead of -13.95, an error of +0.947
// LSB/sample, which over 4112 samples became 21 degrees of drift and
// 21.4% EVM. Rounding instead of truncating only halves it -- still 11
// degrees. Same failure as ap_fixed's AP_TRN default in the HLS build,
// wearing a different hat.
//
// So the shift is not done here at all. This block emits the NEGATED
// ANGLE, and cfo_correct carries log2(L) extra fractional bits below the
// phase word. Its accumulator then advances by exactly -angle_turns per
// sample and the CORDIC reads the top bits, which makes the division by
// L exact -- no rounding error to integrate. See cfo_correct.v.
//
// emit_rtl.py still refuses configurations where fft_size/2 is not a
// power of two: the exactness above depends on it.
//
// NORMALIZATION IS THE POINT OF THIS BLOCK. P is a 48-bit accumulator
// but the CORDIC takes DATA_W bits, and CORDIC cannot resolve an angle
// it has no magnitude to work with -- feed it a few LSBs and the shifted
// terms underflow to zero and the answer is meaningless (measured: an
// input of magnitude 1 gave an error of 1798 LSB against numpy). So both
// components are left-shifted by the same amount until the larger fills
// the word. Scaling both by a common factor cannot change their ratio,
// and the ratio is all atan2 uses.
// ============================================================
`timescale 1ns / 1ps
`include "generated/ofdm_params.vh"

module cfo_estimate #(
    parameter integer ACC_W   = 48,   // width of P from sc_sync_rtl
    parameter integer DATA_W  = 20,   // CORDIC input width
    parameter integer ANGLE_W = 16,
    parameter integer STAGES  = 16,
    parameter integer LOG2_LAG = 7    // log2(fft_size/2)
)(
    input  wire                      clk,
    input  wire                      rst,
    input  wire signed [ACC_W-1:0]   p_re,
    input  wire signed [ACC_W-1:0]   p_im,
    input  wire                      p_valid,
    output wire signed [ANGLE_W-1:0] angle_turns,  // angle(P)/2pi
    output reg  signed [ANGLE_W-1:0] d_phase,      // -angle_turns; see above
    output wire                      out_valid
);

    // ---- normaliser, PIPELINED over three stages ----
    // The shift must be common to both components to preserve the ratio,
    // so the count is taken on whichever is larger.
    //
    // This was one combinational block from the p_re/p_im ports all the
    // way to cx/cy: two 48-bit conditional negates, a 48-bit OR, a
    // 48-bit priority encoder written as a loop, and two 48-bit variable
    // barrel shifts. Post-route that was the worst path in rx_top --
    // 23 logic levels, 14.36 ns, WNS -4.412 -- with 96 wires
    // (p_re + p_im) crossing from sc_sync to feed it.
    //
    // Splitting it costs 2 cycles and ~150 FFs. Both are free here:
    // cfo_estimate runs ONCE PER FRAME, on the detection pulse, and the
    // result is not needed until the training symbol ~256 samples later.
    // openofdm makes the same trade far harder in phase.v -- a 36-cycle
    // divider with a 37-cycle delayT to realign the quadrant.

    // -- stage 1: absolute values and their OR --
    reg signed [ACC_W-1:0] p_re_1, p_im_1;
    reg        [ACC_W-1:0] mag_1;
    reg                    v1;
    always @(posedge clk) begin
        if (rst) begin
            p_re_1 <= {ACC_W{1'b0}}; p_im_1 <= {ACC_W{1'b0}};
            mag_1  <= {ACC_W{1'b0}}; v1 <= 1'b0;
        end else begin
            p_re_1 <= p_re;
            p_im_1 <= p_im;
            mag_1  <= ((p_re < 0) ? -p_re : p_re) | ((p_im < 0) ? -p_im : p_im);
            v1     <= p_valid;
        end
    end

    // -- stage 2: priority encode --
    integer k;
    reg [7:0] hi_c;
    always @* begin
        hi_c = 8'd0;
        // Ascending, so the last assignment that fires is the highest
        // set bit -- a priority encoder written as a loop.
        for (k = 0; k < ACC_W - 1; k = k + 1)
            if (mag_1[k]) hi_c = k[7:0];
    end

    reg signed [ACC_W-1:0] p_re_2, p_im_2;
    reg        [7:0]       hi_2;
    reg                    v2;
    always @(posedge clk) begin
        if (rst) begin
            p_re_2 <= {ACC_W{1'b0}}; p_im_2 <= {ACC_W{1'b0}};
            hi_2 <= 8'd0; v2 <= 1'b0;
        end else begin
            p_re_2 <= p_re_1;
            p_im_2 <= p_im_1;
            hi_2   <= hi_c;
            v2     <= v1;
        end
    end

    // -- stage 3: barrel shift into the CORDIC's input width --
    // Land the top bit just below the sign so nothing overflows.
    /* verilator lint_off WIDTHTRUNC */
    localparam [7:0] TARGET = DATA_W - 2;
    /* verilator lint_on WIDTHTRUNC */
    wire [7:0] sh = (hi_2 >= TARGET) ? (hi_2 - TARGET) : 8'd0;
    wire [7:0] sl = (hi_2 <  TARGET) ? (TARGET - hi_2) : 8'd0;

    wire signed [ACC_W-1:0] nr = (p_re_2 <<< sl) >>> sh;
    wire signed [ACC_W-1:0] ni = (p_im_2 <<< sl) >>> sh;

    reg signed [DATA_W-1:0] cx, cy;
    reg                     cv;
    always @(posedge clk) begin
        if (rst) begin
            cx <= {DATA_W{1'b0}};
            cy <= {DATA_W{1'b0}};
            cv <= 1'b0;
        end else begin
            cx <= nr[DATA_W-1:0];
            cy <= ni[DATA_W-1:0];
            cv <= v2;
        end
    end

    cordic_vec #(.DATA_W(DATA_W), .ANGLE_W(ANGLE_W), .STAGES(STAGES)) atan_inst (
        .clk(clk), .rst(rst),
        .in_x(cx), .in_y(cy), .in_valid(cv),
        .out_angle(angle_turns), .out_mag(), .out_valid(out_valid)
    );

    // Just the negate -- the '-' in spectracuda's exp(-j...). The
    // division by L happens in cfo_correct's wider accumulator, where it
    // is exact. LOG2_LAG is still a parameter here so the two blocks can
    // be checked against each other for consistency.
    always @(posedge clk) begin
        if (rst) d_phase <= {ANGLE_W{1'b0}};
        else if (out_valid) d_phase <= -angle_turns;
    end
endmodule
