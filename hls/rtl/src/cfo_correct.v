// ============================================================
// cfo_correct.v -- apply the CFO estimate as a phase de-rotation
//
// spectracuda's correct() applies
//   corrected[n] = rx[n] * exp(-j*2*pi*cfo*n/fft_size)
// (spectracuda/cfo/schmidl_cox.py). Its numba path already expresses
// that as a phase-accumulator recurrence -- ONE cos/sin for the whole
// frame instead of one per sample -- because a naive per-sample cos/sin
// measured SLOWER than the vectorized path. That recurrence is exactly
// the hardware form, so this block is a direct transcription rather than
// a reinterpretation: accumulate, rotate, repeat.
//
// Sample 0 is rotated by phase 0 (the exponent is zero at n=0), so the
// accumulator advances AFTER each sample, not before. Getting that
// backwards costs one increment of static phase error on every frame --
// small, constant, and invisible unless compared against the reference.
//
// THE ACCUMULATOR CARRIES LOG2_LAG EXTRA FRACTIONAL BITS, and this is
// the whole trick. The required per-sample increment is
// -angle_turns / L. Doing that division up front as a shift truncates,
// and a phase accumulator INTEGRATES the truncation bias: measured at
// +0.947 LSB/sample, which reached 21 degrees of drift and 21.4% EVM
// over one 4112-sample frame. Rounding halves it and is still wrong.
//
// Instead the accumulator runs at ANGLE_W + LOG2_LAG bits and advances
// by the full -angle_turns each sample, with the CORDIC reading the top
// ANGLE_W bits. After n samples the accumulator holds -n*angle_turns and
// its top bits are -n*angle_turns/L exactly -- the division is a wire,
// and there is no rounding error left to integrate. This is why
// emit_rtl.py refuses a fft_size whose half is not a power of two.
//
// Wraparound is still free: the accumulator is modulo 2^LOG2_LAG turns,
// and its top ANGLE_W bits are modulo one turn. No range check, no
// conditional subtract.
// ============================================================
`timescale 1ns / 1ps

module cfo_correct #(
    parameter integer SAMPLE_W = 16,
    parameter integer ANGLE_W  = 16,
    parameter integer STAGES   = 16,
    parameter integer LOG2_LAG = 7    // fractional bits below the phase word
)(
    input  wire                       clk,
    input  wire                       rst,

    // Loaded once per frame from cfo_estimate (which supplies
    // -angle_turns, NOT a pre-divided increment); also clears the
    // accumulator, so each frame starts from zero phase.
    input  wire signed [ANGLE_W-1:0]  d_phase,
    input  wire                       d_phase_valid,

    input  wire signed [SAMPLE_W-1:0] in_i,
    input  wire signed [SAMPLE_W-1:0] in_q,
    input  wire                       in_valid,

    output wire signed [SAMPLE_W-1:0] out_i,
    output wire signed [SAMPLE_W-1:0] out_q,
    output wire                       out_valid
);

    localparam integer ACC_PH_W = ANGLE_W + LOG2_LAG;

    reg signed [ANGLE_W-1:0]  step;
    reg signed [ACC_PH_W-1:0] phase_acc;

    // The CORDIC sees only the whole-LSB part of the phase.
    wire signed [ANGLE_W-1:0] phase = phase_acc[ACC_PH_W-1:LOG2_LAG];

    /* verilator lint_off WIDTHEXPAND */
    always @(posedge clk) begin
        if (rst) begin
            step      <= {ANGLE_W{1'b0}};
            phase_acc <= {ACC_PH_W{1'b0}};
        end else begin
            if (d_phase_valid) begin
                step      <= d_phase;
                phase_acc <= {ACC_PH_W{1'b0}};   // new frame, zero phase
            end else if (in_valid) begin
                phase_acc <= phase_acc + step;   // wraps == modulo 2*pi
            end
        end
    end
    /* verilator lint_on WIDTHEXPAND */

    cordic_rot #(.DATA_W(SAMPLE_W), .ANGLE_W(ANGLE_W), .STAGES(STAGES)) rot_inst (
        .clk(clk), .rst(rst),
        .in_x(in_i), .in_y(in_q), .in_z(phase), .in_valid(in_valid),
        .out_x(out_i), .out_y(out_q), .out_valid(out_valid)
    );
endmodule
