// ============================================================
// cordic_vec.v -- CORDIC in VECTORING mode: atan2(y, x)
//
// Used by the CFO estimator. spectracuda defines
// `cfo = angle(P) / pi` (spectracuda/cfo/schmidl_cox.py), and P is
// exactly what sc_sync_rtl already emits, so the whole estimator is
// this one operation on values the detector produces anyway.
//
// ANGLE UNITS ARE TURNS, NOT RADIANS. The output is a signed fraction
// of a full turn in ANGLE_W bits: 2^ANGLE_W is one turn, so a half
// turn (pi) is the sign bit alone. Two reasons, both practical:
//   * two's-complement wraparound IS modulo-2*pi, so a phase
//     accumulator needs no range check;
//   * the CFO phase increment becomes a shift rather than a multiply
//     (see cfo_correct.v).
//
// No DSP: CORDIC is shift-and-add only. The arctan table comes from
// fpga/gen/emit_rtl.py rather than being typed in -- sixteen hand-
// transcribed constants is sixteen chances to leave a radian where a
// turn belongs, and that failure would present as a small CFO bias
// rather than as anything obviously broken.
//
// Vectoring mode converges only for x > 0, so a negative x is rotated
// by half a turn first and the half turn is added back. In turn units
// that addition is free: adding 2^(ANGLE_W-1) is inverting the MSB,
// and wraparound makes +half and -half the same value.
//
// Gain: vectoring scales the magnitude by K = prod sqrt(1+2^-2i), but
// only the ANGLE is used here, so no 1/K correction is needed. A
// rotation-mode CORDIC would need it.
// ============================================================
`timescale 1ns / 1ps
`include "generated/cordic_table.vh"

module cordic_vec #(
    parameter integer DATA_W  = 20,
    parameter integer ANGLE_W = 16,
    parameter integer STAGES  = 16
)(
    input  wire                      clk,
    input  wire                      rst,
    input  wire signed [DATA_W-1:0]  in_x,
    input  wire signed [DATA_W-1:0]  in_y,
    input  wire                      in_valid,
    output wire signed [ANGLE_W-1:0] out_angle,
    // Magnitude, scaled by the CORDIC gain K = prod sqrt(1+2^-2i)
    // ~= 1.6468. Purely additive -- x[STAGES] was always computed, it
    // just was not brought out. frame_sync uses it so it never has to
    // SQUARE anything: |P|^2/R^2 > T is the same decision as
    // |P| > sqrt(T)*R, and the gain folds into that one constant.
    output wire signed [DATA_W+1:0]  out_mag,
    output wire                      out_valid
);

    // Two guard bits: the CORDIC gain is ~1.647, so x grows by less
    // than one bit, and the pre-rotation negate needs one more.
    localparam integer W = DATA_W + 2;

    reg signed [ANGLE_W-1:0] atan_lut [0:STAGES-1];
    initial begin
        `CORDIC_ATAN_INIT
    end

    reg signed [W-1:0]       x [0:STAGES];
    reg signed [W-1:0]       y [0:STAGES];
    reg signed [ANGLE_W-1:0] z [0:STAGES];
    reg                      v [0:STAGES];

    integer i;
    // Sign-extending DATA_W inputs into the W-bit guarded datapath.
    // WIDTHEXPAND is the intent; WIDTHTRUNC stays on, since losing bits
    // silently is the failure this class of warning exists to catch.
    /* verilator lint_off WIDTHEXPAND */
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i <= STAGES; i = i + 1) begin
                x[i] <= {W{1'b0}};
                y[i] <= {W{1'b0}};
                z[i] <= {ANGLE_W{1'b0}};
                v[i] <= 1'b0;
            end
        end else begin
            // ---- stage 0: fold into the right half plane ----
            v[0] <= in_valid;
            if (in_x >= 0) begin
                x[0] <= in_x;
                y[0] <= in_y;
                z[0] <= {ANGLE_W{1'b0}};
            end else begin
                x[0] <= -in_x;
                y[0] <= -in_y;
                // Half a turn. Wraparound makes the sign irrelevant.
                z[0] <= {1'b1, {(ANGLE_W-1){1'b0}}};
            end

            // ---- vectoring: drive y to zero, accumulate the angle ----
            for (i = 0; i < STAGES; i = i + 1) begin
                v[i+1] <= v[i];
                if (y[i] >= 0) begin
                    x[i+1] <= x[i] + (y[i] >>> i);
                    y[i+1] <= y[i] - (x[i] >>> i);
                    z[i+1] <= z[i] + atan_lut[i];
                end else begin
                    x[i+1] <= x[i] - (y[i] >>> i);
                    y[i+1] <= y[i] + (x[i] >>> i);
                    z[i+1] <= z[i] - atan_lut[i];
                end
            end
        end
    end

    /* verilator lint_on WIDTHEXPAND */

    assign out_angle = z[STAGES];
    assign out_mag   = x[STAGES];
    assign out_valid = v[STAGES];
endmodule
