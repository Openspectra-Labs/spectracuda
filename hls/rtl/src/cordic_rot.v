// ============================================================
// cordic_rot.v -- CORDIC in ROTATION mode: rotate (x,y) by z
//
// The CFO derotator. spectracuda applies
// `rx[n] * exp(-j*2*pi*cfo*n/fft_size)`
// (spectracuda/cfo/schmidl_cox.py's correct()); its numba path already
// expresses that as a phase-accumulator recurrence rather than a cos/sin
// per sample, which is exactly the hardware form -- so this block takes
// the accumulated phase and rotates, and the accumulator lives upstream.
//
// Angle units are TURNS (see cordic_vec.v): 2^ANGLE_W is a full turn, so
// the accumulator's natural wraparound IS modulo-2*pi and needs no range
// check.
//
// ZERO DSP -- shift and add only. The alternative, a sin/cos ROM plus a
// complex multiply, costs ~4 DSP and a BRAM instead of these adders.
// That trade is worth revisiting once the whole chain's DSP budget is
// known: openofdm's complete receiver uses 57.5% of this part's DSPs, so
// DSP is the binding resource and LUTs may be the cheaper currency here.
// Measured, not assumed -- both fit, and the choice should follow the
// budget rather than taste.
//
// CONVERGENCE. Rotation mode converges for |z| <= sum atan(2^-i) ~=
// 1.7433 rad = 0.2775 turns, which is wider than a quarter turn but far
// short of the half turn the accumulator produces. So a z outside the
// quarter turn is folded first: negate (x,y) and add half a turn to z.
// In turn units that addition is an MSB inversion and wraparound makes
// its sign irrelevant.
//
// GAIN. Rotation scales the vector by K = prod sqrt(1+2^-2i) ~= 1.6468,
// so the input is pre-scaled by 1/K (generated as `CORDIC_INV_K, Q15).
// Vectoring mode needs no such correction because only its angle is
// used; this mode does, and forgetting it shows up as a uniform 65%
// amplitude error rather than as anything obviously structural.
// ============================================================
`timescale 1ns / 1ps
`include "generated/cordic_table.vh"

module cordic_rot #(
    parameter integer DATA_W  = 16,
    parameter integer ANGLE_W = 16,
    parameter integer STAGES  = 16
)(
    input  wire                      clk,
    input  wire                      rst,
    input  wire signed [DATA_W-1:0]  in_x,
    input  wire signed [DATA_W-1:0]  in_y,
    input  wire signed [ANGLE_W-1:0] in_z,      // phase, in turns
    input  wire                      in_valid,
    output wire signed [DATA_W-1:0]  out_x,
    output wire signed [DATA_W-1:0]  out_y,
    output wire                      out_valid
);

    // Guard bits for the CORDIC's intermediate growth. The 1/K prescale
    // keeps the FINAL magnitude at the input's, but intermediate stages
    // still swing above it.
    localparam integer W = DATA_W + 4;
    localparam signed [ANGLE_W-1:0] QUARTER = {2'b01, {(ANGLE_W-2){1'b0}}};
    localparam signed [ANGLE_W-1:0] NQUARTER = -QUARTER;
    localparam signed [ANGLE_W-1:0] HALF = {1'b1, {(ANGLE_W-1){1'b0}}};

    reg signed [ANGLE_W-1:0] atan_lut [0:STAGES-1];
    initial begin
        `CORDIC_ATAN_INIT
    end

    reg signed [W-1:0]       x [0:STAGES];
    reg signed [W-1:0]       y [0:STAGES];
    reg signed [ANGLE_W-1:0] z [0:STAGES];
    reg                      v [0:STAGES];

    // 1/K prescale, Q15. Two multiplies, and the only ones in the block.
    //
    // The narrowing below is EXPLICIT rather than lint-waived, because
    // WIDTHTRUNC is the class that loses bits silently. It is provably
    // safe: |in_x| <= 2^(DATA_W-1) and INV_K < 2^15, so
    // |px >>> 15| < 2^(DATA_W-1) -- the result always fits DATA_W bits,
    // and W is DATA_W+4. Widening DATA_W or changing INV_K keeps that
    // relation, so the bound is structural, not a coincidence of today's
    // constants.
    wire signed [DATA_W+16:0] px = in_x * `CORDIC_INV_K;
    wire signed [DATA_W+16:0] py = in_y * `CORDIC_INV_K;
    wire signed [DATA_W+16:0] px_s = px >>> 15;
    wire signed [DATA_W+16:0] py_s = py >>> 15;
    wire signed [W-1:0] sx = {{(W-DATA_W){px_s[DATA_W-1]}}, px_s[DATA_W-1:0]};
    wire signed [W-1:0] sy = {{(W-DATA_W){py_s[DATA_W-1]}}, py_s[DATA_W-1:0]};

    wire fold = (in_z >= QUARTER) || (in_z < NQUARTER);

    integer i;
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
            // ---- stage 0: prescale, and fold into the convergence range ----
            v[0] <= in_valid;
            if (fold) begin
                x[0] <= -sx;
                y[0] <= -sy;
                z[0] <= in_z + HALF;      // wraps; sign irrelevant
            end else begin
                x[0] <= sx;
                y[0] <= sy;
                z[0] <= in_z;
            end

            // ---- rotation: drive z to zero ----
            for (i = 0; i < STAGES; i = i + 1) begin
                v[i+1] <= v[i];
                if (z[i] >= 0) begin
                    x[i+1] <= x[i] - (y[i] >>> i);
                    y[i+1] <= y[i] + (x[i] >>> i);
                    z[i+1] <= z[i] - atan_lut[i];
                end else begin
                    x[i+1] <= x[i] + (y[i] >>> i);
                    y[i+1] <= y[i] - (x[i] >>> i);
                    z[i+1] <= z[i] + atan_lut[i];
                end
            end
        end
    end
    /* verilator lint_on WIDTHEXPAND */

    assign out_x     = x[STAGES][DATA_W-1:0];
    assign out_y     = y[STAGES][DATA_W-1:0];
    assign out_valid = v[STAGES];
endmodule
