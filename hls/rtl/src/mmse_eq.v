// ============================================================
// mmse_eq.v -- MMSE frequency-domain equalizer
//
// Port of spectracuda's MMSEEqualizer (spectracuda/equalizer/mmse.py):
//     w = conj(H) / (|H|^2 + noise_var)
//     y = rx * w
//
// THIS DIVISION IS REAL. The channel estimator's two divisions both had
// waveform-constant denominators and were evaluated at generate time;
// this one depends on the measured channel and cannot be. But it still
// does not need a divider.
//
// The denominator is real and positive, so: normalize it to a mantissa
// in [1,2), look up 1/mantissa in a ROM, and fold the exponent into the
// output shift. How large that ROM must be was MEASURED on real (rx, H)
// pairs, not assumed -- 256 entries gives 0.084% EVM against float,
// against a ~2% budget (32 entries would already pass at 0.654%). See
// hls/gen/emit_rtl.py's emit_eq for the sweep.
//
// openofdm spends 1,784 LUT on a div_gen instance for the same job.
//
// FIXED-POINT CONTRACT, rx and H both Q12:
//     den   = |H|^2 + noise_var            Q24
//     E     = index of den's highest set bit
//     LUT   = round(2^15 / (den >> E))     unsigned, (16384, 32768]
//     num   = rx * conj(H)                 Q24
//     y     = (num * LUT) >>> (3 + E)      Q12
// The shift is 3, not 15, because y is wanted in Q12 while num and den
// are both Q24: 24 - 24 + 12 - 15 = -3.
//
// The ROM is UNSIGNED deliberately. 1/m at m=1 is exactly 1.0, which in
// Q15 is 32768 -- the value that does NOT fit a signed 16-bit field and
// silently wrapped in the channel estimator's reciprocal table, flipping
// the sign of every bin whose reference was +1.
//
// Pipelined from the outset rather than after a failed synthesis.
// ls_chanest went through three rounds of that; the path there turned
// out to be ROM read -> two cascaded DSP48 -> BRAM write in one cycle,
// which is exactly the shape of stage e1/e2 here.
// ============================================================
`timescale 1ns / 1ps
`include "generated/eq_params.vh"

module mmse_eq #(
    parameter integer W     = 18,    // rx and H, Q12
    parameter integer ACC_W = 40,
    parameter integer PRD_W = 60,
    // Sideband carried through the SAME six stages as the sample (e1..e5,
    // y), so y_meta always belongs to y_re/y_im. Opaque here; >= 1 bit.
    parameter integer META_W = 1
)(
    input  wire                    clk,
    input  wire                    rst,

    input  wire signed [W-1:0]     rx_re,
    input  wire signed [W-1:0]     rx_im,
    input  wire signed [W-1:0]     h_re,
    input  wire signed [W-1:0]     h_im,
    input  wire                    in_valid,
    input  wire [META_W-1:0]       in_meta,

    output reg  signed [W-1:0]     y_re,
    output reg  signed [W-1:0]     y_im,
    output reg                     y_valid,
    output reg  [META_W-1:0]       y_meta
);
    reg [META_W-1:0] e1_m, e2_m, e3_m, e4_m, e5_m;

    localparam integer LUT_N  = 1 << `EQ_LUT_BITS;
    localparam integer EXP_W  = 6;

    reg [15:0] recip_rom [0:LUT_N-1];
    initial $readmemh(`EQ_RECIP_MEM, recip_rom);

    // ---- e1: the six products ----
    reg signed [2*W-1:0] e1_hh_re, e1_hh_im;          // |H|^2 parts
    reg signed [2*W-1:0] e1_rr, e1_ii, e1_ir, e1_ri;  // rx * conj(H) parts
    reg                  e1_v;

    // ---- e2: denominator and numerator ----
    reg signed [ACC_W-1:0] e2_den;
    reg signed [ACC_W-1:0] e2_num_re, e2_num_im;
    reg                    e2_v;

    // ---- e3: normalize ----
    reg [EXP_W-1:0]        e3_exp;
    reg [`EQ_LUT_BITS-1:0] e3_idx;
    reg signed [ACC_W-1:0] e3_num_re, e3_num_im;
    reg                    e3_v;

    // ---- e4: reciprocal ----
    reg [15:0]             e4_recip;
    reg [EXP_W-1:0]        e4_exp;
    reg signed [ACC_W-1:0] e4_num_re, e4_num_im;
    reg                    e4_v;

    // ---- e5: scale ----
    reg signed [PRD_W-1:0] e5_re, e5_im;
    reg [EXP_W-1:0]        e5_exp;
    reg                    e5_v;

    // Highest set bit of the denominator, as a priority encoder. den is
    // strictly positive (noise_var > 0), so this always finds one.
    integer k;
    reg [EXP_W-1:0] msb;
    always @* begin
        msb = 0;
        for (k = 0; k < ACC_W - 1; k = k + 1)
            if (e2_den[k]) msb = k[EXP_W-1:0];
    end
    // The EQ_LUT_BITS below the leading one index the ROM.
    wire [ACC_W-1:0] den_sh = e2_den >> (msb - `EQ_LUT_BITS);

    /* verilator lint_off WIDTHEXPAND */
    /* verilator lint_off WIDTHTRUNC */
    always @(posedge clk) begin
        if (rst) begin
            e1_v <= 0; e2_v <= 0; e3_v <= 0; e4_v <= 0; e5_v <= 0;
            y_valid <= 0; y_re <= 0; y_im <= 0;
        end else begin
            // e1
            e1_hh_re <= h_re * h_re;
            e1_hh_im <= h_im * h_im;
            e1_rr    <= rx_re * h_re;
            e1_ii    <= rx_im * h_im;
            e1_ir    <= rx_im * h_re;
            e1_ri    <= rx_re * h_im;
            e1_v     <= in_valid;
            e1_m     <= in_meta;

            // e2: rx * conj(H) = (rr + ii) + j(ir - ri)
            e2_den    <= e1_hh_re + e1_hh_im + `EQ_NOISE_VAR_Q24;
            e2_num_re <= e1_rr + e1_ii;
            e2_num_im <= e1_ir - e1_ri;
            e2_v      <= e1_v;
            e2_m      <= e1_m;

            // e3
            e3_exp    <= msb;
            e3_idx    <= den_sh[`EQ_LUT_BITS-1:0];
            e3_num_re <= e2_num_re;
            e3_num_im <= e2_num_im;
            e3_v      <= e2_v;
            e3_m      <= e2_m;

            // e4
            e4_recip  <= recip_rom[e3_idx];
            e4_exp    <= e3_exp;
            e4_num_re <= e3_num_re;
            e4_num_im <= e3_num_im;
            e4_v      <= e3_v;
            e4_m      <= e3_m;

            // e5
            e5_re  <= e4_num_re * $signed({1'b0, e4_recip});
            e5_im  <= e4_num_im * $signed({1'b0, e4_recip});
            e5_exp <= e4_exp;
            e5_v   <= e4_v;
            e5_m   <= e4_m;

            // e6
            y_re    <= (e5_re >>> (`EQ_OUT_SHIFT + e5_exp));
            y_im    <= (e5_im >>> (`EQ_OUT_SHIFT + e5_exp));
            y_valid <= e5_v;
            y_meta  <= e5_m;
        end
    end
    /* verilator lint_on WIDTHTRUNC */
    /* verilator lint_on WIDTHEXPAND */
endmodule
