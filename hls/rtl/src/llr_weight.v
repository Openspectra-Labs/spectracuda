// ============================================================
// llr_weight.v -- per-frame thresholds for the soft channel weight (0 DSP)
//
// spectracuda Ofdm soft_llr_metric="thresh_wq" (pipeline/ofdm.py) weights
// each subcarrier's soft values by 2^k,
//     k = -3 + #{ m in -3..0 : 216 * |H[k]|^2 >= sqrt(2) * sH * 2^m }
//     sH = sum of |H|^2 over the TRAINING estimate's 216 data subcarriers
// (|H[k]|^2 / mean rounded to a power of two in [1/8, 2]).
//
// sH WITHOUT A MULTIPLIER: the header symbol is equalized with exactly that
// training estimate, and mmse_eq already outputs |H|^2 beside every item
// (y_hh). So sH is the sum of y_hh over the first header symbol's 216 data
// items. Then, once per frame, serially (no DSP):
//     s2   = floor(sqrt(2) * sH)                 shift-add, sqrt(2) in Q24
//     th_j = ceil((s2 << j) / 1728),  j = 0..3   (1728 = 8 * 216; j = m + 3)
// and per data item (in rx_freq_domain) four comparisons:
//     kcnt = #{ j : |H[k]|^2 >= th_j },   k = kcnt - 3
// since 216 h2 >= sqrt(2) sH 2^m  <=>  1728 h2 >= s2 << j  <=>  h2 >= th_j
// for integer h2 (up to the floor in s2: a boundary-only difference).
//
// Valid ~250 clocks after the frame's first header symbol, tagged with its
// frame. FD releases a frame's DATA only when w_valid && w_fseq match. The
// next frame's header reaches the equalizer only after this frame's whole
// body has (B1 is in order), so a data item always sees its own frame's
// thresholds when it leaves the equalizer.
// ============================================================
`timescale 1ns / 1ps
module llr_weight #(
    parameter integer HW = 36                // |H|^2 width (Q24), mmse_eq y_hh
)(
    input  wire                clk,
    input  wire                rst,
    input  wire                s_valid,      // first header symbol item, at the equalizer output
    input  wire [HW-1:0]       s_hh,         // its |H|^2
    input  wire                s_last,       // that symbol's last data item
    input  wire [1:0]          s_fseq,
    output reg  [39:0]         th0, th1, th2, th3,
    output reg                 w_valid,
    output reg  [1:0]          w_fseq
);
    localparam [24:0] SQRT2_Q24 = 25'd23726566;     // sqrt(2) * 2^24
    localparam [10:0] D1728     = 11'd1728;

    localparam S_IDLE = 0, S_MUL = 1, S_DIV = 2;
    reg [1:0]  st;
    reg [44:0] acc, sh;
    reg [24:0] mb;
    reg [70:0] mp;           // sH * sqrt2_Q24
    reg [4:0]  mi;
    reg [1:0]  fq;
    // serial restoring division of (s2 << j) by 1728, one quotient bit per clock
    reg [49:0] num;
    reg [10:0] rem;
    reg [49:0] quo;
    reg [5:0]  di;
    reg [1:0]  j;
    wire [46:0] s2 = 47'(mp >> 24);
    wire [11:0] rem_sh = {rem, num[49]};
    wire [49:0] quo_n  = {quo[48:0], rem_sh >= {1'b0, D1728}};
    wire [10:0] rem_n  = (rem_sh >= {1'b0, D1728}) ? 11'(rem_sh - {1'b0, D1728}) : rem_sh[10:0];
    // ceil: add one when the division leaves a remainder
    wire [39:0] ceil_q = 40'(quo_n + ((rem_n != 11'd0) ? 50'd1 : 50'd0));

    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; acc <= 45'd0; w_valid <= 1'b0; w_fseq <= 2'd0;
            th0 <= 40'd0; th1 <= 40'd0; th2 <= 40'd0; th3 <= 40'd0;
        end else begin
            if (s_valid) acc <= s_last ? 45'd0 : acc + {{(45-HW){1'b0}}, s_hh};
            case (st)
            S_IDLE: if (s_valid && s_last) begin
                sh <= acc + {{(45-HW){1'b0}}, s_hh};
                mb <= SQRT2_Q24; mp <= 71'd0; mi <= 5'd0; fq <= s_fseq;
                w_valid <= 1'b0;
                st <= S_MUL;
            end
            S_MUL: begin
                // mp += sH << i for each set bit i of sqrt2_Q24 (25 clocks)
                if (mb[0]) mp <= mp + ({26'd0, sh} << mi);
                mb <= mb >> 1; mi <= mi + 1'b1;
                if (mi == 5'd24) begin
                    j <= 2'd0; st <= S_DIV; di <= 6'd0; rem <= 11'd0; quo <= 50'd0;
                end
            end
            S_DIV: begin
                if (di == 6'd0) begin
                    // load (s2 << j); the first quotient step is the next clock
                    num <= {3'd0, s2} << j; rem <= 11'd0; quo <= 50'd0; di <= 6'd1;
                end else begin
                    rem <= rem_n; quo <= quo_n; num <= {num[48:0], 1'b0};
                    if (di == 6'd50) begin
                        case (j)
                            2'd0: th0 <= ceil_q;
                            2'd1: th1 <= ceil_q;
                            2'd2: th2 <= ceil_q;
                            default: th3 <= ceil_q;
                        endcase
                        di <= 6'd0;
                        if (j == 2'd3) begin w_valid <= 1'b1; w_fseq <= fq; st <= S_IDLE; end
                        j <= j + 1'b1;
                    end else di <= di + 1'b1;
                end
            end
            default: st <= S_IDLE;
            endcase
        end
    end
endmodule
