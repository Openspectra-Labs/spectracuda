// ============================================================
// demapper_soft.v -- 4-bit soft-decision QAM demapper, table version
//
// Port of spectracuda Modem.demodulate_soft_tableq (modem/mapper.py), the
// soft metric the RTL implements (Ofdm soft_llr_metric="thresh_wq"):
//   per axis   idx = clip(floor(y * 2^s + 1/2), -256, 255)   s = 5/6/7 (QPSK/16/64)
//              T   = TABLE[{mod, idx}]   (3 x 11-bit, T = round(16 t) per axis bit)
//   per bit    q   = clip(round_half_even(7 T 2^k / 64), -7, 7)
// t = (L1 - L0)(2 x/norm - L0 - L1) is the max-log value with a fixed scale
// (L0 / L1: nearest levels whose Gray label bit is 0 / 1). The table is
// spectracuda's own Modem.soft_table(), emitted by hls/gen/emit_soft_table.py
// into ONE 2048 x 36 dual-port ROM (2 BRAM36): port A looks up I, port B Q.
// k (-3..1) is the subcarrier's channel power relative to the training
// mean rounded to a power of two (llr_weight.v), delivered as kcnt = k + 3,
// so the weight is a shift:  q = round(7 T / 2^(9 - kcnt)).
// NO multiplier: 0 DSP. Why this metric: examples/soft_metric_study.py and
// soft_tableq_check.py -- same frame error rate as the full-precision
// max-log with an exact |H|^2/noise weight, on AWGN and fading.
//
// Rounding: idx rounds half up (floor(+1/2), as the reference); q rounds
// half to even (numpy rint). SIGN: Python's q > 0 means bit 1; the RTL soft
// Viterbi takes "> 0 means bit 0", so the output is -q (hard decision =
// sign bit). BPSK (the header) is hard: +/-7 from the sign.
//
// Latency: 5 clocks, fixed, for every item (header and data alike, so the
// stream stays in order).
// ============================================================
`timescale 1ns / 1ps
`include "generated/demap_params.vh"
`include "generated/soft_table.vh"
module demapper_soft #(
    parameter integer W      = 18,      // y, Q12, from mmse_eq / pilot_cpe
    parameter integer META_W = 1
)(
    input  wire                clk,
    input  wire                rst,
    input  wire signed [W-1:0] y_re,
    input  wire signed [W-1:0] y_im,
    input  wire         [2:0]  kcnt,         // channel weight exponent + 3 (0..4)
    input  wire         [1:0]  mod_scheme,   // DM_QPSK / DM_QAM16 / DM_QAM64 / DM_BPSK
    input  wire                in_valid,
    input  wire  [META_W-1:0]  in_meta,
    output reg         [23:0]  llr,      // 6 x 4-bit signed, llr[k] at [4k +: 4], k = 0 first bit
    output reg          [3:0]  n_bits,   // 1 (BPSK), 2, 4 or 6
    output reg                 out_valid,
    output reg   [META_W-1:0]  out_meta
);
    // ---- the table: 2048 x 36, address {mod[1:0], idx[8:0]} ----
    (* rom_style = "block" *) reg [35:0] rom [0:2047];
    initial $readmemh(`SOFT_TABLE_MEM, rom);

    // ===== S1: table index per axis: idx = sat9(floor(y / 2^(12-s) + 1/2)) =====
    // y is Q12, so y * 2^s = y / 2^(12-s): a per-modulation right shift.
    function [8:0] tidx(input signed [W-1:0] y, input [1:0] md);
        reg signed [W:0] r;
        begin
            case (md)
                `DM_QAM16: r = ($signed({y[W-1], y}) + (19'sd1 <<< (11 - `SOFT_TABLE_SHIFT_QAM16))) >>> (12 - `SOFT_TABLE_SHIFT_QAM16);
                `DM_QAM64: r = ($signed({y[W-1], y}) + (19'sd1 <<< (11 - `SOFT_TABLE_SHIFT_QAM64))) >>> (12 - `SOFT_TABLE_SHIFT_QAM64);
                default:   r = ($signed({y[W-1], y}) + (19'sd1 <<< (11 - `SOFT_TABLE_SHIFT_QPSK)))  >>> (12 - `SOFT_TABLE_SHIFT_QPSK);
            endcase
            tidx = (r > 255) ? 9'd255 : (r < -256) ? 9'h100 : r[8:0];
        end
    endfunction
    reg [10:0]        s1_ar, s1_ai;      // {mod, idx}
    reg [3:0]         s1_half;
    reg [2:0]         s1_k;
    reg               s1_bpsk, s1_bbit, s1_v;
    reg [META_W-1:0]  s1_m;

    // ===== S2: table read (registered), S3: table output register =====
    reg [35:0]        s2_tr, s2_ti, s3_tr, s3_ti;
    reg [3:0]         s2_half, s3_half;
    reg [2:0]         s2_k, s3_k;
    reg               s2_bpsk, s2_bbit, s2_v, s3_bpsk, s3_bbit, s3_v;
    reg [META_W-1:0]  s2_m, s3_m;
    always @(posedge clk) begin
        s2_tr <= rom[s1_ar];             // port A: I axis
        s2_ti <= rom[s1_ai];             // port B: Q axis
    end

    // ===== S4: q = round_half_even(7 T / 2^(9 - kcnt)), clamp +/-7, negate =====
    function [3:0] q4(input [10:0] tf, input [2:0] kq);
        reg signed [10:0] t;
        reg signed [14:0] p, fl, r;
        reg [8:0] rem, half;
        reg [3:0] s;
        begin
            t    = tf;
            p    = ($signed({{4{t[10]}}, t}) <<< 3) - $signed({{4{t[10]}}, t});   // 7 T
            s    = 4'd9 - {1'b0, kq};                                              // 5..9
            fl   = p >>> s;
            rem  = 9'(p & ((15'sd1 <<< s) - 15'sd1));
            half = 9'(15'sd1 <<< (s - 4'd1));
            r    = fl + (((rem > half) || (rem == half && fl[0])) ? 15'sd1 : 15'sd0);
            if (r > 15'sd7)       q4 = 4'b1001;      // -(+7)
            else if (r < -15'sd7) q4 = 4'b0111;      // -(-7)
            else                  q4 = 4'(-r);
        end
    endfunction
    reg [23:0]        s4_llr;
    reg [3:0]         s4_n;
    reg               s4_v;
    reg [META_W-1:0]  s4_m;

    /* verilator lint_off WIDTHEXPAND */
    /* verilator lint_off WIDTHTRUNC */
    always @(posedge clk) begin
        if (rst) begin
            s1_v <= 0; s2_v <= 0; s3_v <= 0; s4_v <= 0; out_valid <= 0; llr <= 0; n_bits <= 0;
        end else begin
            // S1
            s1_ar   <= {mod_scheme, tidx(y_re, mod_scheme)};
            s1_ai   <= {mod_scheme, tidx(y_im, mod_scheme)};
            s1_half <= (mod_scheme == `DM_QAM64) ? 4'd3 : (mod_scheme == `DM_QAM16) ? 4'd2 : 4'd1;
            s1_k    <= kcnt;
            s1_bpsk <= (mod_scheme == `DM_BPSK);
            s1_bbit <= ~y_re[W-1];              // same decision as demapper.v
            s1_v    <= in_valid;
            s1_m    <= in_meta;
            // S2 (table read above)
            s2_half <= s1_half; s2_k <= s1_k; s2_bpsk <= s1_bpsk; s2_bbit <= s1_bbit;
            s2_v <= s1_v; s2_m <= s1_m;
            // S3: table output register
            s3_tr <= s2_tr; s3_ti <= s2_ti;
            s3_half <= s2_half; s3_k <= s2_k; s3_bpsk <= s2_bpsk; s3_bbit <= s2_bbit;
            s3_v <= s2_v; s3_m <= s2_m;
            // S4: weight shift, round, clamp, negate; pack in transmission
            // order: I bits (MSB first) then Q bits. Table column j = axis bit j.
            if (s3_bpsk) begin
                s4_llr <= {20'd0, s3_bbit ? 4'b1001 : 4'b0111};   // bit 1 -> -7, bit 0 -> +7
                s4_n   <= 4'd1;
            end else begin
                case (s3_half)
                    4'd3: s4_llr <= {q4(s3_ti[32:22], s3_k), q4(s3_ti[21:11], s3_k), q4(s3_ti[10:0], s3_k),
                                     q4(s3_tr[32:22], s3_k), q4(s3_tr[21:11], s3_k), q4(s3_tr[10:0], s3_k)};
                    4'd2: s4_llr <= {8'd0, q4(s3_ti[21:11], s3_k), q4(s3_ti[10:0], s3_k),
                                     q4(s3_tr[21:11], s3_k), q4(s3_tr[10:0], s3_k)};
                    default: s4_llr <= {16'd0, q4(s3_ti[10:0], s3_k), q4(s3_tr[10:0], s3_k)};
                endcase
                s4_n <= s3_half << 1;
            end
            s4_v <= s3_v; s4_m <= s3_m;
            // output register
            llr <= s4_llr; n_bits <= s4_n; out_valid <= s4_v; out_meta <= s4_m;
        end
    end
    /* verilator lint_on WIDTHTRUNC */
    /* verilator lint_on WIDTHEXPAND */
endmodule
