// ============================================================
// demapper_soft.v -- 4-bit soft-decision (max-log LLR) QAM demapper
//
// Port of spectracuda Modem.demodulate_soft (modem/mapper.py and the
// per-axis numba kernel modem/_numba_mapper.py) in the profile the v3 RTL
// implements (hls/rtl/golden_ref_v3.py): llr_bits = 4, llr_clip = 3.0,
// soft_llr_scale = "stream" (pipeline/ofdm.py).
//
// MATH (Python, per coded bit b of a square Gray QAM, per AXIS -- the
// other axis cancels, see _numba_mapper.py _get_soft_kernel):
//     raw  = min_{label_b=0} (x - l)^2  -  min_{label_b=1} (x - l)^2
//     llr  = raw / 2 * w / clip,   w = |H[k]|^2 / (mean_train|H|^2 * noise_hdr)
//     q    = round(7 * clip(llr, -1, 1))            (4-bit mid-tread, 15 levels)
// With x = xn * norm and the levels l = L * norm, L odd integers, the two
// nearest levels L0 (bit 0) and L1 (bit 1) give
//     raw  = norm^2 * (L1 - L0) * (2 xn - L0 - L1)
// so
//     7 * llr = G_mod * t * W,   t = (L1-L0)(2 xn - L0 - L1),
//     G_mod = 7 norm^2 / (2 clip),   W = |H[k]|^2 * K_frame
// K_frame (= 1 / (mean_train|H|^2 * noise_hdr), with the fixed-point
// units folded in) is computed ONCE per frame by rx_freq_domain from the
// training estimate and the header symbols -- both known before the first
// payload symbol, which is the whole point of soft_llr_scale="stream".
// It arrives here as an 18-bit mantissa fr_km and a right shift fr_sh:
//     W (Q16) = (h2 * fr_km) >> fr_sh
//
// SIGN: Python's llr > 0 means bit 1. The RTL soft Viterbi
// (viterbi_dec_soft.v) takes "LLR > 0 means 0", so the output is -q. The
// hard decision is therefore the sign bit at every width, as before.
//
// NEAREST LEVELS without distances: the overall nearest level a* is the
// hard decision; for each bit, one of L0/L1 is a* itself and the other is
// the nearest level whose Gray bit differs -- either the closest such
// level below a* or above it, chosen by comparing xn with their midpoint.
// Ties are equidistant, so they give the same raw -- no tie rule needed.
//
// PRECISION vs Python's float32: identical quantities, fixed point with
// >= 17-bit mantissas, so a 4-bit level can differ by one only when Python's
// value sits within ~2^-12 (relative) of a rounding boundary. The test
// (run_frame_v3.py) measures that rate on real frames and requires the
// DECODED bytes to equal Python's soft decode.
//
// SATURATIONS (all far outside realistic operation, documented so they
// are not mistaken for the math): t to 25 bits (|t| < 4096 levels^2),
// W to 34 bits Q16 (W < 2^18, i.e. ~54 dB post-equalization SNR).
//
// BPSK (the header) is HARD here: llr = +/-7 from the sign, exactly what
// the hard demapper decided -- the header decoder (header_decode_v3.v) is
// a hard-decision Viterbi, matching Python's header path.
// Latency: 6 clocks, fixed, for every item (header and data alike, so the
// stream stays in order).
// ============================================================
`timescale 1ns / 1ps
`include "generated/demap_params.vh"
module demapper_soft #(
    parameter integer W      = 18,      // y, Q12, from mmse_eq / pilot_cpe
    parameter integer META_W = 1
)(
    input  wire                clk,
    input  wire                rst,
    input  wire signed [W-1:0] y_re,
    input  wire signed [W-1:0] y_im,
    input  wire        [2*W-1:0] h2,     // |H[k]|^2, Q24 (mmse_eq y_hh)
    input  wire         [1:0]  mod_scheme,   // DM_QPSK / DM_QAM16 / DM_QAM64 / DM_BPSK
    input  wire                in_valid,
    input  wire  [META_W-1:0]  in_meta,
    // per-frame LLR scale (rx_freq_domain), constant across the frame's data
    input  wire        [17:0]  fr_km,
    input  wire signed [7:0]   fr_sh,
    output reg         [23:0]  llr,      // 6 x 4-bit signed, llr[k] at [4k +: 4], k = 0 first bit
    output reg          [3:0]  n_bits,   // 1 (BPSK), 2, 4 or 6
    output reg                 out_valid,
    output reg   [META_W-1:0]  out_meta
);
    // ---- G_mod = 7 * norm^2 / (2 * clip), Q18, clip = 3.0 ----
    //   QPSK  norm^2 = 1/2  -> 7/12  = 0.583333 -> 152917
    //   16QAM norm^2 = 1/10 -> 7/60  = 0.116667 ->  30583
    //   64QAM norm^2 = 1/42 -> 7/252 = 0.027778 ->   7282
    // (llr_clip lives in golden_ref_v3.py; change both together.)
    localparam [17:0] G_QPSK = 18'd152917, G_QAM16 = 18'd30583, G_QAM64 = 18'd7282;

    function [2:0] gray3(input [2:0] a); gray3 = a ^ (a >> 1); endfunction

    // ===== S1: normalized axis values, |H|^2 * K mantissa =====
    reg signed [31:0] scale;
    reg        [3:0]  half;        // bits per axis
    always @* begin
        case (mod_scheme)
            `DM_QAM16: begin scale = `DM_QAM16_SCALE; half = 4'd2; end
            `DM_QAM64: begin scale = `DM_QAM64_SCALE; half = 4'd3; end
            default:   begin scale = `DM_QPSK_SCALE;  half = 4'd1; end
        endcase
    end
    reg signed [21:0] s1_xr, s1_xi;      // xn = x / norm, Q12 (levels are odd integers)
    reg        [53:0] s1_p1;             // h2 * fr_km
    reg        [3:0]  s1_half;
    reg        [1:0]  s1_mod;
    reg               s1_bpsk, s1_bbit, s1_v;
    reg  [META_W-1:0] s1_m;
    // y * SCALE is (x / norm) / 2 in Q26 (demapper.v); >>> 13 gives x / norm in Q12
    wire signed [49:0] pr = y_re * scale;
    wire signed [49:0] pi = y_im * scale;

    // ===== S2: W (Q16), nearest overall level per axis =====
    reg        [33:0] s2_w;
    reg signed [21:0] s2_xr, s2_xi;
    reg        [2:0]  s2_ar, s2_ai;      // hard levels a* (0 = most negative)
    reg        [3:0]  s2_half;
    reg        [1:0]  s2_mod;
    reg               s2_bpsk, s2_bbit, s2_v;
    reg  [META_W-1:0] s2_m;
    // W = P1 >> sh, saturated to 34 bits; a negative shift is a left shift
    function [33:0] wsat(input [53:0] p, input signed [7:0] sh);
        reg [53:0] q; reg [7:0] ls;
        begin
            if (sh >= 0) begin
                q = (sh >= 54) ? 54'd0 : (p >> sh);
                wsat = (|q[53:34]) ? {34{1'b1}} : q[33:0];
            end else begin
                ls = -sh;
                // left shift: saturate unless the top ls bits are all zero
                if (ls >= 34 || (p >> (34 - ls)) != 0) wsat = (p == 0) ? 34'd0 : {34{1'b1}};
                else wsat = p[33:0] << ls;
            end
        end
    endfunction
    // a* = clip(floor((xn + m) / 2), 0, m - 1), m = 2^half levels per axis
    function [2:0] alevel(input signed [21:0] xn, input [3:0] hb);
        reg signed [23:0] z; reg [3:0] mlev;
        begin
            mlev = 4'd1 << hb;
            z = ($signed({{2{xn[21]}}, xn}) + $signed({8'd0, mlev, 12'd0})) >>> 13;
            if (z < 0) alevel = 3'd0;
            else if (z > $signed({20'd0, mlev - 4'd1})) alevel = 3'(mlev - 4'd1);
            else alevel = z[2:0];
        end
    endfunction

    // ===== S3: per bit, the two nearest levels L0 / L1 and t =====
    // For bit j (0 = MSB of the axis label), level a* has Gray bit b; the
    // nearest level with bit !b is the closest one below or above a*.
    // Returns {t} for one axis bit: t = (L1-L0)(2 xn - L0 - L1), Q12, 25-bit sat.
    function signed [24:0] tbit(input signed [21:0] xn, input [2:0] as, input [3:0] hb,
                                input [1:0] j);
        integer a, lo, hi;
        reg [2:0] g; reg b, bo;
        reg signed [4:0] Ls, Lo, L0, L1, mid;
        reg signed [27:0] u;
        reg signed [32:0] t;          // |u| < 2^27, |L1-L0| <= 14: 33 bits, then saturate
        reg signed [4:0]  dl;
        reg [3:0] mlev;
        begin
            mlev = 4'd1 << hb;
            g = gray3(as);
            b = g[hb - 1 - j];
            lo = -1; hi = -1;
            for (a = 0; a < 8; a = a + 1) begin
                if (a < mlev) begin
                    g = gray3(a[2:0]); bo = g[hb - 1 - j];
                    if (bo != b && a < as) lo = a;                     // closest below (last one)
                    if (bo != b && a > as && hi < 0) hi = a;           // closest above (first one)
                end
            end
            Ls = $signed({2'd0, as, 1'b0}) - $signed({1'b0, mlev}) + 5'sd1;   // 2a - (m-1)
            if (lo < 0)      Lo = 5'(2*hi) - $signed({1'b0, mlev}) + 5'sd1;
            else if (hi < 0) Lo = 5'(2*lo) - $signed({1'b0, mlev}) + 5'sd1;
            else begin
                // midpoint of L_lo and L_hi = (L_lo + L_hi) / 2 = lo + hi - (m-1)
                mid = 5'(lo + hi) - $signed({1'b0, mlev}) + 5'sd1;
                if ($signed({{2{xn[21]}}, xn}) < $signed({{7{mid[4]}}, mid, 12'd0}))
                     Lo = 5'(2*lo) - $signed({1'b0, mlev}) + 5'sd1;
                else Lo = 5'(2*hi) - $signed({1'b0, mlev}) + 5'sd1;
            end
            if (b) begin L1 = Ls; L0 = Lo; end else begin L0 = Ls; L1 = Lo; end
            u = $signed({{5{xn[21]}}, xn, 1'b0}) - $signed({{11{L0[4]}}, L0, 12'd0})
                                               - $signed({{11{L1[4]}}, L1, 12'd0});
            dl = L1 - L0;
            t = $signed({{5{u[27]}}, u}) * $signed({{28{dl[4]}}, dl});
            tbit = (t > 33'sd16777215) ? 25'sd16777215 : (t < -33'sd16777215) ? -25'sd16777215 : t[24:0];
        end
    endfunction

    reg signed [24:0] s3_t [0:5];        // lane k = axis bit (I bits then Q bits)
    reg        [35:0] s3_gw;             // G_mod * W, Q18
    reg        [3:0]  s3_half;
    reg               s3_bpsk, s3_bbit, s3_v;
    reg  [META_W-1:0] s3_m;
    wire       [17:0] gmod = (s2_mod == `DM_QAM16) ? G_QAM16 : (s2_mod == `DM_QAM64) ? G_QAM64 : G_QPSK;
    wire       [51:0] gw_full = s2_w * gmod;

    // ===== S4: lane products v = t * GW (Q30) =====
    reg signed [61:0] s4_p [0:5];
    reg        [3:0]  s4_half;
    reg               s4_bpsk, s4_bbit, s4_v;
    reg  [META_W-1:0] s4_m;

    // ===== S5: round, clamp to +/-7, negate (RTL sign convention) =====
    function [3:0] q4(input signed [61:0] v);
        reg signed [61:0] r;
        begin
            r = (v + 62'sd536870912) >>> 30;          // round half up, Q30 -> integer
            if (r > 62'sd7)       q4 = 4'b1001;      // -(+7)
            else if (r < -62'sd7) q4 = 4'b0111;      // -(-7)
            else                  q4 = 4'(-r);
        end
    endfunction
    reg [23:0]        s5_llr;
    reg [3:0]         s5_n;
    reg               s5_v;
    reg [META_W-1:0]  s5_m;

    integer k;
    /* verilator lint_off WIDTHEXPAND */
    /* verilator lint_off WIDTHTRUNC */
    always @(posedge clk) begin
        if (rst) begin
            s1_v <= 0; s2_v <= 0; s3_v <= 0; s4_v <= 0; s5_v <= 0; out_valid <= 0;
            llr <= 0; n_bits <= 0;
        end else begin
            // S1
            s1_xr   <= pr >>> 13;
            s1_xi   <= pi >>> 13;
            s1_p1   <= h2 * fr_km;
            s1_half <= half;
            s1_mod  <= mod_scheme;
            s1_bpsk <= (mod_scheme == `DM_BPSK);
            s1_bbit <= ~y_re[W-1];              // same decision as demapper.v
            s1_v    <= in_valid;
            s1_m    <= in_meta;
            // S2
            s2_w    <= wsat(s1_p1, fr_sh);
            s2_xr   <= s1_xr;  s2_xi <= s1_xi;
            s2_ar   <= alevel(s1_xr, s1_half);
            s2_ai   <= alevel(s1_xi, s1_half);
            s2_half <= s1_half; s2_mod <= s1_mod;
            s2_bpsk <= s1_bpsk; s2_bbit <= s1_bbit; s2_v <= s1_v; s2_m <= s1_m;
            // S3
            for (k = 0; k < 3; k = k + 1) begin
                s3_t[k]     <= (k < s2_half) ? tbit(s2_xr, s2_ar, s2_half, k[1:0]) : 25'sd0;
                s3_t[3 + k] <= (k < s2_half) ? tbit(s2_xi, s2_ai, s2_half, k[1:0]) : 25'sd0;
            end
            s3_gw   <= gw_full >> 16;
            s3_half <= s2_half; s3_bpsk <= s2_bpsk; s3_bbit <= s2_bbit; s3_v <= s2_v; s3_m <= s2_m;
            // S4
            for (k = 0; k < 6; k = k + 1)
                s4_p[k] <= s3_t[k] * $signed({1'b0, s3_gw});
            s4_half <= s3_half; s4_bpsk <= s3_bpsk; s4_bbit <= s3_bbit; s4_v <= s3_v; s4_m <= s3_m;
            // S5: lanes in transmission order: I bits (MSB first) then Q bits
            if (s4_bpsk) begin
                s5_llr <= {20'd0, s4_bbit ? 4'b1001 : 4'b0111};   // bit 1 -> -7, bit 0 -> +7
                s5_n   <= 4'd1;
            end else begin
                s5_llr <= 24'd0;
                case (s4_half)
                    4'd3: s5_llr <= {q4(s4_p[5]), q4(s4_p[4]), q4(s4_p[3]),
                                     q4(s4_p[2]), q4(s4_p[1]), q4(s4_p[0])};
                    4'd2: s5_llr <= {8'd0, q4(s4_p[4]), q4(s4_p[3]), q4(s4_p[1]), q4(s4_p[0])};
                    default: s5_llr <= {16'd0, q4(s4_p[3]), q4(s4_p[0])};
                endcase
                s5_n <= s4_half << 1;
            end
            s5_v <= s4_v; s5_m <= s4_m;
            // S6: output register
            llr <= s5_llr; n_bits <= s5_n; out_valid <= s5_v; out_meta <= s5_m;
        end
    end
    /* verilator lint_on WIDTHTRUNC */
    /* verilator lint_on WIDTHEXPAND */
endmodule
