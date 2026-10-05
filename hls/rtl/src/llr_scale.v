// ============================================================
// llr_scale.v -- per-frame soft-LLR scale (soft_llr_scale = "stream")
//
// spectracuda Ofdm, soft_llr_scale="stream" (pipeline/ofdm.py), weights
// every payload LLR by
//     w = |H[k]|^2 / (mean_train|H|^2 * noise_hdr)
//     mean_train|H|^2 = mean over the TRAINING estimate's data subcarriers
//     noise_hdr       = mean over the 2 header symbols' equalized values of
//                       the squared distance to the nearest BPSK point (+/-1)
// Both are known before the first payload symbol, which is what lets a
// streaming receiver use them. This block measures both and produces the
// frame constant the soft demapper multiplies |H[k]|^2 by.
//
// FIXED POINT. H and y are Q12 (mmse_eq contract), so per subcarrier
// |H|^2 and the noise terms are Q24 integers. With sH = sum over the 216
// training data bins of |H|^2 and sN = sum over the 432 header items of
// (|y_re| - 1)^2 + y_im^2 (both Q24 integers):
//     w = h2 * 216 * 432 * 2^24 / (sH * sN) = h2 * K,
//     K = 1458 * 2^30 / (sH * sN)                    (216 * 432 = 1458 * 64)
// sH and sN are normalized to 18-bit mantissas mH, mN (sH = mH * 2^eH),
// and one serial division per frame gives
//     Kc = floor(1458 * 2^55 / (mH * mN))            (32 bits)
//     Kc ~= km * 2^ke                                 (km: 18-bit mantissa)
//     W (Q16) = h2 * K * 2^16 = (h2 * km) >> (9 + eH + eN - ke) = (h2 * km) >> fr_sh
// demapper_soft.v applies that shift.
//
// TIMING OF THE UPDATE. fr_km / fr_sh change only after the frame's LAST
// header item has entered the demapper (header items reach it after every
// earlier data item has left CPE -- rx_freq_domain's header queue rule),
// plus the ~70-clock division. The previous frame's data are past the
// demapper's first stage by then; the new frame's data are released from
// B1 only after BD has decoded this header, long after. The training sum
// is staged (sH_next) because a frame's TRAIN estimate completes before
// its header is equalized.
// ============================================================
`timescale 1ns / 1ps
module llr_scale #(
    parameter integer W = 18                // H and y, Q12
)(
    input  wire                clk,
    input  wire                rst,
    // training estimate, data subcarriers only
    input  wire                h_valid,
    input  wire signed [W-1:0] h_re,
    input  wire signed [W-1:0] h_im,
    input  wire                h_done,      // last bin of the TRAINING estimate written
    // equalized header items, as they enter the demapper
    input  wire                n_valid,
    input  wire signed [W-1:0] n_re,
    input  wire signed [W-1:0] n_im,
    input  wire                n_last,      // the frame's last header item
    // frame scale for demapper_soft
    output reg         [17:0]  fr_km,
    output reg  signed [7:0]   fr_sh,
    output reg                 fr_busy      // division in progress (status / debug)
);
    localparam [W-1:0] ONE_Q12 = 4096;

    // ---- sH: sum of |H|^2 over the training estimate ----
    reg [2*W-1:0] hh;   reg hh_v;
    reg [44:0]    sh_acc, sh_next;
    reg [2:0]     hdone_d;              // h_done delayed past the 2-stage accumulate
    always @(posedge clk) begin
        if (rst) begin
            hh_v <= 1'b0; sh_acc <= 45'd0; sh_next <= 45'd1; hdone_d <= 3'd0;
        end else begin
            hh   <= h_re * h_re + h_im * h_im;
            hh_v <= h_valid;
            if (hh_v) sh_acc <= sh_acc + {9'd0, hh};
            hdone_d <= {hdone_d[1:0], h_done};
            if (hdone_d[2]) begin
                sh_next <= (sh_acc == 45'd0) ? 45'd1 : sh_acc;   // guard a dead estimate
                sh_acc  <= 45'd0;
            end
        end
    end

    // ---- sN: header noise, sum of (|y_re| - 1)^2 + y_im^2 ----
    wire signed [W:0] ar   = n_re[W-1] ? -$signed({n_re[W-1], n_re}) : $signed({n_re[W-1], n_re});
    wire signed [W:0] dre  = ar - $signed({1'b0, ONE_Q12});
    reg  [2*W+1:0]    ee;   reg ee_v, ee_last;
    reg  [45:0]       sn_acc;
    reg               go;                    // start the division (sums final)
    reg  [44:0]       sh_use;
    reg  [45:0]       sn_use;
    always @(posedge clk) begin
        if (rst) begin
            ee_v <= 1'b0; ee_last <= 1'b0; sn_acc <= 46'd0; go <= 1'b0;
        end else begin
            ee      <= dre * dre + n_im * n_im;
            ee_v    <= n_valid;
            ee_last <= n_valid && n_last;
            go      <= 1'b0;
            if (ee_v) sn_acc <= ee_last ? 46'd0 : sn_acc + {8'd0, ee};
            if (ee_v && ee_last) begin
                sn_use <= ((sn_acc + {8'd0, ee}) == 46'd0) ? 46'd1 : sn_acc + {8'd0, ee};
                sh_use <= sh_next;
                go     <= 1'b1;
            end
        end
    end

    // ---- normalize, divide, renormalize (sequential, once per frame) ----
    function [5:0] msb46(input [45:0] x);
        integer i;
        begin msb46 = 0; for (i = 0; i < 46; i = i + 1) if (x[i]) msb46 = i[5:0]; end
    endfunction
    localparam S_IDLE = 0, S_NORM = 1, S_MUL = 2, S_DIV = 3, S_FIN = 4;
    reg  [2:0]  st;
    reg  [17:0] mH, mN;
    reg  signed [7:0] eH, eN;
    reg  [35:0] dd;                         // mH * mN, in [2^34, 2^36)
    reg  [66:0] num;                        // 1458 * 2^55, shifted out MSB first
    reg  [36:0] rem;
    reg  [31:0] kc;
    reg  [6:0]  it;
    wire [5:0]  bh = msb46({1'b0, sh_use});
    wire [5:0]  bn = msb46(sn_use);
    wire [37:0] rem_sh = {rem, num[66]};
    wire [5:0]  bk = msb46({14'd0, kc});
    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; fr_km <= 18'd1; fr_sh <= 8'sd0; fr_busy <= 1'b0;
        end else case (st)
            S_IDLE: if (go) begin st <= S_NORM; fr_busy <= 1'b1; end
            S_NORM: begin
                // x = m * 2^e with m in [2^17, 2^18): shift right if wider, left if narrower
                mH <= (bh >= 17) ? 18'(sh_use >> (bh - 17)) : 18'(sh_use << (17 - bh));
                eH <= $signed({2'd0, bh}) - 8'sd17;
                mN <= (bn >= 17) ? 18'(sn_use >> (bn - 17)) : 18'(sn_use << (17 - bn));
                eN <= $signed({2'd0, bn}) - 8'sd17;
                st <= S_MUL;
            end
            S_MUL: begin
                dd  <= mH * mN;
                num <= 67'd1458 << 55;
                rem <= 37'd0; kc <= 32'd0; it <= 7'd0;
                st  <= S_DIV;
            end
            S_DIV: begin
                // restoring division, one numerator bit per clock (67 clocks)
                if (rem_sh >= {2'd0, dd}) begin
                    rem <= 37'(rem_sh - {2'd0, dd}); kc <= {kc[30:0], 1'b1};
                end else begin
                    rem <= rem_sh[36:0];             kc <= {kc[30:0], 1'b0};
                end
                num <= {num[65:0], 1'b0};
                if (it == 7'd66) st <= S_FIN;
                it <= it + 1'b1;
            end
            S_FIN: begin
`ifdef LLR_SCALE_DEBUG
                $display("LLR_SCALE sH=%0d sN=%0d meanH2_q24=%0f noise=%0f km=%0d sh=%0d",
                         sh_use, sn_use, $itor(sh_use) / 216.0 / 16777216.0,
                         $itor(sn_use) / 432.0 / 16777216.0, 18'(kc >> (bk - 6'd17)),
                         8'sd9 + eH + eN - ($signed({2'd0, bk}) - 8'sd17));
`endif
                // Kc ~= km * 2^ke, km 18 bits;  fr_sh = 9 + eH + eN - ke
                fr_km   <= 18'(kc >> (bk - 6'd17));
                fr_sh   <= 8'sd9 + eH + eN - ($signed({2'd0, bk}) - 8'sd17);
                fr_busy <= 1'b0;
                st      <= S_IDLE;
            end
            default: st <= S_IDLE;
        endcase
    end
endmodule
