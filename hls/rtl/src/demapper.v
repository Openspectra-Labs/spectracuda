// ============================================================
// demapper.v -- hard-decision QAM demapper, runtime modulation
//
// Port of spectracuda's Modem.demodulate() hard decision
// (spectracuda/modem/mapper.py:212-220), per axis:
//     level = clip(round(x/norm + (m-1)) / 2), 0, m-1)
//     gray  = level ^ (level >> 1)
//     bits  = MSB-first unpack of gray, I first then Q
// with m = 2**(bits_per_symbol/2). The two constants fold into one
// multiply-add, both emitted by hls/gen/emit_rtl.py:
//     level = clip((x * SCALE + BIAS) >> DM_SHIFT, 0, MAXLVL)
// BIAS carries the (m-1)/2 offset AND the round-half-up constant.
//
// ALL THREE CONSTELLATIONS ARE BUILT, selected at run time by
// `mod_scheme`. This is not a convenience -- the modulation is runtime
// state from both directions:
//   * RX decodes mod_scheme from the frame header, so a receiver cannot
//     know it at synthesis time;
//   * TX changes it mid-session: Mac.set_tx_scheme() (mac.py:533) drives
//     Ofdm.reconfigure_tx_scheme() off measured PER/BER, and that
//     method's docstring is explicit that it mutates fields rather than
//     rebuilding the object -- which maps exactly onto a control
//     register changing while the circuits stay put.
// openofdm does the same (rate_to_idx.v -> demodulate.v holds BPSK/QPSK/
// 16QAM/64QAM concurrently); that is what lets openwifi run adaptive MCS.
// The area cost is every constellation, always, used or not.
//
// HARD DECISION, deliberately -- see plan doc section 7. openofdm emits
// soft decisions (BPSK_SOFT_0..4), worth ~2 dB of Viterbi coding gain,
// and spectracuda leaves that on the table today too
// (fec/_native.py:631 calls correct_convolutional_decode, the hard entry
// point, while libcorrect ships _decode_soft alongside it). Matching the
// golden model is this project's verification contract, so soft comes
// after Python has it -- not before, or there is nothing to check
// against.
//
// Rounding note: numpy's round() is half-to-EVEN, this is half-UP. They
// differ only for a symbol landing exactly on a decision boundary, which
// after a real channel is a measure-zero event; the test sweeps real
// frames and reports any mismatch rather than assuming.
// ============================================================
`timescale 1ns / 1ps
`include "generated/demap_params.vh"

module demapper #(
    parameter integer W = 18            // Q12 input, from mmse_eq
)(
    input  wire                clk,
    input  wire                rst,

    input  wire signed [W-1:0] y_re,
    input  wire signed [W-1:0] y_im,
    input  wire         [1:0]  mod_scheme,   // DM_QPSK / DM_QAM16 / DM_QAM64
    input  wire                in_valid,

    output reg          [5:0]  bits,         // MSB-first, I bits then Q
    output reg          [3:0]  n_bits,       // 2, 4 or 6
    output reg                 out_valid
);

    // ---- per-scheme constants, muxed ----
    reg signed [31:0] scale;
    reg signed [63:0] bias;
    reg        [3:0]  half;
    reg        [3:0]  maxlvl;
    always @* begin
        case (mod_scheme)
            `DM_QAM16: begin
                scale = `DM_QAM16_SCALE; bias = `DM_QAM16_BIAS;
                half = `DM_QAM16_HALF;   maxlvl = `DM_QAM16_MAXLVL;
            end
            `DM_QAM64: begin
                scale = `DM_QAM64_SCALE; bias = `DM_QAM64_BIAS;
                half = `DM_QAM64_HALF;   maxlvl = `DM_QAM64_MAXLVL;
            end
            default: begin
                scale = `DM_QPSK_SCALE;  bias = `DM_QPSK_BIAS;
                half = `DM_QPSK_HALF;    maxlvl = `DM_QPSK_MAXLVL;
            end
        endcase
    end

    // ---- d1: scale and bias ----
    reg signed [63:0] d1_re, d1_im;
    reg        [3:0]  d1_half, d1_max;
    reg               d1_v;

    // ---- d2: shift, clip, gray ----
    reg        [3:0]  d2_i, d2_q;
    reg        [3:0]  d2_half;
    reg               d2_v;

    wire signed [63:0] sh_re = d1_re >>> `DM_SHIFT;
    wire signed [63:0] sh_im = d1_im >>> `DM_SHIFT;
    // clip to [0, maxlvl].
    //
    // max_ext is sign-extended EXPLICITLY rather than letting the
    // comparison widen d1_max implicitly: mixing a signed operand with an
    // unsigned one makes the WHOLE expression unsigned in Verilog, so
    // `sh_re < 0` would stop working and every negative symbol would clip
    // to maxlvl instead of 0. Widening by hand keeps both sides signed.
    wire signed [63:0] max_ext = {{60{1'b0}}, d1_max};
    wire [3:0] lvl_i = (sh_re < 64'sd0) ? 4'd0
                     : (sh_re > max_ext) ? d1_max : sh_re[3:0];
    wire [3:0] lvl_q = (sh_im < 64'sd0) ? 4'd0
                     : (sh_im > max_ext) ? d1_max : sh_im[3:0];

    /* verilator lint_off WIDTHEXPAND */
    /* verilator lint_off WIDTHTRUNC */
    always @(posedge clk) begin
        if (rst) begin
            d1_v <= 1'b0; d2_v <= 1'b0; out_valid <= 1'b0;
            bits <= 6'd0; n_bits <= 4'd0;
        end else begin
            // d1
            d1_re   <= y_re * scale + bias;
            d1_im   <= y_im * scale + bias;
            d1_half <= half;
            d1_max  <= maxlvl;
            d1_v    <= in_valid;

            // d2: binary level -> Gray
            d2_i    <= lvl_i ^ (lvl_i >> 1);
            d2_q    <= lvl_q ^ (lvl_q >> 1);
            d2_half <= d1_half;
            d2_v    <= d1_v;

            // d3: pack MSB-first, I bits then Q, matching
            // mapper.py's concatenate([i_bits, q_bits]).
            case (d2_half)
                4'd2: bits <= {d2_i[1:0], d2_q[1:0], 2'b00};
                4'd3: bits <= {d2_i[2:0], d2_q[2:0]};
                default: bits <= {d2_i[0], d2_q[0], 4'b0000};
            endcase
            n_bits    <= d2_half << 1;
            out_valid <= d2_v;
        end
    end
    /* verilator lint_on WIDTHTRUNC */
    /* verilator lint_on WIDTHEXPAND */
endmodule
