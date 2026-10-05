// ============================================================
// sc_sync_rtl.v -- Schmidl & Cox preamble metric, streaming, pipelined
//
// Our own Verilog. Structure follows openofdm's sync_short.v
// (reference/openofdm/verilog/sync_short.v): delay -> conjugate multiply
// -> accumulate the product -> accumulate the energy -> compare. What
// differs is the lag and window: openofdm is 802.11, whose short preamble
// repeats every 16 samples, so it instantiates `fifo_sample_delay` with
// `.delay_ctl(16)` and averages over WINDOW_SHIFT=4. Schmidl-Cox on
// spectracuda's waveform correlates at lag L = fft_size/2 = 128 and sums
// over the same 128.
//
//   P(d) = sum_{m<L} conj(r[d+m]) * r[d+m+L]
//   R(d) = ( sum_{m<L} |r[d+m]|^2 + sum_{m<L} |r[d+m+L]|^2 ) / 2
//
// Algorithm is the sliding-window form from
// spectracuda/sync/_numba_schmidl_cox.py: each accumulator is an O(1)
// update per sample, so this is one adder per accumulator rather than a
// 128-tap sum.
//
// THE PRIMING PHASE IS NOT OPTIONAL. `acc += entering - leaving` only
// tracks the window sum if the accumulator already HOLDS that sum. An
// earlier version started every accumulator at zero and went straight to
// sliding, so it computed P(d) - P(0) and reported a peak metric of 1.64
// where the true value is bounded by 1.0. The window is built explicitly
// over the first 2L samples before any sliding begins, exactly as the
// numba reference computes d=0 directly and only then enters its loop.
//
//   n < L        : r1 += |r[n]|^2                          (phase A)
//   L <= n < 2L  : P  += conj(r[n-L])*r[n] ; r2 += |r[n]|^2 (phase B)
//   n == 2L-1    : emit d = 0
//   n >= 2L      : slide, emit d = n - 2L + 1              (phase C)
//
// PIPELINE. The combinational version was 1084 LUT but missed 100 MHz by
// 8.1 ns: RAM read -> four 16x16 multiplies -> sum -> 48-bit accumulate
// is far too much for one cycle. Split four ways, so the only thing left
// in the feedback loop is a single add:
//
//   S0  write sample, register the three RAM taps and the phase flags
//   S1  multiply (DSP48 input+output registers)
//   S2  combine products into per-accumulator deltas
//   S3  accumulate; this stage alone carries the loop
//
// The per-phase behaviour is folded into the S2 deltas rather than into
// three different accumulate statements. That keeps S3 a single uniform
// adder per accumulator -- which is what makes the loop short enough to
// close -- and costs only the mux at S2, which is off the critical path.
//
// Output latency is 3 cycles behind the sample that produced it. Order
// and count are unchanged, so a consumer counting out_valid pulses still
// numbers candidates exactly as Python numbers start_index.
//
// Emits P and R, NOT the metric |P|^2/R^2. A divider is expensive
// (measured: 4456 LUT, 55% of the HLS build of this same block) and every
// consumer compares the metric against something -- `|P|^2 > T * R^2` is
// a multiply. Peak search and threshold both live downstream.
// ============================================================
`timescale 1ns / 1ps

module sc_sync_rtl #(
    parameter integer SAMPLE_W = 16,   // I and Q width, hardware native
    parameter integer LAG      = 128,  // L = fft_size/2
    parameter integer ACC_W    = 48    // running-sum width, see below
)(
    input  wire                       clk,
    input  wire                       rst,

    input  wire signed [SAMPLE_W-1:0] in_i,
    input  wire signed [SAMPLE_W-1:0] in_q,
    input  wire                       in_valid,

    output reg  signed [ACC_W-1:0]    p_re,
    output reg  signed [ACC_W-1:0]    p_im,
    output reg  signed [ACC_W-1:0]    r_sum,   // r1 + r2; halving is free downstream
    output reg                        out_valid
);

    localparam integer SPAN   = 2*LAG + 1;
    localparam integer ADDR_W = $clog2(SPAN);
    localparam integer CNT_W  = 32;
    localparam integer PROD_W = 2*SAMPLE_W + 1;

    reg signed [SAMPLE_W-1:0] buf_i [0:SPAN-1];
    reg signed [SAMPLE_W-1:0] buf_q [0:SPAN-1];

    reg [ADDR_W-1:0] wp;        // write pointer -> r[n]
    reg [ADDR_W-1:0] rp_mid;    // wp - LAG      -> r[n-L]
    reg [ADDR_W-1:0] rp_out;    // wp - 2*LAG    -> r[n-2L]
    reg [CNT_W-1:0]  n;

    /* verilator lint_off WIDTHTRUNC */
    localparam [ADDR_W-1:0] SPAN_M1   = SPAN - 1;
    localparam [ADDR_W-1:0] ZERO_ADDR = {ADDR_W{1'b0}};
    localparam [ADDR_W-1:0] INIT_MID  = SPAN - LAG;
    localparam [ADDR_W-1:0] INIT_OUT  = SPAN - 2*LAG;
    /* verilator lint_on WIDTHTRUNC */

    function [ADDR_W-1:0] wrap_inc(input [ADDR_W-1:0] a);
        wrap_inc = (a == SPAN_M1) ? ZERO_ADDR : a + 1'b1;
    endfunction

    // ---------------- S0: fetch ----------------
    // rp_out is wp+1 and rp_mid is wp+L+1 modulo SPAN=2L+1, so neither
    // read address ever collides with the write address; the taps are
    // safe to register alongside the write.
    reg signed [SAMPLE_W-1:0] s0_m_i, s0_m_q, s0_o_i, s0_o_q, s0_n_i, s0_n_q;
    reg s0_ph_b, s0_ph_c, s0_emit, s0_v;

    // ---------------- S1: multiply ----------------
    reg signed [PROD_W-1:0] s1_ain_re, s1_ain_im, s1_aout_re, s1_aout_im;
    reg signed [PROD_W-1:0] s1_e_out, s1_e_mid, s1_e_in;
    reg s1_ph_b, s1_ph_c, s1_emit, s1_v;

    // ---------------- S2: deltas ----------------
    reg signed [ACC_W-1:0] s2_d_p_re, s2_d_p_im, s2_d_r1, s2_d_r2;
    reg s2_emit_or_slide, s2_v;

    // ---------------- S3: accumulate ----------------
    // ACC_W: each term is bounded by 2*(2^15)^2 = 2^31, summed over
    // LAG=128 -> 2^38. 48 bits leaves ten bits of margin and costs only
    // flip-flops, of which this part has 65200.
    reg signed [ACC_W-1:0] acc_p_re, acc_p_im, acc_r1, acc_r2;

    /* verilator lint_off WIDTHEXPAND */
    always @(posedge clk) begin
        if (rst) begin
            wp     <= ZERO_ADDR;
            rp_mid <= INIT_MID;
            rp_out <= INIT_OUT;
            n      <= {CNT_W{1'b0}};
            s0_v <= 1'b0; s1_v <= 1'b0; s2_v <= 1'b0;
            s0_emit <= 1'b0; s1_emit <= 1'b0; s2_emit_or_slide <= 1'b0;
            s0_ph_b <= 1'b0; s0_ph_c <= 1'b0;
            s1_ph_b <= 1'b0; s1_ph_c <= 1'b0;
            acc_p_re <= 0; acc_p_im <= 0; acc_r1 <= 0; acc_r2 <= 0;
            p_re <= 0; p_im <= 0; r_sum <= 0;
            out_valid <= 1'b0;
        end else begin
            // ---- S0 ----
            s0_v <= in_valid;
            if (in_valid) begin
                buf_i[wp] <= in_i;
                buf_q[wp] <= in_q;

                s0_m_i <= buf_i[rp_mid]; s0_m_q <= buf_q[rp_mid];
                s0_o_i <= buf_i[rp_out]; s0_o_q <= buf_q[rp_out];
                s0_n_i <= in_i;          s0_n_q <= in_q;

                s0_ph_b <= (n >= LAG) && (n < 2*LAG);
                s0_ph_c <= (n >= 2*LAG);
                s0_emit <= (n == 2*LAG - 1);

                wp     <= wrap_inc(wp);
                rp_mid <= wrap_inc(rp_mid);
                rp_out <= wrap_inc(rp_out);
                n      <= n + 1'b1;
            end

            // ---- S1: seven 16x16 products, one DSP48E1 each ----
            s1_v    <= s0_v;
            s1_ph_b <= s0_ph_b;
            s1_ph_c <= s0_ph_c;
            s1_emit <= s0_emit;
            s1_ain_re  <= s0_m_i*s0_n_i + s0_m_q*s0_n_q;
            s1_ain_im  <= s0_m_i*s0_n_q - s0_m_q*s0_n_i;
            s1_aout_re <= s0_o_i*s0_m_i + s0_o_q*s0_m_q;
            s1_aout_im <= s0_o_i*s0_m_q - s0_o_q*s0_m_i;
            s1_e_out   <= s0_o_i*s0_o_i + s0_o_q*s0_o_q;
            s1_e_mid   <= s0_m_i*s0_m_i + s0_m_q*s0_m_q;
            s1_e_in    <= s0_n_i*s0_n_i + s0_n_q*s0_n_q;

            // ---- S2: per-phase deltas, so S3 is one uniform adder ----
            s2_v <= s1_v;
            s2_emit_or_slide <= s1_emit | s1_ph_c;
            if (s1_ph_c) begin
                s2_d_p_re <= s1_ain_re - s1_aout_re;
                s2_d_p_im <= s1_ain_im - s1_aout_im;
                s2_d_r1   <= s1_e_mid  - s1_e_out;
                s2_d_r2   <= s1_e_in   - s1_e_mid;
            end else if (s1_ph_b) begin
                s2_d_p_re <= s1_ain_re;
                s2_d_p_im <= s1_ain_im;
                s2_d_r1   <= {ACC_W{1'b0}};
                s2_d_r2   <= s1_e_in;
            end else begin                      // phase A
                s2_d_p_re <= {ACC_W{1'b0}};
                s2_d_p_im <= {ACC_W{1'b0}};
                s2_d_r1   <= s1_e_in;
                s2_d_r2   <= {ACC_W{1'b0}};
            end

            // ---- S3: the only stage inside the feedback loop ----
            out_valid <= 1'b0;
            if (s2_v) begin
                acc_p_re <= acc_p_re + s2_d_p_re;
                acc_p_im <= acc_p_im + s2_d_p_im;
                acc_r1   <= acc_r1   + s2_d_r1;
                acc_r2   <= acc_r2   + s2_d_r2;

                if (s2_emit_or_slide) begin
                    p_re  <= acc_p_re + s2_d_p_re;
                    p_im  <= acc_p_im + s2_d_p_im;
                    r_sum <= (acc_r1 + s2_d_r1) + (acc_r2 + s2_d_r2);
                    out_valid <= 1'b1;
                end
            end
        end
    end
    /* verilator lint_on WIDTHEXPAND */
endmodule
