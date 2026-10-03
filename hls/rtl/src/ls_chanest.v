// ============================================================
// ls_chanest.v -- least-squares channel estimate + interpolation
//
// Port of spectracuda's LSChannelEstimator (spectracuda/channel/ls.py):
//     h_pilot = rx_pilots / tx_pilots
//     h_full  = interp(all_bins, pilot_indices, h_pilot)
//
// BOTH DIVISIONS ARE GONE, and neither by approximation -- the generator
// evaluated them at generate time because both denominators are known
// constants.
//
//   * tx_pilots are fixed by the waveform, so 1/tx_pilot is a constant.
//     hls/gen/emit_rtl.py emits it as a Q15 ROM (pilot_recip.mem). For
//     this configuration every pilot is exactly +1, making the reciprocal
//     +1 and the "division" a pass-through -- but the ROM is emitted
//     regardless so a different pilot set needs no RTL change.
//   * np.interp's per-bin (k - p_left)/(p_right - p_left) is likewise
//     known, so it is baked into a per-bin weight (chanest_interp.mem)
//     and the hardware does one multiply-accumulate per bin:
//         h[k] = h_left + (h_right - h_left) * w[k]
//     The pilot gaps here are 32,32,32,62,32,32,32 -- mostly powers of
//     two, but the 62 spanning the guard band is not, which is why this
//     is a generated table rather than a shift.
//
// That matters for area: openofdm's equalizer spends 1,784 LUT on a
// single div_gen instance. Divisions whose denominator is a waveform
// constant should never reach the fabric at all.
//
// Edge bins outside the pilot span are CLAMPED, matching np.interp
// (which holds the end value rather than extrapolating). Those ROM
// entries set left == right, so the interpolation term is exactly zero
// and no special case is needed here.
//
// Sequencing: 8 pilots then 256 bins, once per training symbol. At
// 10 Msps a symbol is 2,880 clocks, so one multiplier reused across all
// 256 bins is ample -- there is no reason to parallelise this.
// ============================================================
`timescale 1ns / 1ps
`include "generated/chanest_params.vh"

module ls_chanest #(
    parameter integer IN_W    = 20,   // per component, from the FFT
    parameter integer H_W     = 18,   // channel estimate width
    parameter integer N_PILOT = `CE_N_PILOT,
    parameter integer N_FFT   = `CE_N_FFT,
    parameter integer SLOT_W  = `CE_SLOT_W
)(
    input  wire                     clk,
    input  wire                     rst,

    // The N_PILOT received pilot bins, in ascending subcarrier order.
    input  wire signed [IN_W-1:0]   pilot_re,
    input  wire signed [IN_W-1:0]   pilot_im,
    input  wire                     pilot_valid,

    // Channel estimate for every subcarrier, bin 0 first.
    output reg  signed [H_W-1:0]    h_re,
    output reg  signed [H_W-1:0]    h_im,
    output reg                      h_valid,
    output reg                      h_last,
    // Natural-order bin of h_re/h_im. The sweep's own counter, carried
    // through the same pipeline as the sample, so no consumer has to
    // count h_valid pulses to find out which bin this is.
    output reg  [7:0]               h_bin
);

    localparam integer PIDX_W = (N_PILOT <= 1) ? 1 : $clog2(N_PILOT);
    localparam integer BIN_W  = $clog2(N_FFT);

    // ---- ROMs, both generated ----
    reg [31:0] interp_rom [0:N_FFT-1];   // {left, right, weight Q15}
    reg [31:0] recip_rom  [0:N_PILOT-1]; // {re Q15, im Q15}
    initial begin
        $readmemh(`CE_INTERP_MEM, interp_rom);
        $readmemh(`CE_RECIP_MEM,  recip_rom);
    end

    // ---- stage 1: h_pilot = rx * (1/tx) ----
    reg signed [H_W-1:0] hp_re [0:N_PILOT-1];
    reg signed [H_W-1:0] hp_im [0:N_PILOT-1];
    reg [PIDX_W:0]       p_cnt;
    reg                  have_pilots;

    // The reference-scaling multiply is PIPELINED. Timing analysis put
    // the critical path here, not in the interpolation where it was
    // assumed to be: reciprocal-ROM read -> two cascaded DSP48E1 -> pilot
    // BRAM write, 10.3 ns in a single cycle (-0.654 ns of slack). Reading
    // the actual report beat two rounds of guessing at the interpolator.
    //
    //   qa : ROM output and the sample, registered
    //   qb : the four products, registered (DSP48 gets registered inputs)
    //   qc : combine and write
    wire signed [15:0] rc_re = recip_rom[p_cnt[PIDX_W-1:0]][31:16];
    wire signed [15:0] rc_im = recip_rom[p_cnt[PIDX_W-1:0]][15:0];

    reg signed [15:0]     qa_rc_re, qa_rc_im;
    reg signed [IN_W-1:0] qa_p_re, qa_p_im;
    reg [PIDX_W-1:0]      qa_idx, qb_idx, qc_idx;
    reg                   qa_v, qb_v, qc_v;

    reg signed [IN_W+16:0] qb_rr, qb_ii, qb_ri, qb_ir;

    // (a+jb)(c+jd)
    wire signed [IN_W+16:0] pr  = qb_rr - qb_ii;
    wire signed [IN_W+16:0] pi_ = qb_ri + qb_ir;

    // ---- stage 2: interpolate, pipelined ----
    //
    // The first version did ROM read -> array read -> subtract ->
    // multiply -> shift -> add -> register in ONE cycle and missed
    // 100 MHz by 3.4 ns across 324 endpoints. Split four ways; the block
    // emits one bin per cycle either way, only the latency changes, and
    // at 10 Msps there are 2,880 clocks per symbol for 256 bins.
    reg [BIN_W:0] bin;
    reg           running;

    // p1: address the ROM
    reg [SLOT_W-1:0]  l1, r1;
    reg signed [16:0] w1;
    reg               v1, last1;
    reg  [7:0]        bin1, bin2, bin2b, bin3;
    // p2: read the pilot memory
    reg signed [H_W-1:0] hl2_re, hl2_im, hr2_re, hr2_im;
    reg signed [16:0]    w2;
    reg                  v2, last2;
    // p2b: the difference, registered on its own.
    //
    // Computing (hr - hl) and multiplying by w in the SAME cycle left
    // -0.654 ns across 72 endpoints: the subtract sits in front of the
    // DSP instead of behind a register, so the multiplier's input arrives
    // late. Registering the difference gives the DSP48 registered inputs,
    // which is what it wants.
    reg signed [H_W:0]     d_re_r, d_im_r;
    reg signed [H_W-1:0]   hl2b_re, hl2b_im;
    reg signed [16:0]      w2b;
    reg                    v2b, last2b;
    // p3: difference times weight
    reg signed [H_W-1:0]   hl3_re, hl3_im;
    reg signed [H_W+18:0]  t3_re, t3_im;
    reg                    v3, last3;

    wire [31:0]       rom = interp_rom[bin[BIN_W-1:0]];
    wire signed [H_W:0] d2_re = hr2_re - hl2_re;
    wire signed [H_W:0] d2_im = hr2_im - hl2_im;

    /* verilator lint_off WIDTHEXPAND */
    /* verilator lint_off WIDTHTRUNC */
    always @(posedge clk) begin
        if (rst) begin
            p_cnt <= 0; bin <= 0;
            have_pilots <= 1'b0; running <= 1'b0;
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0;
            last1 <= 1'b0; last2 <= 1'b0; last3 <= 1'b0;
            h_valid <= 1'b0; h_last <= 1'b0;
            h_re <= 0; h_im <= 0;
            // hp_re/hp_im are deliberately NOT cleared. `have_pilots`
            // guarantees write-before-read, and resetting a 224-entry
            // memory would infer a reset network on every bit and stop
            // Vivado inferring block RAM. The lint tool also rejects a
            // delayed assignment to an array inside a for loop, which is
            // what surfaced this.
        end else begin
            // ---- qa: address the ROM, capture the sample ----
            qa_v <= 1'b0;
            if (pilot_valid && !running) begin
                qa_rc_re <= rc_re;  qa_rc_im <= rc_im;
                qa_p_re  <= pilot_re; qa_p_im <= pilot_im;
                qa_idx   <= p_cnt[PIDX_W-1:0];
                qa_v     <= 1'b1;
                p_cnt    <= (p_cnt == N_PILOT - 1) ? 0 : p_cnt + 1'b1;
            end

            // ---- qb: the four products ----
            qb_rr <= qa_p_re * qa_rc_re;
            qb_ii <= qa_p_im * qa_rc_im;
            qb_ri <= qa_p_re * qa_rc_im;
            qb_ir <= qa_p_im * qa_rc_re;
            qb_idx <= qa_idx;  qb_v <= qa_v;

            // ---- qc: combine, scale, store ----
            // Shift by the GENERATED fractional width, not a literal 15:
            // the reciprocals are Q14 because 1.0 in Q15 is 32768 and
            // wraps a signed 16-bit field.
            qc_idx <= qb_idx;  qc_v <= qb_v;
            if (qb_v) begin
                hp_re[qb_idx] <= pr  >>> `CE_RECIP_FRAC;
                hp_im[qb_idx] <= pi_ >>> `CE_RECIP_FRAC;
                // Start the sweep once the LAST reference bin has
                // actually been written, not when it arrived -- the
                // pipeline means those are three cycles apart.
                if (qb_idx == N_PILOT - 1) begin
                    have_pilots <= 1'b1; running <= 1'b1; bin <= 0;
                end
            end

            // p1
            v1 <= 1'b0;
            if (running && have_pilots) begin
                l1 <= rom[SLOT_W+16 +: SLOT_W];
                r1 <= rom[16 +: SLOT_W];
                w1 <= {1'b0, rom[15:0]};
                v1 <= 1'b1;
                last1 <= (bin == N_FFT - 1);
                bin1  <= bin[7:0];
                if (bin == N_FFT - 1) begin
                    running <= 1'b0; have_pilots <= 1'b0; bin <= 0;
                end else begin
                    bin <= bin + 1'b1;
                end
            end

            // p2
            hl2_re <= hp_re[l1]; hl2_im <= hp_im[l1];
            hr2_re <= hp_re[r1]; hr2_im <= hp_im[r1];
            w2 <= w1; v2 <= v1; last2 <= last1; bin2 <= bin1;

            // p2b
            d_re_r  <= d2_re;   d_im_r  <= d2_im;
            hl2b_re <= hl2_re;  hl2b_im <= hl2_im;
            w2b <= w2; v2b <= v2; last2b <= last2; bin2b <= bin2;

            // p3
            hl3_re <= hl2b_re; hl3_im <= hl2b_im;
            t3_re <= d_re_r * w2b;
            t3_im <= d_im_r * w2b;
            v3 <= v2b; last3 <= last2b; bin3 <= bin2b;

            // p4
            h_re    <= hl3_re + (t3_re >>> 15);
            h_im    <= hl3_im + (t3_im >>> 15);
            h_valid <= v3;
            h_last  <= last3;
            h_bin   <= bin3;
        end
    end
    /* verilator lint_on WIDTHTRUNC */
    /* verilator lint_on WIDTHEXPAND */
endmodule
