`timescale 1ns / 1ps
`include "ofdm_params.vh"
`include "grid_params.vh"
`include "header_params.vh"
`include "demap_params.vh"

// ============================================================
// rx_time_domain.v -- everything before the FFT, and the FFT itself
//
//   in -> sc_sync -> frame_sync -> cfo_estimate/cfo_correct
//      -> frame phase FSM -> cp_fft -> frequency-domain bins out
//
// The frame phase FSM lives here because it counts TIME-DOMAIN samples.
// It is also the one place that knows which slot is training, header or
// payload, so it sets the symbol-type tag (rx_stype.vh) that rides with
// the bins from here on. The frame length comes back on the C1 config
// interface (cfg_body_syms, only meaningful while cfg_body_valid): until
// it arrives the FSM keeps capturing BODY symbols, and the frequency
// domain drops any past the real end (docs/rx_modular_architecture.md
// section 7, I1 has no frame_end).
//
// Moved verbatim out of rx_top.v (sections 1-4); see that file's
// history notes for why each register stage is there.
// ============================================================
module rx_time_domain #(
    parameter integer SAMPLE_W  = 16,
    parameter integer ACC_W     = 48,
    parameter integer FFT_W     = 32,
    parameter integer ANGLE_W   = 16,
    parameter integer BUF_W     = 12,
    parameter integer MAX_PAYLOAD_SYM = 128
)(
    input  wire                       clk,
    input  wire                       rst,

    input  wire signed [SAMPLE_W-1:0] in_i,
    input  wire signed [SAMPLE_W-1:0] in_q,
    input  wire                       in_valid,

    // C1: frame length after the header, in OFDM symbols. Read ONLY
    // while cfg_body_valid. It used to be rx_header's n_pay_sym read
    // unconditionally -- reset value 1, so if the header had not been
    // decoded by the end of the first payload slot the frame silently
    // ended after one symbol (doc H2). Exposed in step 3c, when the new
    // frequency domain correctly made the header wait for its H.
    input  wire                       cfg_body_valid,
    input  wire [7:0]                 cfg_body_syms,

    // Frame boundary: one pulse at the first replayed sample. Every
    // downstream wrapper uses it to reset its per-frame state.
    output wire                       frame_start,

    // Frequency-domain bins, natural order, with the symbol sideband.
    output wire signed [FFT_W-1:0]    fft_re,
    output wire signed [FFT_W-1:0]    fft_im,
    output wire                       fft_valid,
    output wire                       fft_sof,
    output wire [1:0]                 fft_stype,
    output wire [7:0]                 fft_bin
);
    `include "rx_stype.vh"
    localparam integer FFT_SIZE   = `FFT_SIZE;
    localparam integer CP_LEN     = `CP_LEN;
    localparam integer SLOT_LEN   = `SLOT_LEN;
    localparam integer N_TRAINING = `N_TRAINING;
    localparam integer LAG        = `SC_LAG;

    // ---------------------------------------------------------------
    // 1. Detection and replay
    // ---------------------------------------------------------------
    wire signed [ACC_W-1:0] p_re, p_im, r_sum;
    wire sc_valid;

    sc_sync_rtl #(.SAMPLE_W(SAMPLE_W), .LAG(LAG), .ACC_W(ACC_W)) u_sync (
        .clk(clk), .rst(rst), .in_i(in_i), .in_q(in_q), .in_valid(in_valid),
        .p_re(p_re), .p_im(p_im), .r_sum(r_sum), .out_valid(sc_valid));

    wire signed [SAMPLE_W-1:0] fs_i, fs_q;
    wire fs_valid, fs_start, fs_detected;
    wire [BUF_W-1:0] fs_index;

    localparam integer FRAME_LEN =
        FFT_SIZE + (N_TRAINING + 1 + MAX_PAYLOAD_SYM) * SLOT_LEN;

    frame_sync #(.SAMPLE_W(SAMPLE_W), .ACC_W(ACC_W), .BUF_W(BUF_W),
                 .FRAME_LEN(FRAME_LEN)) u_fs (
        .clk(clk), .rst(rst), .in_i(in_i), .in_q(in_q), .in_valid(in_valid),
        .p_re(p_re), .p_im(p_im), .r_sum(r_sum), .sc_valid(sc_valid),
        .out_i(fs_i), .out_q(fs_q), .out_valid(fs_valid),
        .frame_start(fs_start), .detected(fs_detected), .start_index(fs_index),
        .det_p_re(det_p_re), .det_p_im(det_p_im), .cfo_loaded(cfo_loaded));

    // ---------------------------------------------------------------
    // 2. CFO: estimated once from the winning P, applied to the replay
    // ---------------------------------------------------------------
    // cfo_estimate already emits d_phase = -angle_turns, which is the
    // sense cfo_correct wants -- no negation here.
    wire signed [ANGLE_W-1:0] cfo_angle, cfo_dphase;
    wire cfo_angle_valid;

    // P of the accepted candidate, from frame_sync (see its port comment).
    wire signed [ACC_W-1:0] det_p_re, det_p_im;

    // cfo_estimate registers d_phase ON its out_valid edge, so in the
    // out_valid cycle d_phase still holds the PREVIOUS frame's value.
    // cfo_correct used to latch it right then: a single frame was never
    // CFO-corrected at all, and back-to-back frames were corrected with
    // the previous frame's estimate. One cycle later d_phase is the new
    // value. frame_sync waits for this pulse before replaying.
    reg cfo_loaded;
    always @(posedge clk) begin
        if (rst) cfo_loaded <= 1'b0;
        else     cfo_loaded <= cfo_angle_valid;
    end

    cfo_estimate #(.ACC_W(ACC_W), .ANGLE_W(ANGLE_W)) u_cfo_est (
        .clk(clk), .rst(rst),
        .p_re(det_p_re), .p_im(det_p_im), .p_valid(fs_detected),
        .angle_turns(cfo_angle), .d_phase(cfo_dphase),
        .out_valid(cfo_angle_valid));

    // ---- registered interface: frame_sync -> cfo_correct -------------
    // frame_sync's out_i/out_q come straight off its sample BRAM, and
    // Vivado merges that output register into cordic_rot's input
    // register inside cfo_correct -- so BRAM clock-to-out drove the
    // CORDIC's quadrant logic with nothing between. Post-route that was
    // the last failing path in the design.
    //
    // This is the THIRD instance of the same shape (the others were
    // pilot_cpe's symbol buffer -> cordic_rot, and the coded-bit FIFO ->
    // viterbi_dec). Standing rule for this design: a memory output that
    // feeds a consumer gets its own register stage.
    //
    // frame_start rides the same delay so the phase sequencer downstream
    // stays aligned with the samples it is counting.
    reg signed [SAMPLE_W-1:0] fsq_i, fsq_q;
    reg                       fsq_valid, fsq_start;
    always @(posedge clk) begin
        if (rst) begin
            fsq_i <= {SAMPLE_W{1'b0}}; fsq_q <= {SAMPLE_W{1'b0}};
            fsq_valid <= 1'b0; fsq_start <= 1'b0;
        end else begin
            fsq_i     <= fs_i;
            fsq_q     <= fs_q;
            fsq_valid <= fs_valid;
            fsq_start <= fs_start;
        end
    end

    wire signed [SAMPLE_W-1:0] cor_i, cor_q;
    wire cor_valid;

    cfo_correct #(.SAMPLE_W(SAMPLE_W), .ANGLE_W(ANGLE_W)) u_cfo_cor (
        .clk(clk), .rst(rst),
        .in_i(fsq_i), .in_q(fsq_q), .in_valid(fsq_valid),
        .d_phase(cfo_dphase), .d_phase_valid(cfo_loaded),
        .out_i(cor_i), .out_q(cor_q), .out_valid(cor_valid));

    // ---------------------------------------------------------------
    // 3. Frame phase sequencer
    // ---------------------------------------------------------------
    localparam [2:0] PH_IDLE = 3'd0, PH_PRE = 3'd1, PH_TRAIN = 3'd2,
                     PH_HDR  = 3'd3, PH_PAY = 3'd4;
    reg [2:0] phase;
    reg [15:0] ph_cnt;      // samples within the current phase
    reg [7:0]  slot_idx;    // slots consumed in this phase

    wire slot_last = (ph_cnt == 16'(SLOT_LEN - 1));
    wire slot_sof  = (ph_cnt == 16'd0);

    always @(posedge clk) begin
        if (rst) begin
            phase <= PH_IDLE; ph_cnt <= 16'd0; slot_idx <= 8'd0;
        end else if (fsq_start) begin
            phase <= PH_PRE;  ph_cnt <= 16'd0; slot_idx <= 8'd0;
        end else if (cor_valid) begin
            case (phase)
            PH_PRE: begin
                // The preamble carries no CP -- it is the sync word.
                if (ph_cnt == 16'(FFT_SIZE - 1)) begin
                    phase <= PH_TRAIN; ph_cnt <= 16'd0;
                end else ph_cnt <= ph_cnt + 1'b1;
            end
            PH_TRAIN: if (slot_last) begin
                ph_cnt <= 16'd0;
                if (slot_idx == 8'(N_TRAINING - 1)) begin
                    phase <= PH_HDR; slot_idx <= 8'd0;
                end else slot_idx <= slot_idx + 1'b1;
            end else ph_cnt <= ph_cnt + 1'b1;
            PH_HDR: if (slot_last) begin
                ph_cnt <= 16'd0; slot_idx <= 8'd0; phase <= PH_PAY;
            end else ph_cnt <= ph_cnt + 1'b1;
            PH_PAY: if (slot_last) begin
                ph_cnt <= 16'd0;
                // Known length: stop after it. Unknown: keep going (FD drops
                // the extras), bounded by what frame_sync replays.
                if ((cfg_body_valid && slot_idx >= cfg_body_syms - 1'b1) ||
                    slot_idx == 8'(MAX_PAYLOAD_SYM - 1)) begin
                    phase <= PH_IDLE; slot_idx <= 8'd0;
                end else slot_idx <= slot_idx + 1'b1;
            end else ph_cnt <= ph_cnt + 1'b1;
            default: ;
            endcase
        end
    end

    wire in_symbol = (phase == PH_TRAIN) || (phase == PH_HDR) || (phase == PH_PAY);

    // ---------------------------------------------------------------
    // 4. CP strip + FFT
    // ---------------------------------------------------------------
    // ---- symbol-type sideband ----------------------------------
    // Set HERE, once, from the time-domain phase FSM -- the one place
    // that actually knows -- and carried with the data from then on, so
    // no block downstream has to re-derive "which symbol is this" from
    // its own counter. Every phase-domain bug in the stage 5 notes was a
    // downstream counter disagreeing with this FSM.
    wire [1:0] slot_stype = (phase == PH_TRAIN) ? ST_TRAIN :
                            (phase == PH_HDR)   ? ST_HDR   : ST_PAY;

    cp_fft #(.SAMPLE_W(SAMPLE_W), .FFT_SIZE(FFT_SIZE), .CP_LEN(CP_LEN),
             .OUT_W(FFT_W)) u_fft (
        .clk(clk), .rst(rst),
        .in_i(cor_i), .in_q(cor_q),
        .in_valid(cor_valid && in_symbol), .sof(slot_sof && in_symbol),
        .in_stype(slot_stype),
        .out_re(fft_re), .out_im(fft_im), .out_valid(fft_valid),
        .out_last(), .out_sof(fft_sof), .out_stype(fft_stype),
        .out_bin(fft_bin),
        .overflow());

    assign frame_start = fsq_start;
endmodule
