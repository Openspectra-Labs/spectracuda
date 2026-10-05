// ============================================================
// rx_freq_domain.v -- RX frequency-domain stage (standalone IP)
//
//   TD -> FD  FFT bins + metadata, NO backpressure   (in_*)
//   C1        per-frame config from the bit domain   (cfg_*)
//   FD -> BIT LLR groups + metadata, valid/ready      (out_*)
//
// Interfaces are the frozen ones in fpga/docs/rx_modular_architecture.md
// section 7 (I1 = in_*, I2 = out_*, C1 = cfg_*); widths in rx_if.vh.
// Nothing outside this module needs to know how it schedules anything.
//
// INSIDE, every item travels as {data + metadata}; no block re-derives
// "which symbol / which subcarrier / header or payload" from a counter:
//
//   in_* -> [B1] -> release/classify -> grid_extract -> TRAIN: ls_chanest
//                                          |                 -> grid_extract(H)
//                                          |                 -> H store (addr = sc)
//                                          +-> HEADER/DATA: mmse_eq (data)
//                                          +-> DATA pilots: mmse_eq (pilot)
//            DATA:   eq -> pilot_cpe (+ metadata FIFO) --+
//            HEADER: eq -> header queue ----------------+-> demapper -> [B2] -> out_*
//
// B1 (BODY_BUF_SYMS symbols): the FD ingress SCHEDULER. TD cannot be
//    stalled, so every item whose processing prerequisites are not yet
//    satisfied waits here -- mainly BODY waiting for its frame's config,
//    but also HEADER waiting for its frame's channel estimate. B1 is one
//    in-order FIFO; its head is released by these rules:
//      TRAIN  -> always.
//      HEADER -> once H for the SAME frame is complete. The header is
//                equalized with that H; the old RTL relied on the
//                estimator's sweep staying ahead of the header bins, which
//                a burst out of B1 would break.
//      BODY   -> cfg_err for its frame: dropped.
//                cfg valid for its frame AND its header already emitted:
//                classified (all DATA for now -- DMRS / C2 belong here)
//                and forwarded; body symbols past cfg_body_syms dropped.
//                Otherwise: wait.
//    Config is latched per fseq (4 entries) on the cfg_valid / cfg_err
//    event, and cleared when that fseq's next frame enters B1, so a frame
//    can never consume another frame's config.
//
// B2 (B2_DEPTH groups): the only point that sees out_ready. Everything
//    before it runs at input rate and never stalls. B2's head is a
//    register, so a presented item stays stable until accepted.
//    512 is a step-3 test depth, NOT the final size (doc section 9).
//    BACKPRESSURE IS BOUNDED, not arbitrary: B2 accepts every demapper
//    output unconditionally (nothing upstream can stall), so out_ready may
//    be low for at most the free space B2 has when the stall starts (worst
//    case B2_DEPTH groups, i.e. ~2 symbols of 216 data groups). A longer
//    stall overflows B2: the group is dropped and st_b2_overflow is set --
//    an error, never silent loss. In rx_top the consumer is the FD->BD
//    CDC FIFO feeding BD at clk_bd, which drains faster than FD fills and
//    whose coded-bit FIFO holds a whole frame, so it only stalls briefly
//    (measured B2 high-water in the regressions: fd_b2_hwm).
//
// H store: single bank, written at the item's sc ordinal, valid for the
//    frame in h_fseq. Organised so a second bank (DMRS ping-pong) is an
//    added bank select, not a restructure.
//
// No per-frame reset: rst is power-on only. Frames are delimited by the
// metadata that flows with the items.
//
// DSP blocks (ls_chanest, mmse_eq, pilot_cpe, demapper) are unchanged in
// arithmetic; scaling into them is exactly the old rx_top / legacy FD.
// ============================================================
`timescale 1ns / 1ps
`include "ofdm_params.vh"
`include "grid_params.vh"
`include "header_params.vh"
`include "demap_params.vh"

module rx_freq_domain #(
    parameter integer FFT_W           = 32,
    parameter integer CE_IN_W         = 20,
    parameter integer EQ_W            = 18,
    parameter integer ANGLE_W         = 16,
    parameter integer SHIFT_FFT_TO_CE = 3,
    parameter integer SHIFT_FFT_TO_EQ = 3,
    parameter integer LLR_W           = 1,     // 1 = hard demapper, 4 = soft (demapper_soft); rx_top sets 4
    parameter integer BODY_BUF_SYMS   = 5,     // B1 = BODY_BUF_SYMS x FFT_SIZE bins
    parameter integer B2_DEPTH        = 512    // step-3 test depth
)(
    input  wire                     clk,
    input  wire                     rst,

    // ---- TD -> FD: one FFT bin per transfer, no ready ----
    input  wire                     in_valid,
    input  wire signed [FFT_W-1:0]  in_re,
    input  wire signed [FFT_W-1:0]  in_im,
    input  wire [7:0]               in_bin,
    input  wire [7:0]               in_sym_idx,
    input  wire [2:0]               in_stype,      // TRAIN / HEADER / BODY
    input  wire [1:0]               in_fseq,
    input  wire                     in_frame_start,

    // ---- C1: configuration, published by the bit domain ----
    input  wire                     cfg_valid,
    input  wire                     cfg_err,
    input  wire [1:0]               cfg_fseq,
    input  wire [2:0]               cfg_mod,       // header mod_scheme code
    input  wire [7:0]               cfg_body_syms,
    input  wire [7:0]               cfg_c2_syms,   // must be 0 until C2 exists
    input  wire [1:0]               cfg_dmrs_period, // must be 0 until DMRS exists

    // ---- FD -> BIT: one LLR group (one subcarrier) per transfer ----
    output wire                     out_valid,
    input  wire                     out_ready,
    output wire [6*LLR_W-1:0]       out_llr,       // llr[k] at [k*LLR_W +: LLR_W]
    output wire [2:0]               out_n,
    output wire [7:0]               out_sc,
    output wire [7:0]               out_sym_idx,
    output wire [2:0]               out_stype,     // HEADER / DATA
    output wire [1:0]               out_fseq,
    output wire                     out_sym_start,
    output wire                     out_sym_end,
    output wire                     out_frame_start,
    output wire                     out_frame_end,

    // ---- status (sticky errors; levels for measurement) ----
    output wire                     st_b1_overflow,
    output wire                     st_b2_overflow,
    output reg                      st_seq_err,       // metadata FIFO / merge / CPE pairing
    output reg                      st_cfg_unsupported,
    output reg                      st_fseq_collision, // frame entered on an unretired fseq
    output reg                      st_hdr_no_train,   // HEADER waiting for H nobody is making
    output reg                      st_hq_held,        // a header actually waited for older CPE data
    output wire [$clog2(BODY_BUF_SYMS*`FFT_SIZE):0] st_b1_level,
    output reg  [$clog2(BODY_BUF_SYMS*`FFT_SIZE):0] st_b1_hwm,
    output reg  [$clog2(B2_DEPTH):0]                st_b2_hwm
);
    `include "rx_if.vh"

    localparam integer N_FFT   = `FFT_SIZE;
    localparam integer N_DATA  = `N_DATA;
    localparam integer N_PILOT = `N_PILOT;
    localparam integer N_TRAIN = `N_TRAINING;
    localparam integer N_HDR   = `HDR_TOTAL_SLOTS / `N_DATA;  // header symbols
    localparam integer BODY0   = N_TRAIN + N_HDR;             // first body sym_idx
    localparam [7:0]   SC_DLAST = 8'(N_DATA - 1);
    localparam [7:0]   SC_PLAST = 8'(N_PILOT - 1);
    localparam [1:0]   DMB      = 2'(`DM_BPSK);
    localparam [1:0]   DMQ      = 2'(`DM_QPSK);

    // =================================================================
    // C1: per-fseq config table
    // =================================================================
    reg        cfg_valid_q, cfg_err_q;
    reg  [1:0] cfg_fseq_q;
    wire ok_evt  = cfg_valid && !(cfg_valid_q && cfg_fseq_q == cfg_fseq);
    wire err_evt = cfg_err   && !(cfg_err_q   && cfg_fseq_q == cfg_fseq);

    reg        t_ok   [0:3];
    reg        t_bad  [0:3];
    reg        t_hdr  [0:3];     // this frame's header has left (into B2)
    reg  [7:0] t_body [0:3];
    reg  [1:0] t_dm   [0:3];     // demapper scheme for its DATA

    // header mod_scheme code -> demapper code (bpsk0 qpsk1 qam16_2 qam64_3
    // -> qpsk0 qam16_1 qam64_2); payload BPSK falls back to QPSK, exactly
    // as the old rx_top / rx_header did.
    wire [1:0] cfg_dm = (cfg_mod >= 3'd1) ? 2'(cfg_mod - 3'd1) : DMQ;

    // header-emitted event, from the B2 push side (below)
    wire       hdr_done_evt;
    wire [1:0] hdr_done_fseq;

    integer k;
    always @(posedge clk) begin
        if (rst) begin
            cfg_valid_q <= 1'b0; cfg_err_q <= 1'b0; cfg_fseq_q <= 2'd0;
            st_cfg_unsupported <= 1'b0;
            for (k = 0; k < 4; k = k + 1) begin
                t_ok[k] <= 1'b0; t_bad[k] <= 1'b0; t_hdr[k] <= 1'b0;
                t_body[k] <= 8'd0; t_dm[k] <= DMQ;
            end
        end else begin
            cfg_valid_q <= cfg_valid; cfg_err_q <= cfg_err; cfg_fseq_q <= cfg_fseq;
            // A frame entering B1 retires whatever its fseq held before
            // (fseq wrap rule: that older frame has fully left the receiver).
            if (in_valid && in_frame_start) begin
                t_ok[in_fseq] <= 1'b0; t_bad[in_fseq] <= 1'b0; t_hdr[in_fseq] <= 1'b0;
            end
            if (ok_evt) begin
                t_ok[cfg_fseq]   <= 1'b1;
                t_body[cfg_fseq] <= cfg_body_syms;
                t_dm[cfg_fseq]   <= cfg_dm;
                if (cfg_c2_syms != 8'd0 || cfg_dmrs_period != 2'd0)
                    st_cfg_unsupported <= 1'b1;
            end
            if (err_evt) t_bad[cfg_fseq] <= 1'b1;
            if (hdr_done_evt) t_hdr[hdr_done_fseq] <= 1'b1;
        end
    end

    // =================================================================
    // B1: ingress buffer, in order
    // =================================================================
    localparam integer B1_DEPTH = BODY_BUF_SYMS * N_FFT;
    localparam integer B1_W     = 2*FFT_W + 8 + 8 + 3 + 2 + 1;

    wire              b1_hv;
    wire [B1_W-1:0]   b1_head;
    wire              b1_pop;
    wire              b1_full;

    sync_fifo_fwft #(.WIDTH(B1_W), .DEPTH(B1_DEPTH)) u_b1 (
        .clk(clk), .rst(rst),
        .push(in_valid),
        .push_data({in_re, in_im, in_bin, in_sym_idx, in_stype, in_fseq, in_frame_start}),
        .head_valid(b1_hv), .head_data(b1_head), .pop(b1_pop),
        .level(st_b1_level), .full(b1_full), .overflow(st_b1_overflow), .underflow());

    wire signed [FFT_W-1:0] h1_re, h1_im;
    wire [7:0] h1_bin, h1_sym;
    wire [2:0] h1_st;
    wire [1:0] h1_fq;
    wire       h1_fs;
    assign {h1_re, h1_im, h1_bin, h1_sym, h1_st, h1_fq, h1_fs} = b1_head;

    // ---- release / classify ----
    reg        h_done;       // H store holds a complete estimate ...
    reg  [1:0] h_fseq;       // ... for this frame
    wire [7:0] body_j    = h1_sym - 8'(BODY0);
    wire       is_train  = (h1_st == ST_TRAIN);
    wire       is_hdr    = (h1_st == ST_HEADER);
    wire       is_body   = (h1_st == ST_BODY);
    // Soft LLRs: a frame's DATA must not leave B1 before ITS channel-weight
    // reference (llr_weight, ~30 clocks after the frame's training estimate)
    // is final. Explicit, not assumed from header/config latency.
    wire       w_valid;
    wire [1:0] w_fseq;
    wire       scale_go  = (LLR_W == 1) || (w_valid && w_fseq == h1_fq);
    wire       body_go   = t_bad[h1_fq] || (t_ok[h1_fq] && t_hdr[h1_fq] && scale_go);
    wire       hdr_go    = h_done && (h_fseq == h1_fq);

    assign b1_pop = b1_hv && (is_train || (is_hdr && hdr_go) || (is_body && body_go));

    wire       body_fwd  = !t_bad[h1_fq] && (body_j < t_body[h1_fq]);
    wire       s_valid   = b1_pop && (is_train || is_hdr || body_fwd);
    wire [2:0] s_stype   = is_body ? ST_DATA : h1_st;          // classifier: BODY -> DATA
    wire [1:0] s_dm      = is_body ? t_dm[h1_fq] : DMB;
    wire       s_lastb   = is_body && (body_j == t_body[h1_fq] - 8'd1);

    always @(posedge clk) begin
        if (rst) st_b1_hwm <= 0;
        else if (st_b1_level > st_b1_hwm) st_b1_hwm <= st_b1_level;
    end

    // =================================================================
    // Grid split, metadata attached
    // =================================================================
    localparam integer GM_W = 8 + 3 + 2 + 2 + 1;   // sym, stype, fseq, dm, lastb

    wire signed [FFT_W-1:0] gd_re, gd_im, gp_re, gp_im;
    wire        gd_valid, gp_valid;
    wire [7:0]  g_sc;
    wire [GM_W-1:0] g_meta;

    grid_extract #(.DATA_W(FFT_W), .META_W(GM_W)) u_grid (
        .clk(clk), .rst(rst),
        .in_re(h1_re), .in_im(h1_im), .in_valid(s_valid), .in_bin(h1_bin),
        .in_meta({h1_sym, s_stype, h1_fq, s_dm, s_lastb}),
        .data_re(gd_re), .data_im(gd_im), .data_valid(gd_valid),
        .pilot_re(gp_re), .pilot_im(gp_im), .pilot_valid(gp_valid),
        .out_sc(g_sc), .out_meta(g_meta), .sym_done());

    wire [7:0] g_sym;
    wire [2:0] g_st;
    wire [1:0] g_fq, g_dm;
    wire       g_lastb;
    assign {g_sym, g_st, g_fq, g_dm, g_lastb} = g_meta;

    // ---- scaling, identical to the old rx_top ----
    wire known_valid = gd_valid || gp_valid;
    wire signed [FFT_W-1:0] known_re = gd_valid ? gd_re : gp_re;
    wire signed [FFT_W-1:0] known_im = gd_valid ? gd_im : gp_im;
    wire signed [FFT_W-1:0] ce_sh_re = known_re >>> SHIFT_FFT_TO_CE;
    wire signed [FFT_W-1:0] ce_sh_im = known_im >>> SHIFT_FFT_TO_CE;
    wire signed [FFT_W-1:0] eqd_sh_re = gd_re >>> SHIFT_FFT_TO_EQ;
    wire signed [FFT_W-1:0] eqd_sh_im = gd_im >>> SHIFT_FFT_TO_EQ;
    wire signed [FFT_W-1:0] eqp_sh_re = gp_re >>> SHIFT_FFT_TO_EQ;
    wire signed [FFT_W-1:0] eqp_sh_im = gp_im >>> SHIFT_FFT_TO_EQ;

    // =================================================================
    // Channel estimate (TRAIN) -> H store, addressed by sc
    // =================================================================
    wire ce_in = known_valid && (g_st == ST_TRAIN);
    reg  [1:0] ce_fseq;      // frame whose training is in the estimator

    wire signed [EQ_W-1:0] ch_re, ch_im;
    wire       ch_valid, ch_last;
    wire [7:0] ch_bin;

    ls_chanest #(.IN_W(CE_IN_W), .H_W(EQ_W)) u_ce (
        .clk(clk), .rst(rst),
        .pilot_re(ce_sh_re[CE_IN_W-1:0]), .pilot_im(ce_sh_im[CE_IN_W-1:0]),
        .pilot_valid(ce_in),
        .h_re(ch_re), .h_im(ch_im), .h_valid(ch_valid), .h_last(ch_last),
        .h_bin(ch_bin));

    wire signed [EQ_W-1:0] hd_re, hd_im, hp_re, hp_im;
    wire       hd_valid, hp_valid;
    wire [7:0] h_sc;
    wire       h_last_q;    // ch_last, aligned with the split H

    grid_extract #(.DATA_W(EQ_W), .META_W(1)) u_grid_h (
        .clk(clk), .rst(rst),
        .in_re(ch_re), .in_im(ch_im), .in_valid(ch_valid), .in_bin(ch_bin),
        .in_meta(ch_last),
        .data_re(hd_re), .data_im(hd_im), .data_valid(hd_valid),
        .pilot_re(hp_re), .pilot_im(hp_im), .pilot_valid(hp_valid),
        .out_sc(h_sc), .out_meta(h_last_q), .sym_done());

    // H store. One bank today; a DMRS ping-pong adds a bank select here.
    reg signed [EQ_W-1:0] hs_d_re [0:N_DATA-1];
    reg signed [EQ_W-1:0] hs_d_im [0:N_DATA-1];
    reg signed [EQ_W-1:0] hs_p_re [0:N_PILOT-1];
    reg signed [EQ_W-1:0] hs_p_im [0:N_PILOT-1];

    always @(posedge clk) begin
        if (hd_valid) begin hs_d_re[h_sc] <= hd_re; hs_d_im[h_sc] <= hd_im; end
        if (hp_valid) begin hs_p_re[h_sc[2:0]] <= hp_re; hs_p_im[h_sc[2:0]] <= hp_im; end
    end

    // h_done / h_fseq: the estimate is complete once its last bin (bin 255,
    // a pilot) has been written. Bins come out of the sweep in order, so
    // the last write is the last bin.
    always @(posedge clk) begin
        if (rst) begin
            ce_fseq <= 2'd0; h_done <= 1'b0; h_fseq <= 2'd0;
        end else begin
            // A new training symbol starts a new estimate: whatever H held
            // is no longer complete for anyone. (Not relying on fseq alone:
            // fseq values repeat every 4 frames.)
            if (ce_in) begin
                ce_fseq <= g_fq;
                h_done  <= 1'b0;
            end
            if ((hd_valid || hp_valid) && h_last_q) begin
                h_done <= 1'b1;
                h_fseq <= ce_fseq;
            end
        end
    end

    // =================================================================
    // Equalizers -- data (HEADER + DATA) and pilots (DATA only)
    // =================================================================
    localparam integer EM_W = 8 + 3 + 2 + 8 + 2 + 1;   // sym, stype, fseq, sc, dm, lastb
    localparam integer PM_W = 8 + 2 + 8;               // sym, fseq, sc

    wire signed [EQ_W-1:0] eqd_re, eqd_im, eqp_re, eqp_im;
    wire        eqd_valid, eqp_valid;
    wire [EM_W-1:0] eqd_meta;
    wire [2*EQ_W-1:0] ed_hh;      // |H|^2 the item was equalized with (soft LLR weight)
    // Channel-weight exponent of this item (soft_llr_metric "thresh_wq"):
    //   kcnt = #{ j in 0..3 : |H|^2 >= th_j },  k = kcnt - 3
    // i.e. |H|^2 / mean_train|H|^2 rounded to a power of two in [1/8, 2].
    // The four per-frame thresholds come from llr_weight.v (0 DSP).
    wire [39:0] w_th0, w_th1, w_th2, w_th3;
    wire [39:0] ed_hh40 = {{(40-2*EQ_W){1'b0}}, ed_hh};
    wire [2:0]  ed_kcnt = {2'd0, ed_hh40 >= w_th0} + {2'd0, ed_hh40 >= w_th1}
                        + {2'd0, ed_hh40 >= w_th2} + {2'd0, ed_hh40 >= w_th3};
    wire [PM_W-1:0] eqp_meta;

    mmse_eq #(.W(EQ_W), .META_W(EM_W)) u_eq_data (
        .clk(clk), .rst(rst),
        .rx_re(eqd_sh_re[EQ_W-1:0]), .rx_im(eqd_sh_im[EQ_W-1:0]),
        .h_re(hs_d_re[g_sc]), .h_im(hs_d_im[g_sc]),
        .in_valid(gd_valid && (g_st == ST_HEADER || g_st == ST_DATA)),
        .in_meta({g_sym, g_st, g_fq, g_sc, g_dm, g_lastb}),
        .y_re(eqd_re), .y_im(eqd_im), .y_valid(eqd_valid), .y_meta(eqd_meta),
        .y_hh(ed_hh));

    mmse_eq #(.W(EQ_W), .META_W(PM_W)) u_eq_pilot (
        .clk(clk), .rst(rst),
        .rx_re(eqp_sh_re[EQ_W-1:0]), .rx_im(eqp_sh_im[EQ_W-1:0]),
        .h_re(hs_p_re[g_sc[2:0]]), .h_im(hs_p_im[g_sc[2:0]]),
        .in_valid(gp_valid && (g_st == ST_DATA)),
        .in_meta({g_sym, g_fq, g_sc}),
        .y_re(eqp_re), .y_im(eqp_im), .y_valid(eqp_valid), .y_meta(eqp_meta),
        .y_hh());

    wire [7:0] ed_sym, ed_sc, ep_sym, ep_sc;
    wire [2:0] ed_st;
    wire [1:0] ed_fq, ed_dm, ep_fq;
    wire       ed_lastb;
    assign {ed_sym, ed_st, ed_fq, ed_sc, ed_dm, ed_lastb} = eqd_meta;
    assign {ep_sym, ep_fq, ep_sc} = eqp_meta;

    // =================================================================
    // CPE (DATA only) + its metadata FIFO
    // =================================================================
    wire cpe_dv  = eqd_valid && (ed_st == ST_DATA);
    // A symbol is complete when both its last data item and its last
    // pilot have been seen -- from their sc markers, not from counts.
    wire d_end   = cpe_dv    && (ed_sc == SC_DLAST);
    wire p_end   = eqp_valid && (ep_sc == SC_PLAST);
    reg  d_seen, p_seen;
    reg  [7:0] end_sym;      // symbol the first-seen end belonged to ...
    reg  [1:0] end_fq;       // ... and its frame (same sym_idx recurs every frame)
    wire cpe_sym_done = (d_end && (p_seen || p_end)) || (p_end && d_seen);

    always @(posedge clk) begin
        if (rst) begin
            d_seen <= 1'b0; p_seen <= 1'b0; end_sym <= 8'd0; end_fq <= 2'd0;
        end else if (cpe_sym_done) begin
            d_seen <= 1'b0; p_seen <= 1'b0;
        end else begin
            if (d_end) begin d_seen <= 1'b1; end_sym <= ed_sym; end_fq <= ed_fq; end
            if (p_end) begin p_seen <= 1'b1; end_sym <= ep_sym; end_fq <= ep_fq; end
        end
    end

    wire signed [EQ_W-1:0] cpe_re, cpe_im;
    wire cpe_ov;

    pilot_cpe #(.DATA_W(EQ_W), .ANGLE_W(ANGLE_W)) u_cpe (
        .clk(clk), .rst(rst),
        .data_re(eqd_re), .data_im(eqd_im), .data_valid(cpe_dv),
        .pilot_re(eqp_re), .pilot_im(eqp_im), .pilot_valid(eqp_valid),
        .sym_done(cpe_sym_done),
        .out_re(cpe_re), .out_im(cpe_im), .out_valid(cpe_ov),
        .cpe_angle(), .cpe_valid());

    // Metadata beside pilot_cpe: pushed when an item ENTERS, popped when an
    // item LEAVES. pilot_cpe is in-order and 1:1 for data items, so the
    // head is always the leaving item's metadata, whatever the latency.
    localparam integer CM_W = 3 + 8 + 2 + 8 + 2 + 1;   // kcnt, sym, fseq, sc, dm, lastb
    wire            cm_hv;
    wire [CM_W-1:0] cm_head;
    wire [$clog2(512):0] cm_level;
    wire            cm_ovf, cm_unf;

    sync_fifo_fwft #(.WIDTH(CM_W), .DEPTH(512)) u_cpe_meta (
        .clk(clk), .rst(rst),
        .push(cpe_dv), .push_data({ed_kcnt, ed_sym, ed_fq, ed_sc, ed_dm, ed_lastb}),
        .head_valid(cm_hv), .head_data(cm_head), .pop(cpe_ov),
        .level(cm_level), .full(), .overflow(cm_ovf), .underflow(cm_unf));

    wire [7:0] cm_sym, cm_sc;
    wire [1:0] cm_fq, cm_dm;
    wire       cm_lastb;
    wire [2:0] cm_kcnt;
    assign {cm_kcnt, cm_sym, cm_fq, cm_sc, cm_dm, cm_lastb} = cm_head;

    // =================================================================
    // Header queue: HEADER waits while OLDER data is still inside CPE
    // =================================================================
    // Both paths share the demapper. CPE output cannot stall, so a header
    // that arrives while a previous frame's data is still in CPE waits
    // here. A header's own frame's DATA cannot be in CPE yet: BODY is only
    // released from B1 after that header has left (t_hdr).
    localparam integer HQ_W = 2*EQ_W + 8 + 2 + 8;      // y, sym, fseq, sc
    wire            hq_hv;
    wire [HQ_W-1:0] hq_head;
    wire            hq_ovf;
    // cm_level counts every DATA item inside CPE (stored + head).
    wire hq_pop = hq_hv && (cm_level == 0) && !cpe_ov;

    sync_fifo_fwft #(.WIDTH(HQ_W), .DEPTH(256)) u_hdrq (
        .clk(clk), .rst(rst),
        .push(eqd_valid && (ed_st == ST_HEADER)),
        .push_data({eqd_re, eqd_im, ed_sym, ed_fq, ed_sc}),
        .head_valid(hq_hv), .head_data(hq_head), .pop(hq_pop),
        .level(), .full(), .overflow(hq_ovf), .underflow());

    wire signed [EQ_W-1:0] hq_re, hq_im;
    wire [7:0] hq_sym, hq_sc;
    wire [1:0] hq_fq;
    assign {hq_re, hq_im, hq_sym, hq_fq, hq_sc} = hq_head;

    // =================================================================
    // Demapper (BPSK header / QAM data), metadata through its pipeline
    // =================================================================
    localparam integer DMM_W = 8 + 3 + 2 + 8 + 1;      // sym, stype, fseq, sc, lastb
    wire       dm_in_v = cpe_ov || hq_pop;
    wire [5:0] dm_bits;
    wire [3:0] dm_n;
    wire       dm_v;
    wire [DMM_W-1:0] dm_meta;

    // ---- per-frame channel-weight thresholds ----
    // The training |H|^2 sum is taken from the FIRST header symbol's items at
    // the equalizer output: they are equalized with that training estimate
    // and mmse_eq already gives their |H|^2 (no multiplier). See llr_weight.v.
    llr_weight #(.HW(2*EQ_W)) u_llr_weight (
        .clk(clk), .rst(rst),
        .s_valid(eqd_valid && ed_st == ST_HEADER && ed_sym == 8'(N_TRAIN)),
        .s_hh(ed_hh), .s_last(ed_sc == SC_DLAST), .s_fseq(ed_fq),
        .th0(w_th0), .th1(w_th1), .th2(w_th2), .th3(w_th3),
        .w_valid(w_valid), .w_fseq(w_fseq));

    wire [23:0] dm_llr;           // soft: 6 x 4-bit, llr[k] at [4k +: 4]
    generate if (LLR_W == 1) begin : g_hard_dm
        demapper #(.W(EQ_W), .META_W(DMM_W)) u_dm (
            .clk(clk), .rst(rst),
            .y_re(cpe_ov ? cpe_re : hq_re), .y_im(cpe_ov ? cpe_im : hq_im),
            .mod_scheme(cpe_ov ? cm_dm : DMB),
            .in_valid(dm_in_v),
            .in_meta(cpe_ov ? {cm_sym, ST_DATA, cm_fq, cm_sc, cm_lastb}
                            : {hq_sym, ST_HEADER, hq_fq, hq_sc, 1'b0}),
            .bits(dm_bits), .n_bits(dm_n), .out_valid(dm_v), .out_meta(dm_meta));
        assign dm_llr = 24'd0;
    end else begin : g_soft_dm
        // 4-bit soft values for DATA ("thresh_w": fixed-scale max-log x
        // 2^k channel weight; header stays hard, +/-7). One in-order pipeline
        // for header and data, so the B2 stream keeps its order.
        demapper_soft #(.W(EQ_W), .META_W(DMM_W)) u_dm (
            .clk(clk), .rst(rst),
            .y_re(cpe_ov ? cpe_re : hq_re), .y_im(cpe_ov ? cpe_im : hq_im),
            .kcnt(cpe_ov ? cm_kcnt : 3'd3),
            .mod_scheme(cpe_ov ? cm_dm : DMB),
            .in_valid(dm_in_v),
            .in_meta(cpe_ov ? {cm_sym, ST_DATA, cm_fq, cm_sc, cm_lastb}
                            : {hq_sym, ST_HEADER, hq_fq, hq_sc, 1'b0}),
            .llr(dm_llr), .n_bits(dm_n), .out_valid(dm_v), .out_meta(dm_meta));
        assign dm_bits = 6'd0;
        initial if (LLR_W != 4) $fatal(1, "rx_freq_domain: soft LLRs are 4 bits (LLR_W = 4)");
    end endgenerate

    wire [7:0] o_sym, o_sc;
    wire [2:0] o_st;
    wire [1:0] o_fq;
    wire       o_lastb;
    assign {o_sym, o_st, o_fq, o_sc, o_lastb} = dm_meta;

    // llr[0] = first bit sent; groups beyond n are 0.
    //   LLR_W = 1 (hard): 0 for bit 0, 1 for bit 1 (demapper bits[5-k]).
    //   LLR_W = 4 (soft): signed 4-bit max-log LLR, > 0 means bit 0
    //   (demapper_soft). Either way the hard decision is the sign bit.
    wire [6*LLR_W-1:0] o_llr;
    genvar gi;
    generate for (gi = 0; gi < 6; gi = gi + 1) begin : g_llr
        if (LLR_W == 1) begin : g_h
            assign o_llr[gi*LLR_W +: LLR_W] =
                (gi < dm_n) ? {LLR_W{dm_bits[5-gi]}} : {LLR_W{1'b0}};
        end else begin : g_s
            assign o_llr[gi*LLR_W +: LLR_W] = dm_llr[4*gi +: 4];
        end
    end endgenerate

    wire o_sym_start   = (o_sc == 8'd0);
    wire o_sym_end     = (o_sc == SC_DLAST);
    wire o_frame_start = (o_st == ST_HEADER) && (o_sym == 8'(N_TRAIN)) && o_sym_start;
    wire o_frame_end   = (o_st == ST_DATA) && o_lastb && o_sym_end;

    assign hdr_done_evt  = dm_v && (o_st == ST_HEADER) && (o_sym == 8'(BODY0 - 1)) && o_sym_end;
    assign hdr_done_fseq = o_fq;

    // =================================================================
    // B2: egress FIFO, the only place out_ready is seen
    // =================================================================
    localparam integer B2_W = 6*LLR_W + 3 + 8 + 8 + 3 + 2 + 4;
    wire [B2_W-1:0] b2_head;
    wire [$clog2(B2_DEPTH):0] b2_level;

    sync_fifo_fwft #(.WIDTH(B2_W), .DEPTH(B2_DEPTH)) u_b2 (
        .clk(clk), .rst(rst),
        .push(dm_v),
        .push_data({o_llr, dm_n[2:0], o_sc, o_sym, o_st, o_fq,
                    o_sym_start, o_sym_end, o_frame_start, o_frame_end}),
        .head_valid(out_valid), .head_data(b2_head), .pop(out_ready),
        .level(b2_level), .full(), .overflow(st_b2_overflow), .underflow());

    assign {out_llr, out_n, out_sc, out_sym_idx, out_stype, out_fseq,
            out_sym_start, out_sym_end, out_frame_start, out_frame_end} = b2_head;

    always @(posedge clk) begin
        if (rst) st_b2_hwm <= 0;
        else if (b2_level > st_b2_hwm) st_b2_hwm <= b2_level;
    end

    // =================================================================
    // Frame residency: fseq wrap safety, deadlock and header-queue checks
    // =================================================================
    // A frame RETIRES when nothing of it is left anywhere in FD: none of
    // its bins are in B1, and its last output item has been ACCEPTED --
    // frame_end, or for a cfg_err frame the last header group (its body
    // is dropped in B1). Everything between B1 and the output is in
    // order, so the last item leaving means every item has left.
    //
    // A frame entering on an fseq whose previous frame has not retired is
    // the wrap rule broken (doc section 7): st_fseq_collision. This is
    // also exactly when the per-fseq config entry is cleared, so the
    // check proves the clear is safe.
    localparam integer B1_AWL = $clog2(BODY_BUF_SYMS * `FFT_SIZE) + 1;
    reg [B1_AWL-1:0] b1_cnt [0:3];
    reg              fe_acc [0:3];      // frame_end accepted
    reg              he_acc [0:3];      // last header group accepted
    wire acc       = out_valid && out_ready;
    wire acc_fe    = acc && out_frame_end;
    wire acc_he    = acc && (out_stype == ST_HEADER) && out_sym_end &&
                     (out_sym_idx == 8'(BODY0 - 1));
    wire in_fs_ev  = in_valid && in_frame_start;
    wire retired_in = (b1_cnt[in_fseq] == 0) &&
                      (fe_acc[in_fseq] || (t_bad[in_fseq] && he_acc[in_fseq]));

    // HEADER may only wait for H if its own TRAIN has already been
    // released into the estimator -- otherwise nothing will ever make
    // that H and B1 deadlocks. Tracked at B1 release, not at the
    // estimator, so there is no pipeline-delay window.
    reg       tr_seen;
    reg [1:0] tr_fseq;

    integer q;
    always @(posedge clk) begin
        if (rst) begin
            for (q = 0; q < 4; q = q + 1) begin
                b1_cnt[q] <= 0; fe_acc[q] <= 1'b1; he_acc[q] <= 1'b1;
            end
            st_fseq_collision <= 1'b0;
            st_hdr_no_train   <= 1'b0;
            st_hq_held        <= 1'b0;
            tr_seen <= 1'b0; tr_fseq <= 2'd0;
        end else begin
            for (q = 0; q < 4; q = q + 1) begin
                if (in_valid && in_fseq == q[1:0] && !(b1_pop && h1_fq == q[1:0]))
                    b1_cnt[q] <= b1_cnt[q] + 1'b1;
                else if (!(in_valid && in_fseq == q[1:0]) && b1_pop && h1_fq == q[1:0])
                    b1_cnt[q] <= b1_cnt[q] - 1'b1;
            end
            if (in_fs_ev) begin
                if (!retired_in) st_fseq_collision <= 1'b1;
                fe_acc[in_fseq] <= 1'b0;
                he_acc[in_fseq] <= 1'b0;
            end
            if (acc_fe) fe_acc[out_fseq] <= 1'b1;
            if (acc_he) he_acc[out_fseq] <= 1'b1;

            if (b1_pop && is_train) begin tr_seen <= 1'b1; tr_fseq <= h1_fq; end
            if (b1_hv && is_hdr && !hdr_go && !(tr_seen && tr_fseq == h1_fq))
                st_hdr_no_train <= 1'b1;

            // a header entering the queue while older data is inside CPE
            if (eqd_valid && ed_st == ST_HEADER && cm_level != 0)
                st_hq_held <= 1'b1;
        end
    end

    // =================================================================
    // Sequence checks (sticky, cheap; read by the stage testbench)
    // =================================================================
    reg [7:0] cm_exp_sc;
    always @(posedge clk) begin
        if (rst) begin
            st_seq_err <= 1'b0; cm_exp_sc <= 8'd0;
        end else begin
            if (cm_ovf || cm_unf || hq_ovf) st_seq_err <= 1'b1;
            // CPE must emit each symbol's data items in sc order.
            if (cpe_ov) begin
                if (cm_sc != cm_exp_sc) st_seq_err <= 1'b1;
                cm_exp_sc <= (cm_sc == SC_DLAST) ? 8'd0 : cm_sc + 8'd1;
            end
            // The two ends that complete a CPE symbol must be one symbol.
            // soft: a data item must be weighted with its own frame's reference
            if (LLR_W != 1 && cpe_dv && !(w_valid && w_fseq == ed_fq)) st_seq_err <= 1'b1;
            // the two ends of a CPE symbol must be the SAME (frame, symbol)
            if (d_end && p_seen && {end_fq, end_sym} != {ed_fq, ed_sym}) st_seq_err <= 1'b1;
            if (p_end && d_seen && {end_fq, end_sym} != {ep_fq, ep_sym}) st_seq_err <= 1'b1;
            if (d_end && p_end  && ed_sym  != ep_sym) st_seq_err <= 1'b1;
        end
    end
endmodule
