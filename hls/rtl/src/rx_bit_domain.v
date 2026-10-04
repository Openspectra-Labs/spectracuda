// ============================================================
// rx_bit_domain.v -- RX bit/FEC stage (standalone IP)
//
//   FD -> BIT  LLR groups + metadata, valid/ready      (in_*)
//   host cfg   payload geometry, host-supplied (H8)     (cfg_encoded_bits,
//                                                        cfg_di_*)
//   C1         per-frame config, published to FD / TD   (cfg_*)
//   O1         deinterleaved bytes + frame end + fseq   (out_*) -> host
//
// Interfaces: docs/rx_modular_architecture.md section 7. Reed-Solomon,
// payload CRC and the MAC stay on the host (rundown section 2).
//
// Inside, the input is routed by symbol type:
//
//   HEADER -> header_decode (unchanged) -> config publisher -> C1
//   DATA   -> compact coded-bit FIFO -> unpack (skid) -> viterbi_dec
//          -> bit->byte packer -> hold -> deinterleaver -> O1
//
// CONFIG PUBLISHER. cfg_valid (or cfg_err) is driven only once every
// field is FINAL -- the header is decoded and the payload symbol count has
// finished accumulating -- tagged with the fseq of the header it came
// from, and held until the next frame's bundle replaces it.
//
// COMPACT CODED-BIT FIFO (CB): option (a) of the step-4 plan. At C=1 the
// demapper outruns the Viterbi for a whole frame, so a frame's coded bits
// must be held somewhere; here an entry is 9 bits ({first, n code, 6
// bits}), which maps to BRAM's 4K x 9 shape -- a whole frame (CB_DEPTH,
// default MAX_PAYLOAD_SYM x N_DATA) costs what the old coded-bit FIFO
// did. in_ready is low only when CB is full, so FD's B2 stays small. CB
// absorbs BURSTS; it is not a substitute for the sustained-rate contract
// (doc section 9).
//
// A FRAME STARTS ON ITS OWN DATA, NOT ON hdr_done (H4). The Viterbi and
// the deinterleaver are started when the frame's first DATA entry is
// issued -- and only once the previous frame has fully left (its last
// byte out of the deinterleaver). viterbi_dec ignores `start` unless idle,
// so the old design would have dropped a start that arrived mid-flush.
//
// No per-frame reset: rst is power-on only (H1). Per-frame state resets
// on that frame's start event.
// ============================================================
`timescale 1ns / 1ps
`include "ofdm_params.vh"
`include "header_params.vh"

module rx_bit_domain #(
    parameter integer LLR_W           = 1,
    parameter integer MAX_PAYLOAD_SYM = 128,
    parameter integer CB_DEPTH        = MAX_PAYLOAD_SYM * `N_DATA
)(
    input  wire                     clk,
    input  wire                     rst,

    // ---- FD -> BIT ----
    input  wire                     in_valid,
    output wire                     in_ready,
    input  wire [6*LLR_W-1:0]       in_llr,
    input  wire [2:0]               in_n,
    input  wire [7:0]               in_sc,
    input  wire [7:0]               in_sym_idx,
    input  wire [2:0]               in_stype,
    input  wire [1:0]               in_fseq,
    input  wire                     in_sym_start,
    input  wire                     in_sym_end,
    input  wire                     in_frame_start,
    input  wire                     in_frame_end,

    // ---- host-supplied payload geometry (H8, unchanged) ----
    input  wire [15:0]              cfg_encoded_bits,
    input  wire [12:0]              cfg_di_units,
    input  wire [12:0]              cfg_di_rows,
    input  wire [12:0]              cfg_di_cols,

    // ---- C1: config published from the decoded header ----
    output reg                      cfg_valid,
    output reg                      cfg_err,
    output reg  [1:0]               cfg_fseq,
    output reg  [2:0]               cfg_mod,
    output reg  [7:0]               cfg_body_syms,
    output wire [7:0]               cfg_c2_syms,     // 0 until C2 exists
    output wire [1:0]               cfg_dmrs_period, // 0 until the new header
    // header fields for the host (valid with cfg_valid)
    output reg  [15:0]              hdr_payload_len_bits,
    output reg  [7:0]               hdr_mod_scheme,
    output reg  [4:0]               hdr_fec0,
    output reg  [4:0]               hdr_fec1,
    output reg  [2:0]               hdr_crc,

    // ---- O1: deinterleaved bytes to the host ----
    output wire                     out_valid,
    output wire [7:0]               out_byte,
    output wire                     out_last,
    output reg  [1:0]               out_fseq,
    output wire                     frame_done,      // Viterbi finished (compat)

    // ---- status ----
    output wire                     st_cb_overflow,
    output reg                      st_unit_collision, // byte arrived while one waited
    output reg                      st_seq_err,        // DATA for a frame without valid config
    output reg  [$clog2(CB_DEPTH):0] st_cb_hwm
);
    `include "rx_if.vh"
    localparam integer N_DATA = `N_DATA;

    wire acc     = in_valid && in_ready;
    wire acc_hdr = acc && (in_stype == ST_HEADER);
    wire acc_dat = acc && (in_stype == ST_DATA);

    assign cfg_c2_syms     = 8'd0;
    assign cfg_dmrs_period = 2'd0;

    // =================================================================
    // Header path: header_decode -> config publisher
    // =================================================================
    wire        hd_done, hd_valid;
    wire [7:0]  hd_ver, hd_mod;
    wire [15:0] hd_len;
    wire [3:0]  hd_bps;
    wire [2:0]  hd_crc;
    wire [4:0]  hd_fec0, hd_fec1;
    wire [63:0] hd_user;

    // hard bit = sign of llr[0] (BPSK: n = 1)
    header_decode u_hdr (
        .clk(clk), .rst(rst),
        .in_bit(in_llr[LLR_W-1]), .in_valid(acc_hdr),
        .in_sof(acc_hdr && in_frame_start),
        .done(hd_done), .fields_valid(hd_valid),
        .protocol_version(hd_ver), .payload_len_bits(hd_len),
        .mod_scheme(hd_mod), .bits_per_symbol(hd_bps),
        .crc_code(hd_crc), .fec0_code(hd_fec0), .fec1_code(hd_fec1),
        .user_data(hd_user));

    reg  [1:0]  hdr_fseq;      // frame whose header is being decoded
    reg         counting, pend_valid;
    reg  [31:0] sym_acc;
    reg  [7:0]  n_sym;
    // Payload symbol count = ceil(encoded_bits / (N_DATA * bps)), by
    // accumulation exactly as the old rx_header did (no divider).
    wire [31:0] bits_per_sym = 32'(N_DATA) * 32'(hd_bps);

    always @(posedge clk) begin
        if (rst) begin
            hdr_fseq <= 2'd0; counting <= 1'b0; pend_valid <= 1'b0;
            sym_acc <= 32'd0; n_sym <= 8'd0;
            cfg_valid <= 1'b0; cfg_err <= 1'b0; cfg_fseq <= 2'd0;
            cfg_mod <= 3'd0; cfg_body_syms <= 8'd0;
            hdr_payload_len_bits <= 16'd0; hdr_mod_scheme <= 8'd0;
            hdr_fec0 <= 5'd0; hdr_fec1 <= 5'd0; hdr_crc <= 3'd0;
        end else begin
            if (acc_hdr && in_frame_start) hdr_fseq <= in_fseq;

            if (hd_done) begin
                pend_valid <= hd_valid;
                n_sym      <= 8'd0;
                sym_acc    <= 32'd0;
                counting   <= 1'b1;
            end else if (counting) begin
                // bps == 0 or a rejected header: nothing to count
                if (!pend_valid || hd_bps == 4'd0 ||
                    sym_acc >= 32'(cfg_encoded_bits)) begin
                    counting <= 1'b0;
                    // ---- publish: every field final, one bundle ----
                    cfg_fseq <= hdr_fseq;
                    cfg_valid <= pend_valid;
                    cfg_err   <= !pend_valid;
                    cfg_mod   <= hd_mod[2:0];
                    cfg_body_syms <= n_sym;
                    hdr_payload_len_bits <= hd_len;
                    hdr_mod_scheme <= hd_mod;
                    hdr_fec0 <= hd_fec0; hdr_fec1 <= hd_fec1; hdr_crc <= hd_crc;
                end else begin
                    sym_acc <= sym_acc + bits_per_sym;
                    n_sym   <= n_sym + 1'b1;
                end
            end
        end
    end

    // =================================================================
    // Payload path: compact coded-bit FIFO
    // =================================================================
    // first: this is the frame's first DATA item (derived from the
    // stream: the first DATA after that frame's header started).
    reg  in_seen_data;
    always @(posedge clk) begin
        if (rst) in_seen_data <= 1'b0;
        else if (acc_hdr && in_frame_start) in_seen_data <= 1'b0;
        else if (acc_dat) in_seen_data <= 1'b1;
    end
    wire in_first = acc_dat && !in_seen_data;

    // hard bits MSB-first (llr[0] = first bit -> bits[5]); n coded 2 bits:
    // n = 2/4/6 -> 1/2/3 (n/2). BPSK never reaches this path.
    wire [5:0] in_bits;
    genvar gk;
    generate for (gk = 0; gk < 6; gk = gk + 1) begin : g_bits
        assign in_bits[5-gk] = in_llr[gk*LLR_W + LLR_W-1];
    end endgenerate

    localparam integer CB_W = 1 + 2 + 6;
    wire            cb_hv, cb_full;
    wire [CB_W-1:0] cb_head;
    wire            cb_pop;
    wire [$clog2(CB_DEPTH):0] cb_level;

    sync_fifo_fwft #(.WIDTH(CB_W), .DEPTH(CB_DEPTH)) u_cb (
        .clk(clk), .rst(rst),
        .push(acc_dat), .push_data({in_first, in_n[2:1], in_bits}),
        .head_valid(cb_hv), .head_data(cb_head), .pop(cb_pop),
        .level(cb_level), .full(cb_full), .overflow(st_cb_overflow), .underflow());

    // One fseq per frame, pushed with its first DATA item.
    wire       fq_hv;
    wire [1:0] fq_head;
    wire       fq_pop;
    sync_fifo_fwft #(.WIDTH(2), .DEPTH(4)) u_fq (
        .clk(clk), .rst(rst),
        .push(in_first), .push_data(in_fseq),
        .head_valid(fq_hv), .head_data(fq_head), .pop(fq_pop),
        .level(), .full(), .overflow(), .underflow());

    assign in_ready = !cb_full;

    always @(posedge clk) begin
        if (rst) st_cb_hwm <= 0;
        else if (cb_level > st_cb_hwm) st_cb_hwm <= cb_level;
    end

    // =================================================================
    // Unpack into the Viterbi (skid-buffered, as the old bit decoder)
    // =================================================================
    wire       h_first = cb_head[8];
    wire [1:0] h_half  = cb_head[7:6];     // n / 2: symbols in this entry
    wire [5:0] h_bits  = cb_head[5:0];

    reg  [2:0] sub;                        // symbol index within the entry
    reg  [1:0] sym_q;
    reg        sym_valid_q;
    reg        busy;                       // a frame is in Viterbi/deinterleaver
    wire       vit_ready;
    wire       b_drains = sym_valid_q ? vit_ready : 1'b1;

    // A frame's first entry may start only when the previous frame is out.
    wire       at_first = h_first && (sub == 3'd0);
    wire       start_ok = !at_first || !busy;

    // PADDING DISCARD. The frame's last OFDM symbol is padded, so it holds
    // coded symbols past cfg_encoded_bits. Once `last` has been pushed the
    // Viterbi accepts nothing more, and those leftovers used to sit in the
    // skid buffer until TD's broadcast frame_start reset flushed them (H1).
    // Now: after the last push, the rest of the frame's entries are popped
    // and dropped -- as Python truncates to the encoded length.
    reg        pushed_all;                 // this frame's `last` has gone in
    wire       drop     = cb_hv && pushed_all && !at_first;
    wire       issue    = cb_hv && !drop && start_ok && b_drains;
    wire       f_start  = issue && at_first;
    wire       last_sub = (sub == {1'b0, h_half} - 3'd1);
    assign     cb_pop   = drop || (issue && last_sub);
    assign     fq_pop   = f_start;

    // bits leave MSB-first; viterbi_dec takes sym[0] = first bit of the pair
    wire [2:0] hi      = 3'd5 - {sub[1:0], 1'b0};
    wire [1:0] sym_now = {h_bits[hi - 3'd1], h_bits[hi]};

    wire vit_last;
    always @(posedge clk) begin
        if (rst) begin
            sub <= 3'd0; sym_q <= 2'd0; sym_valid_q <= 1'b0; pushed_all <= 1'b0;
        end else begin
            if (drop)       sub <= 3'd0;
            else if (issue) sub <= last_sub ? 3'd0 : sub + 3'd1;
            if (b_drains) begin
                sym_q       <= sym_now;
                // a symbol loaded on the cycle `last` goes in is past the end
                sym_valid_q <= issue && !vit_last;
            end
            if (f_start)       pushed_all <= 1'b0;
            else if (vit_last) pushed_all <= 1'b1;
        end
    end

    wire vit_push = sym_valid_q && vit_ready;
    wire vit_bit, vit_valid, vit_done;

    // `last` on the final coded symbol, or the decoder never flushes its
    // last traceback group. The Viterbi's own counter, reset per frame.
    reg  [15:0] push_cnt;
    wire [15:0] n_sym_total = cfg_encoded_bits >> 1;
    always @(posedge clk) begin
        if (rst || f_start) push_cnt <= 16'd0;
        else if (vit_push)  push_cnt <= push_cnt + 1'b1;
    end
    assign vit_last = vit_push && (push_cnt == n_sym_total - 16'd1);

    viterbi_dec u_vit (
        .clk(clk), .rst(rst), .start(f_start),
        .sym(sym_q), .in_valid(vit_push), .in_ready(vit_ready),
        .last(vit_last),
        .out_bit(vit_bit), .out_valid(vit_valid), .frame_done(vit_done));
    assign frame_done = vit_done;

    // =================================================================
    // Bits -> bytes -> deinterleaver (unit_bits = 8, rs_m8)
    // =================================================================
    reg [7:0] pack;
    reg [3:0] pack_n;
    reg [7:0] di_unit;
    reg       di_pend;             // a byte is waiting for the deinterleaver
    wire      di_ready;

    always @(posedge clk) begin
        if (rst) begin
            pack_n <= 4'd0; di_pend <= 1'b0; st_unit_collision <= 1'b0;
        end else begin
            if (f_start) pack_n <= 4'd0;
            if (di_pend && di_ready) di_pend <= 1'b0;
            if (vit_valid) begin
                pack <= {pack[6:0], vit_bit};
                if (pack_n == 4'd7) begin
                    // Hold until the deinterleaver takes it: its in_ready
                    // drops for one cycle per virtual cell. A byte is made
                    // every >= 8 cycles, so one slot is enough; a new byte
                    // while one still waits is flagged, never dropped silently.
                    if (di_pend && !di_ready) st_unit_collision <= 1'b1;
                    di_unit <= {pack[6:0], vit_bit};
                    di_pend <= 1'b1;
                    pack_n  <= 4'd0;
                end else pack_n <= pack_n + 1'b1;
            end
        end
    end

    deinterleaver #(.UNIT_W(8)) u_di (
        .clk(clk), .rst(rst),
        .n_units(cfg_di_units), .rows(cfg_di_rows), .cols(cfg_di_cols),
        .start(f_start), .in_unit(di_unit), .in_valid(di_pend),
        .in_ready(di_ready), .out_unit(out_byte), .out_valid(out_valid),
        .out_last(out_last));

    // Frame in flight from its start until its last byte is out.
    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0; out_fseq <= 2'd0; st_seq_err <= 1'b0;
        end else begin
            if (f_start) begin
                busy     <= 1'b1;
                out_fseq <= fq_head;
                if (!fq_hv) st_seq_err <= 1'b1;
            end else if (out_valid && out_last) begin
                busy <= 1'b0;
            end
            // DATA must belong to a frame whose config was published valid
            if (acc_dat && in_first && !(cfg_valid && cfg_fseq == in_fseq))
                st_seq_err <= 1'b1;
        end
    end
endmodule
