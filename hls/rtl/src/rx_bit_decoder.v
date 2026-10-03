`timescale 1ns / 1ps
`include "ofdm_params.vh"
`include "grid_params.vh"
`include "header_params.vh"
`include "demap_params.vh"

// ============================================================
// rx_bit_decoder.v -- symbols in, deinterleaved bytes out
//
//   symbols -> demapper -> coded-bit FIFO -> skid buffer -> viterbi_dec
//           -> bit->byte packer -> deinterleaver -> bytes to host
//
// RS and CRC are NOT here, by decision -- they run on the host
// (hls/rundown.md section 2). Viterbi comes BEFORE the deinterleaver:
// this is a concatenated code.
//
// Moved verbatim out of rx_top.v (section 10).
// ============================================================
module rx_bit_decoder #(
    parameter integer EQ_W            = 18,
    parameter integer MAX_PAYLOAD_SYM = 128
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   frame_start,

    input  wire signed [EQ_W-1:0] cpe_re,
    input  wire signed [EQ_W-1:0] cpe_im,
    input  wire                   cpe_out_valid,

    // From rx_header.
    input  wire                   hdr_done,
    input  wire [3:0]             hdr_bps,
    input  wire [1:0]             dm_scheme,

    // Host-supplied geometry (see rx_top.v).
    input  wire [15:0]            cfg_encoded_bits,
    input  wire [12:0]            cfg_di_units,
    input  wire [12:0]            cfg_di_rows,
    input  wire [12:0]            cfg_di_cols,

    output wire [7:0]             out_unit,
    output wire                   out_unit_valid,
    output wire                   frame_done,
    // Sticky; high means coded bits were dropped.
    output reg                    fifo_overflow
);
    localparam integer N_DATA = `N_DATA;

    // ---------------------------------------------------------------
    // 10. Payload: demap -> bit FIFO -> Viterbi -> deinterleave
    // ---------------------------------------------------------------
    wire [5:0] dm_bits;
    wire [3:0] dm_nbits;
    wire dm_valid;

    demapper #(.W(EQ_W)) u_dm (
        .clk(clk), .rst(rst),
        .y_re(cpe_re), .y_im(cpe_im), .mod_scheme(dm_scheme),
        .in_valid(cpe_out_valid),
        .bits(dm_bits), .n_bits(dm_nbits), .out_valid(dm_valid));

    // ---- coded-bit FIFO, WORD-WIDE ---------------------------------
    // Stores one demapper output per entry, NOT one bit. A 1-bit-wide
    // array written 6 bits at a time is a six-write-port memory: it
    // simulates fine in Verilator and Vivado refuses it outright
    // ("Unable to infer a block/distributed RAM ... memory pattern not
    // supported"). One write per cycle into a 6-bit word is an ordinary
    // simple-dual-port RAM.
    //
    // n_bits is always EVEN (2/4/6), so a word holds a whole number of
    // rate-1/2 symbols and no symbol ever straddles two words -- the
    // read side needs a sub-counter, not a barrel shifter.
    //
    // Depth is a whole frame's worth of subcarriers because the demapper
    // outruns the Viterbi for the entire frame at 1 sample/clock: it
    // emits a word every payload subcarrier while the Viterbi retires
    // one 2-bit symbol per clock. At the real ~10 clocks/sample budget
    // this shrinks to almost nothing.
    localparam integer FIFO_DEPTH = MAX_PAYLOAD_SYM * N_DATA;
    localparam integer FIFO_AW    = $clog2(FIFO_DEPTH);

    reg  [5:0] bit_fifo [0:FIFO_DEPTH-1];
    reg  [FIFO_AW-1:0] f_wr, f_rd;
    reg  [2:0] sub;                       // symbol index within a word
    reg  [3:0] rd_nbits;                  // n_bits of the word being read

    wire [FIFO_AW-1:0] f_used = f_wr - f_rd;

    always @(posedge clk) begin
        if (rst || frame_start) begin
            f_wr          <= {FIFO_AW{1'b0}};
            fifo_overflow <= 1'b0;
        end else if (dm_valid) begin
            bit_fifo[f_wr] <= dm_bits;
            f_wr           <= f_wr + 1'b1;
            if (f_used > FIFO_AW'(FIFO_DEPTH - 2))
                fifo_overflow <= 1'b1;
        end
    end

    // ---- FIFO read side: skid-buffered into the Viterbi -------------
    // Without a register here the path runs FIFO-RAM -> bit-select mux
    // -> Viterbi -> its traceback SRAM, memory at both ends and nothing
    // in between: post-route WNS -4.846 ns, the worst path in the whole
    // design.
    //
    // It has to be a proper single-entry SKID BUFFER, not just a
    // register. viterbi_dec applies backpressure with in_ready, so a
    // naive pipeline register would drop a symbol (or repeat one) every
    // time it stalls. The rule is: only load when the holding register
    // is empty or is being drained this cycle.
    reg  [5:0] rd_word_q;
    reg  [2:0] sub_q;
    reg        word_q_valid;

    wire       word_avail = (f_wr != f_rd);
    wire       vit_ready;

    // Stage A drains when stage B can take it; stage B drains when the
    // Viterbi can.
    wire       b_drains = sym_valid_q ? vit_ready : 1'b1;
    wire       a_load   = word_avail && (!word_q_valid || b_drains);

    reg  [1:0] sym_q;
    reg        sym_valid_q;

    // Bits leave the demapper MSB-first (bits[5] is the first bit), and
    // viterbi_dec takes sym[0] = first bit of the pair.
    wire [2:0] hi_q    = 3'd5 - {sub_q[1:0], 1'b0};
    wire [1:0] sym_now = {rd_word_q[hi_q-3'd1], rd_word_q[hi_q]};

    always @(posedge clk) begin
        if (rst || frame_start) begin
            f_rd         <= {FIFO_AW{1'b0}};
            sub          <= 3'd0;
            rd_nbits     <= 4'd6;
            rd_word_q    <= 6'd0;
            sub_q        <= 3'd0;
            word_q_valid <= 1'b0;
            sym_q        <= 2'd0;
            sym_valid_q  <= 1'b0;
        end else begin
            if (hdr_done) rd_nbits <= hdr_bps;

            // --- stage A: read one symbol's worth out of the FIFO ---
            if (a_load) begin
                rd_word_q    <= bit_fifo[f_rd];
                sub_q        <= sub;
                word_q_valid <= 1'b1;
                if (sub == (rd_nbits[3:1] - 3'd1)) begin
                    sub  <= 3'd0;
                    f_rd <= f_rd + 1'b1;
                end else begin
                    sub <= sub + 1'b1;
                end
            end else if (b_drains) begin
                word_q_valid <= 1'b0;
            end

            // --- stage B: extract the 2-bit symbol, hold for the Viterbi ---
            if (b_drains) begin
                sym_q       <= sym_now;
                sym_valid_q <= word_q_valid;
            end
        end
    end

    wire [1:0] vit_sym  = sym_q;
    wire       vit_push = sym_valid_q && vit_ready;

    wire vit_bit, vit_valid, vit_done;

    // `last` must be asserted on the final coded symbol, or the decoder
    // never flushes its last traceback group: measured 2520 bits out
    // where the frame carried 2528, which left the deinterleaver one
    // unit short of n_units and so emitting nothing at all.
    reg [15:0] push_cnt;
    wire [15:0] n_sym_total = cfg_encoded_bits >> 1;
    always @(posedge clk) begin
        if (rst || frame_start) push_cnt <= 16'd0;
        else if (vit_push)   push_cnt <= push_cnt + 1'b1;
    end
    wire vit_last = vit_push && (push_cnt == n_sym_total - 16'd1);

    viterbi_dec u_vit (
        .clk(clk), .rst(rst), .start(hdr_done),
        .sym(vit_sym), .in_valid(vit_push), .in_ready(vit_ready),
        .last(vit_last),
        .out_bit(vit_bit), .out_valid(vit_valid), .frame_done(vit_done));

    // Viterbi bits -> bytes for the deinterleaver (unit_bits=8, rs_m8).
    reg [7:0] pack;
    reg [3:0] pack_n;
    reg [7:0] di_unit;
    reg       di_valid;
    always @(posedge clk) begin
        if (rst || frame_start) begin pack_n <= 4'd0; di_valid <= 1'b0; end
        else begin
            di_valid <= 1'b0;
            if (vit_valid) begin
                pack <= {pack[6:0], vit_bit};
                if (pack_n == 4'd7) begin
                    di_unit  <= {pack[6:0], vit_bit};
                    di_valid <= 1'b1;
                    pack_n   <= 4'd0;
                end else pack_n <= pack_n + 1'b1;
            end
        end
    end

    deinterleaver #(.UNIT_W(8)) u_di (
        .clk(clk), .rst(rst),
        .n_units(cfg_di_units), .rows(cfg_di_rows), .cols(cfg_di_cols),
        .start(hdr_done), .in_unit(di_unit), .in_valid(di_valid),
        .in_ready(), .out_unit(out_unit), .out_valid(out_unit_valid),
        .out_last());

    assign frame_done = vit_done;
endmodule
