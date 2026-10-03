`timescale 1ns / 1ps
`include "ofdm_params.vh"
`include "grid_params.vh"
`include "header_params.vh"
`include "demap_params.vh"

// ============================================================
// rx_freq_domain.v -- FFT bins in, equalized + CPE-corrected symbols out
//
//   bins -> grid_extract -> ls_chanest (training) -> grid_extract (H)
//        -> H hold -> mmse_eq x2 (data, pilots) -> pilot_cpe
//        -> payload symbols out
//        -> header: BPSK hard decision -> header bits out
//
// Which symbol a bin belongs to comes from the stype sideband set in
// rx_time_domain, not from a counter here.
//
// Moved verbatim out of rx_top.v (sections 5-9). The two grid_extract
// and two mmse_eq instances are deliberate -- see rx_top.v's header.
// ============================================================
module rx_freq_domain #(
    parameter integer FFT_W     = 32,
    parameter integer CE_IN_W   = 20,
    parameter integer EQ_W      = 18,
    parameter integer ANGLE_W   = 16,
    parameter integer SHIFT_FFT_TO_CE = 3,
    parameter integer SHIFT_FFT_TO_EQ = 3
)(
    input  wire                       clk,
    input  wire                       rst,
    input  wire                       frame_start,

    input  wire signed [FFT_W-1:0]    fft_re,
    input  wire signed [FFT_W-1:0]    fft_im,
    input  wire                       fft_valid,
    input  wire                       fft_sof,
    input  wire [1:0]                 fft_stype,

    // Header symbol: one hard-decided BPSK bit per data subcarrier.
    output wire                       hdr_bit,
    output wire                       hdr_valid_bit,
    output wire                       hdr_bit_sof,

    // Payload symbols, equalized and CPE-corrected, data bins only.
    output wire signed [EQ_W-1:0]     cpe_re,
    output wire signed [EQ_W-1:0]     cpe_im,
    output wire                       cpe_out_valid
);
    `include "rx_stype.vh"
    localparam integer FFT_SIZE   = `FFT_SIZE;
    localparam integer N_DATA     = `N_DATA;
    localparam integer N_PILOT    = `N_PILOT;

    // ---------------------------------------------------------------
    // 5. Grid split (rx bins)
    // ---------------------------------------------------------------
    // ---- FREQUENCY-DOMAIN PHASE ----------------------------------
    // `phase` counts INPUT samples, but cp_fft holds a whole symbol
    // before emitting anything, so by the time the training symbol's
    // bins appear `phase` has already advanced to HEADER. Gating
    // ls_chanest on `phase` meant it was never enabled and the channel
    // estimate was never produced -- found by counting h_valid pulses in
    // simulation, not by inspection.
    //
    // So the frequency-domain side does not look at `phase` at all: it
    // reads the symbol-type tag that cp_fft carried across the core with
    // the data. That is latency-independent -- it does not care how many
    // cycles cp_fft or the core take.
    wire signed [FFT_W-1:0] gd_re, gd_im, gp_re, gp_im;
    wire gd_valid, gp_valid, g_symdone;
    wire [1:0] g_stype;

    wire op_train = (g_stype == ST_TRAIN);
    wire op_hdr   = (g_stype == ST_HDR);
    wire op_pay   = (g_stype == ST_PAY);

    grid_extract #(.DATA_W(FFT_W)) u_grid (
        .clk(clk), .rst(rst),
        .in_re(fft_re), .in_im(fft_im), .in_valid(fft_valid),
        .sof(fft_sof), .in_stype(fft_stype),
        .data_re(gd_re), .data_im(gd_im), .data_valid(gd_valid),
        .pilot_re(gp_re), .pilot_im(gp_im), .pilot_valid(gp_valid),
        .sym_done(g_symdone), .out_stype(g_stype));

    // Every occupied bin, data and pilot merged back in bin order --
    // what ls_chanest wants, since all 224 are known in a training
    // symbol. A bin is never both, so this mux cannot drop one.
    wire known_valid = gd_valid || gp_valid;
    wire signed [FFT_W-1:0] known_re = gd_valid ? gd_re : gp_re;
    wire signed [FFT_W-1:0] known_im = gd_valid ? gd_im : gp_im;

    // ---- scaling, see caveat 1 in the header ----
    localparam integer FFT_TO_CE = SHIFT_FFT_TO_CE;
    localparam integer FFT_TO_EQ = SHIFT_FFT_TO_EQ;
    wire signed [FFT_W-1:0] ce_sh_re = known_re >>> FFT_TO_CE;
    wire signed [FFT_W-1:0] ce_sh_im = known_im >>> FFT_TO_CE;
    wire signed [CE_IN_W-1:0] ce_in_re = ce_sh_re[CE_IN_W-1:0];
    wire signed [CE_IN_W-1:0] ce_in_im = ce_sh_im[CE_IN_W-1:0];

    wire signed [FFT_W-1:0] eqd_sh_re = gd_re >>> FFT_TO_EQ;
    wire signed [FFT_W-1:0] eqd_sh_im = gd_im >>> FFT_TO_EQ;
    wire signed [FFT_W-1:0] eqp_sh_re = gp_re >>> FFT_TO_EQ;
    wire signed [FFT_W-1:0] eqp_sh_im = gp_im >>> FFT_TO_EQ;

    // ---------------------------------------------------------------
    // 6. Channel estimate, then re-split so H lines up with the bins
    // ---------------------------------------------------------------
    wire signed [EQ_W-1:0] h_re, h_im;
    wire h_valid, h_last;

    ls_chanest #(.IN_W(CE_IN_W), .H_W(EQ_W)) u_ce (
        .clk(clk), .rst(rst),
        .pilot_re(ce_in_re), .pilot_im(ce_in_im),
        .pilot_valid(known_valid && op_train),
        .h_re(h_re), .h_im(h_im), .h_valid(h_valid), .h_last(h_last));

    reg [8:0] h_bin;
    always @(posedge clk) begin
        if (rst) h_bin <= 9'd0;
        else if (h_valid) h_bin <= (h_bin == 9'(FFT_SIZE-1)) ? 9'd0 : h_bin + 1'b1;
    end

    wire signed [EQ_W-1:0] hd_re, hd_im, hp_re, hp_im;
    wire hd_valid, hp_valid;

    grid_extract #(.DATA_W(EQ_W)) u_grid_h (
        .clk(clk), .rst(rst),
        .in_re(h_re), .in_im(h_im), .in_valid(h_valid),
        .sof(h_valid && (h_bin == 9'd0)), .in_stype(2'd0),
        .data_re(hd_re), .data_im(hd_im), .data_valid(hd_valid),
        .pilot_re(hp_re), .pilot_im(hp_im), .pilot_valid(hp_valid),
        .sym_done(), .out_stype());

    // Held for the whole frame: one training estimate serves every
    // later symbol, which is what N_TRAINING=1 means.
    reg signed [EQ_W-1:0] h_data_re [0:N_DATA-1];
    reg signed [EQ_W-1:0] h_data_im [0:N_DATA-1];
    reg signed [EQ_W-1:0] h_pil_re  [0:N_PILOT-1];
    reg signed [EQ_W-1:0] h_pil_im  [0:N_PILOT-1];
    localparam integer D_AW = $clog2(N_DATA);
    localparam integer P_AW = $clog2(N_PILOT);
    reg [D_AW-1:0] hd_wr;
    reg [P_AW-1:0] hp_wr;

    always @(posedge clk) begin
        if (rst || frame_start) begin hd_wr <= {D_AW{1'b0}}; hp_wr <= {P_AW{1'b0}}; end
        else begin
            if (hd_valid) begin
                h_data_re[hd_wr] <= hd_re; h_data_im[hd_wr] <= hd_im;
                hd_wr <= hd_wr + 1'b1;
            end
            if (hp_valid) begin
                h_pil_re[hp_wr] <= hp_re; h_pil_im[hp_wr] <= hp_im;
                hp_wr <= hp_wr + 1'b1;
            end
        end
    end

    // ---------------------------------------------------------------
    // 7. Equalize data and pilots against their own estimates
    // ---------------------------------------------------------------
    reg [D_AW-1:0] d_rd;
    reg [P_AW-1:0] p_rd;
    always @(posedge clk) begin
        if (rst || g_symdone) begin d_rd <= {D_AW{1'b0}}; p_rd <= {P_AW{1'b0}}; end
        else begin
            if (gd_valid) d_rd <= d_rd + 1'b1;
            if (gp_valid) p_rd <= p_rd + 1'b1;
        end
    end

    wire eq_en = op_hdr || op_pay;

    wire signed [EQ_W-1:0] eqd_re, eqd_im, eqp_re, eqp_im;
    wire eqd_valid, eqp_valid;

    mmse_eq #(.W(EQ_W)) u_eq_data (
        .clk(clk), .rst(rst),
        .rx_re(eqd_sh_re[EQ_W-1:0]), .rx_im(eqd_sh_im[EQ_W-1:0]),
        .h_re(h_data_re[d_rd]), .h_im(h_data_im[d_rd]),
        .in_valid(gd_valid && eq_en),
        .y_re(eqd_re), .y_im(eqd_im), .y_valid(eqd_valid));

    mmse_eq #(.W(EQ_W)) u_eq_pilot (
        .clk(clk), .rst(rst),
        .rx_re(eqp_sh_re[EQ_W-1:0]), .rx_im(eqp_sh_im[EQ_W-1:0]),
        .h_re(h_pil_re[p_rd]), .h_im(h_pil_im[p_rd]),
        .in_valid(gp_valid && op_pay),
        .y_re(eqp_re), .y_im(eqp_im), .y_valid(eqp_valid));

    // ---- EQUALIZER-DOMAIN PHASE ----------------------------------
    // Third and last phase domain. out_sym is right for the grid, but
    // mmse_eq has its own latency, so its last few outputs for a symbol
    // land after out_sym has already moved on -- that cost 5 of the
    // header's 216 bits. Counting the equalizer's OWN outputs is
    // immune to every latency upstream of it.
    //
    // The equalizer only runs for header and payload, so eq_sym 0 is
    // the header and anything above it is payload.
    reg [D_AW:0] eq_dcnt;
    reg [P_AW:0] eq_pcnt;
    reg [7:0]    eq_sym;

    wire eq_d_last = eqd_valid && (eq_dcnt == (D_AW+1)'(N_DATA - 1));
    wire eq_p_last = eqp_valid && (eq_pcnt == (P_AW+1)'(N_PILOT - 1));
    // Bin 255 is a PILOT, so the last pilot of a symbol arrives AFTER
    // the last data. The symbol is complete when both counts are full.
    // The HEADER symbol has no pilot equalization at all -- Python does
    // not CPE-correct the header, so u_eq_pilot is payload-only. Waiting
    // on a pilot count that never advances left eq_sym stuck at 0 and
    // every symbol was treated as header.
    wire eq_sym_done = eq_is_hdr
        ? eq_d_last
        : ((eq_d_last && (eq_pcnt == (P_AW+1)'(N_PILOT))) ||
           (eq_p_last && (eq_dcnt == (D_AW+1)'(N_DATA))));

    always @(posedge clk) begin
        if (rst || frame_start) begin
            eq_dcnt <= {(D_AW+1){1'b0}};
            eq_pcnt <= {(P_AW+1){1'b0}};
            eq_sym  <= 8'd0;
        end else begin
            if (eqd_valid) eq_dcnt <= eq_dcnt + 1'b1;
            if (eqp_valid) eq_pcnt <= eq_pcnt + 1'b1;
            if (eq_sym_done) begin
                eq_dcnt <= {(D_AW+1){1'b0}};
                eq_pcnt <= {(P_AW+1){1'b0}};
                eq_sym  <= eq_sym + 1'b1;
            end
        end
    end

    wire eq_is_hdr = (eq_sym == 8'd0);
    wire eq_is_pay = (eq_sym != 8'd0);

    // ---------------------------------------------------------------
    // 8. CPE (payload only -- the header has no pilots to track with)
    // ---------------------------------------------------------------

    pilot_cpe #(.DATA_W(EQ_W), .ANGLE_W(ANGLE_W)) u_cpe (
        .clk(clk), .rst(rst),
        .data_re(eqd_re), .data_im(eqd_im),
        .data_valid(eqd_valid && eq_is_pay),
        .pilot_re(eqp_re), .pilot_im(eqp_im), .pilot_valid(eqp_valid),
        .sym_done(eq_sym_done && eq_is_pay),
        .out_re(cpe_re), .out_im(cpe_im), .out_valid(cpe_out_valid),
        .cpe_angle(), .cpe_valid());

    // ---------------------------------------------------------------
    // 9. Header: BPSK, decoded in place
    // ---------------------------------------------------------------
    // demapper.v codes only QPSK/QAM16/QAM64 -- there is no BPSK entry,
    // because BPSK is only ever used for the header. Its hard decision
    // is the sign of the real part, so it is done here rather than
    // widening the demapper's table for one caller.
    assign hdr_bit       = ~eqd_re[EQ_W-1];
    assign hdr_valid_bit = eqd_valid && eq_is_hdr;

    reg hdr_sof;
    reg [8:0] hdr_cnt;
    always @(posedge clk) begin
        if (rst || frame_start) begin hdr_cnt <= 9'd0; hdr_sof <= 1'b0; end
        else if (hdr_valid_bit) begin
            hdr_sof <= (hdr_cnt == 9'd0);
            hdr_cnt <= hdr_cnt + 1'b1;
        end else hdr_sof <= 1'b0;
    end


    assign hdr_bit_sof = hdr_valid_bit && (hdr_cnt == 9'd0);
endmodule
