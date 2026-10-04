// ============================================================
// rx_top.v -- the OFDM receiver, antenna samples in, PDU bytes out
//
// Wires the blocks that check_chain.py validated individually and
// pairwise against spectracuda. The chain, and where it stops:
//
//   in -> sc_sync -> frame_sync -> cfo_correct -> cp_fft -> grid_extract
//      -> ls_chanest (training) / mmse_eq (header+payload)
//      -> pilot_cpe (payload) -> demapper -> viterbi -> deinterleaver -> out
//
// Reed-Solomon and CRC are NOT here and never will be: they run on the
// host. See hls/rundown.md section 2 -- the fabric covers everything at
// sample rate, the host takes over where the rate collapses to PDU rate.
//
// FRAME PHASES. frame_sync hands over the frame from its first sample:
//   PREAMBLE  fft_size samples, NO cyclic prefix (it is what sync found)
//   TRAIN     N_TRAINING slots -- all 224 occupied bins are known, so
//             they all feed ls_chanest (CE_N_PILOT is 224, not 8)
//   HEADER    1 slot, BPSK, -> header_decode -> payload geometry
//   PAYLOAD   n slots, modulation taken from the decoded header
//
// H IS RE-SPLIT BY A SECOND grid_extract. ls_chanest emits H for all
// N_FFT bins in natural order -- exactly the shape grid_extract already
// consumes -- so feeding it through another instance yields h_data and
// h_pilot in the same order as the rx data and pilot streams, and the
// equalizer just consumes pairs. No index arithmetic, no third table.
//
// TWO EQUALIZERS, not one. Python equalizes data against h_data and
// pilots against h_pilot as separate calls, and ofdm.py's comment is
// explicit that the pilot estimate genuinely differs per subcarrier
// rather than being h_data reused at other indices. One shared instance
// would have to serialise them; two cost ~287 LUT each and keep the
// payload path streaming.
//
// ---- THREE THINGS THAT ARE NOT YET PROVEN ----------------------------
//
// 1. INTER-STAGE SCALING. Each block was verified at its own Q format
//    (cp_fft emits 32-bit, ls_chanest wants 20, mmse_eq wants 18 Q12).
//    The shifts below reconcile them and are the least-tested thing in
//    this file -- check_chain.py fed every block Python-scaled data, so
//    these particular truncations have never run against the golden
//    model. Suspect them first.
//
// 2. BACK-END RATE. The demapper emits up to 6 bits per clock; the
//    Viterbi consumes one 2-bit symbol per clock. At 1 sample/clock the
//    back end is oversubscribed ~3x at QAM64, which is what BIT_FIFO is
//    for -- 216*6 = 1296 bits per symbol against 288 sample-clocks. It
//    absorbs a symbol, not a frame. At the real 10 Msps / 100 MHz
//    budget there are ~10 clocks per sample and the problem disappears;
//    see the folding analysis in docs/vitis-hls-ofdm-ip-plan.md.
//
// 3. NOTHING HERE HAS BEEN SYNTHESIZED OR SIMULATED END TO END.
//    cp_fft instantiates xfft_256, which needs Xilinx simulation
//    primitives, so this top cannot run under Verilator.
// ============================================================
`timescale 1ns / 1ps
`include "ofdm_params.vh"
`include "grid_params.vh"
`include "header_params.vh"
`include "demap_params.vh"

module rx_top #(
    parameter integer SAMPLE_W  = 16,
    parameter integer ACC_W     = 48,
    parameter integer FFT_W     = 32,
    parameter integer CE_IN_W   = 20,
    parameter integer EQ_W      = 18,
    parameter integer ANGLE_W   = 16,
    parameter integer BUF_W     = 12,
    parameter integer MAX_PAYLOAD_SYM = 128,
    // ---- INTER-STAGE SCALING, NOT YET CALIBRATED ---------------------
    // How far to shift cp_fft's output down before the channel
    // estimator and the equalizer. These are PARAMETERS because the
    // right values depend on xfft_256's actual output scaling
    // (scaling_options=unscaled grows by the full FFT gain) and on the
    // input level -- neither of which can be measured against the
    // behavioural stub in tb/stubs/. Measured with the stub: the payload
    // comes out ~700x small with correlation 0.18, i.e. crushed into
    // quantization noise. The BPSK header still decodes perfectly
    // because it only needs the sign.
    //
    // CALIBRATE THESE UNDER XSIM AGAINST THE REAL CORE before trusting
    // any payload result. Do not tune them against the stub -- that
    // calibrates to the stub, not to the hardware.
    parameter integer SHIFT_FFT_TO_CE = 3,
    parameter integer SHIFT_FFT_TO_EQ = 3
)(
    input  wire                       clk,
    input  wire                       rst,

    // ---- HOST-SUPPLIED FRAME GEOMETRY --------------------------------
    // The header carries payload_len_bits, which is the INFORMATION
    // length. What is actually on the air is the ENCODED length, and
    // deriving it needs the rs_m8 + conv_v27 geometry (RS parity per
    // codeword, rate 1/2, K=7 tail) -- arithmetic that belongs with the
    // FEC, and the FEC lives on the host (rundown.md section 2).
    //
    // So the host supplies it, along with the deinterleaver's grid. The
    // host already owns both: it configures the link and it runs RS.
    // Computing these in fabric instead is a real open item, not a
    // detail -- a receiver that must decode a peer's arbitrary FEC
    // choice from the header alone cannot ask the host first.
    input  wire [15:0]                cfg_encoded_bits,
    input  wire [12:0]                cfg_di_units,
    input  wire [12:0]                cfg_di_rows,
    input  wire [12:0]                cfg_di_cols,

    input  wire signed [SAMPLE_W-1:0] in_i,
    input  wire signed [SAMPLE_W-1:0] in_q,
    input  wire                       in_valid,

    // Decoded header, valid from hdr_valid until the next frame.
    output wire                       hdr_valid,
    output wire [15:0]                payload_len_bits,
    output wire [7:0]                 mod_scheme,
    output wire [4:0]                 fec0_code,
    output wire [4:0]                 fec1_code,
    output wire [2:0]                 crc_code,

    // Deinterleaved payload bytes -> host (RS + CRC happen there).
    output wire [7:0]                 out_unit,
    output wire                       out_unit_valid,
    output wire                       frame_done,
    // Sticky. High means coded bits were dropped and the output is
    // garbage -- check it before trusting a decode.
    output wire                       fifo_overflow,

    // Frequency-domain stage status (sticky errors; any 1 = do not trust
    // the frame): {cfg_unsupported, hdr_no_train, fseq_collision,
    //              seq_err, b2_overflow, b1_overflow}
    output wire [5:0]                 fd_err,
    output wire [11:0]                fd_b1_hwm,    // B1 high-water, of 1280
    output wire [9:0]                 fd_b2_hwm     // B2 high-water, of 512
);
    `include "rx_if.vh"
    localparam integer LLR_W = 1;

    // ---------------------------------------------------------------
    // Stages, wired. Target (step 5): rx_time_domain -> rx_freq_domain
    // -> rx_bit_domain plus the C1 config wires, and nothing else.
    //
    // Today (step 3c) only the frequency domain is the new IP. The time
    // domain and the header/bit blocks are still the pre-refactor ones,
    // joined to it by three TEMPORARY adapters, each marked below.
    // ---------------------------------------------------------------
    wire                     frame_start;
    wire signed [FFT_W-1:0]  fft_re, fft_im;
    wire                     fft_valid, fft_sof;
    wire [1:0]               fft_stype;
    wire [7:0]               fft_bin;

    wire                     hdr_done;
    wire [3:0]               hdr_bps;
    wire [7:0]               n_pay_sym;
    wire                     cfg_final;

    rx_time_domain #(.SAMPLE_W(SAMPLE_W), .ACC_W(ACC_W), .FFT_W(FFT_W),
                     .ANGLE_W(ANGLE_W), .BUF_W(BUF_W),
                     .MAX_PAYLOAD_SYM(MAX_PAYLOAD_SYM)) u_td (
        .clk(clk), .rst(rst),
        .in_i(in_i), .in_q(in_q), .in_valid(in_valid),
        // TEMPORARY until step 4: cfg_valid / n_pay_sym are adapter C's C1.
        .cfg_body_valid(cfg_valid), .cfg_body_syms(n_pay_sym),
        .frame_start(frame_start),
        .fft_re(fft_re), .fft_im(fft_im), .fft_valid(fft_valid),
        .fft_sof(fft_sof), .fft_stype(fft_stype), .fft_bin(fft_bin));

    // =================================================================
    // TEMPORARY STEP-3 ADAPTER A (old TD -> TD->FD interface)
    // REMOVE IN STEP 5: rx_time_domain will emit sym_idx / fseq /
    // frame_start itself. Until then they are derived here from the old
    // FFT markers: a symbol starts on fft_sof, and a frame starts on its
    // TRAIN symbol (N_TRAINING = 1).
    // =================================================================
    reg  [7:0] a_sym;
    reg  [1:0] a_fseq;
    reg        a_first;
    wire       a_sof       = fft_valid && fft_sof;
    wire       a_new_frame = a_sof && (fft_stype == 2'd0);
    wire [7:0] i1_sym      = a_new_frame ? 8'd0 : (a_sof ? a_sym + 8'd1 : a_sym);
    wire [1:0] i1_fseq     = a_new_frame ? (a_first ? 2'd0 : a_fseq + 2'd1) : a_fseq;
    always @(posedge clk) begin
        if (rst) begin
            a_sym <= 8'd0; a_fseq <= 2'd0; a_first <= 1'b1;
        end else if (a_sof) begin
            a_sym  <= i1_sym;
            a_fseq <= i1_fseq;
            if (a_new_frame) a_first <= 1'b0;
        end
    end

    // ---- C1 bundle (driven by adapter C below) ----
    wire       cfg_valid, cfg_err;
    wire [1:0] cfg_fseq;

    // ---- FD -> BIT interface ----
    wire              fd_out_valid, fd_out_ready;
    wire [6*LLR_W-1:0] fd_out_llr;
    wire [2:0]        fd_out_n, fd_out_stype;
    wire [7:0]        fd_out_sc, fd_out_sym_idx;
    wire [1:0]        fd_out_fseq;
    wire              fd_out_sym_start, fd_out_sym_end, fd_out_frame_start, fd_out_frame_end;

    rx_freq_domain #(.FFT_W(FFT_W), .CE_IN_W(CE_IN_W), .EQ_W(EQ_W),
                     .ANGLE_W(ANGLE_W),
                     .SHIFT_FFT_TO_CE(SHIFT_FFT_TO_CE),
                     .SHIFT_FFT_TO_EQ(SHIFT_FFT_TO_EQ),
                     .LLR_W(LLR_W)) u_fd (
        .clk(clk), .rst(rst),
        .in_valid(fft_valid), .in_re(fft_re), .in_im(fft_im), .in_bin(fft_bin),
        .in_sym_idx(i1_sym), .in_stype({1'b0, fft_stype}), .in_fseq(i1_fseq),
        .in_frame_start(a_new_frame),
        .cfg_valid(cfg_valid), .cfg_err(cfg_err), .cfg_fseq(cfg_fseq),
        .cfg_mod(mod_scheme[2:0]), .cfg_body_syms(n_pay_sym),
        .cfg_c2_syms(8'd0), .cfg_dmrs_period(2'd0),
        .out_valid(fd_out_valid), .out_ready(fd_out_ready), .out_llr(fd_out_llr),
        .out_n(fd_out_n), .out_sc(fd_out_sc), .out_sym_idx(fd_out_sym_idx),
        .out_stype(fd_out_stype), .out_fseq(fd_out_fseq),
        .out_sym_start(fd_out_sym_start), .out_sym_end(fd_out_sym_end),
        .out_frame_start(fd_out_frame_start), .out_frame_end(fd_out_frame_end),
        .st_b1_overflow(fd_err[0]), .st_b2_overflow(fd_err[1]),
        .st_seq_err(fd_err[2]), .st_fseq_collision(fd_err[3]),
        .st_hdr_no_train(fd_err[4]), .st_cfg_unsupported(fd_err[5]),
        .st_hq_held(), .st_b1_level(), .st_b1_hwm(fd_b1_hwm), .st_b2_hwm(fd_b2_hwm));

    // =================================================================
    // TEMPORARY STEP-3 ADAPTER B (FD->BIT interface -> old header/bit blocks)
    // REMOVE IN STEP 4: rx_bit_domain takes the FD->BIT interface
    // directly. The old blocks have no backpressure (the bit FIFO holds a
    // whole frame), so ready is tied high here.
    // =================================================================
    assign fd_out_ready = 1'b1;
    wire   b_hdr = fd_out_valid && (fd_out_stype == ST_HEADER);
    wire   b_dat = fd_out_valid && (fd_out_stype == ST_DATA);
    // hard bit k = sign of llr[k]; the old blocks want them MSB-first
    wire [5:0] b_bits;
    genvar bk;
    generate for (bk = 0; bk < 6; bk = bk + 1) begin : g_b
        assign b_bits[5-bk] = fd_out_llr[bk*LLR_W + LLR_W-1];
    end endgenerate

    rx_header u_hd (
        .clk(clk), .rst(rst), .frame_start(frame_start),
        .hdr_bit(b_bits[5]), .hdr_valid_bit(b_hdr),
        .hdr_bit_sof(b_hdr && fd_out_frame_start),
        .cfg_encoded_bits(cfg_encoded_bits),
        .hdr_valid(hdr_valid), .hdr_done(hdr_done),
        .payload_len_bits(payload_len_bits), .mod_scheme(mod_scheme),
        .hdr_bps(hdr_bps), .fec0_code(fec0_code), .fec1_code(fec1_code),
        .crc_code(crc_code),
        .dm_scheme(), .n_pay_sym(n_pay_sym), .cfg_final(cfg_final));

    rx_bit_decoder #(.EQ_W(EQ_W), .MAX_PAYLOAD_SYM(MAX_PAYLOAD_SYM)) u_bd (
        .clk(clk), .rst(rst), .frame_start(frame_start),
        .dm_bits(b_bits), .dm_nbits({1'b0, fd_out_n}), .dm_valid(b_dat),
        .hdr_done(hdr_done), .hdr_bps(hdr_bps),
        .cfg_encoded_bits(cfg_encoded_bits), .cfg_di_units(cfg_di_units),
        .cfg_di_rows(cfg_di_rows), .cfg_di_cols(cfg_di_cols),
        .out_unit(out_unit), .out_unit_valid(out_unit_valid),
        .frame_done(frame_done), .fifo_overflow(fifo_overflow));

    // =================================================================
    // TEMPORARY STEP-3 ADAPTER C (old header block -> C1 config bundle)
    // REMOVE IN STEP 4: rx_bit_domain's config publisher drives C1.
    // cfg_valid only once every field is FINAL (cfg_final: the header is
    // decoded and n_pay_sym has finished accumulating), tagged with the
    // fseq of the header those fields came from.
    // =================================================================
    reg [1:0] c_fseq;
    always @(posedge clk) begin
        if (rst) c_fseq <= 2'd0;
        else if (b_hdr && fd_out_frame_start) c_fseq <= fd_out_fseq;
    end
    assign cfg_valid = cfg_final && hdr_valid;
    assign cfg_err   = cfg_final && !hdr_valid;
    assign cfg_fseq  = c_fseq;
endmodule
