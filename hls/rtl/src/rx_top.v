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
    input  wire                       clk,      // TD + FD (100 MHz target)
    input  wire                       rst,
    input  wire                       clk_bd,   // bit domain (125 MHz target), asynchronous to clk

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
    output wire [9:0]                 fd_b2_hwm,    // B2 high-water, of 512
    // Bit-domain status: {seq_err, unit_collision}; coded-bit FIFO high-water
    output wire [1:0]                 bd_err,
    output wire [15:0]                bd_cb_hwm
);
    `include "rx_if.vh"
    localparam integer LLR_W = 1;

    // ---------------------------------------------------------------
    // Stages, wired. Target (step 5): rx_time_domain -> rx_freq_domain
    // -> rx_bit_domain plus the C1 config wires, and nothing else.
    //
    // Since step 4b the frequency and bit domains are the new IP blocks,
    // wired directly. The time domain is still the pre-refactor one, joined
    // by the one remaining TEMPORARY adapter (A), removed in step 5.
    // ---------------------------------------------------------------
    wire                     frame_start;
    wire signed [FFT_W-1:0]  fft_re, fft_im;
    wire                     fft_valid, fft_sof;
    wire [1:0]               fft_stype;
    wire [7:0]               fft_bin;

    wire                     td_cfg_valid;
    wire [7:0]               cfg_body_syms;

    rx_time_domain #(.SAMPLE_W(SAMPLE_W), .ACC_W(ACC_W), .FFT_W(FFT_W),
                     .ANGLE_W(ANGLE_W), .BUF_W(BUF_W),
                     .MAX_PAYLOAD_SYM(MAX_PAYLOAD_SYM)) u_td (
        .clk(clk), .rst(rst),
        .in_i(in_i), .in_q(in_q), .in_valid(in_valid),
        // C1 frame length, qualified to TD's own frame by adapter A below.
        .cfg_body_valid(td_cfg_valid), .cfg_body_syms(cfg_body_syms),
        .frame_start(frame_start),
        .fft_re(fft_re), .fft_im(fft_im), .fft_valid(fft_valid),
        .fft_sof(fft_sof), .fft_stype(fft_stype), .fft_bin(fft_bin));

    // =================================================================
    // TEMPORARY STEP-3 ADAPTER A (old TD -> TD->FD interface)
    // REMOVE IN STEP 5: rx_time_domain will emit sym_idx / fseq /
    // frame_start itself, and qualify C1 by its own fseq (td_fseq above). Until then they are derived here from the old
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

    // ---- C1 bundle, from rx_bit_domain ----
    wire       cfg_valid, cfg_err;
    wire [1:0] cfg_fseq;
    wire [2:0] cfg_mod;
    wire [7:0] cfg_c2_syms;
    wire [1:0] cfg_dmrs_period;

    // TD's own frame count, for the C1 qualification below. The old TD has
    // no fseq; rx_bit_domain holds a bundle until the NEXT frame's replaces
    // it, so without this the TD could read the previous frame's length.
    reg  [1:0] td_fseq;
    reg        td_first;
    always @(posedge clk) begin
        if (rst) begin td_fseq <= 2'd0; td_first <= 1'b1; end
        else if (frame_start) begin
            td_fseq  <= td_first ? 2'd0 : td_fseq + 2'd1;
            td_first <= 1'b0;
        end
    end
    assign td_cfg_valid = cfg_valid && (cfg_fseq == td_fseq);


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
        .cfg_mod(cfg_mod), .cfg_body_syms(cfg_body_syms),
        .cfg_c2_syms(cfg_c2_syms), .cfg_dmrs_period(cfg_dmrs_period),
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
    // Clock crossing (infrastructure, like an AXIS clock converter in a
    // block design): FD and BD do not know each other's clock.
    // =================================================================
    wire rst_bd;
    cdc_reset_sync u_rst_bd (.clk(clk_bd), .rst_in(rst), .rst_out(rst_bd));

    // FD -> BIT stream: async FIFO, every field crosses together
    localparam integer I2_W = 6*LLR_W + 3 + 8 + 8 + 3 + 2 + 4;
    wire              bd_in_valid, bd_in_ready;
    wire [6*LLR_W-1:0] bd_in_llr;
    wire [2:0]        bd_in_n, bd_in_stype;
    wire [7:0]        bd_in_sc, bd_in_sym_idx;
    wire [1:0]        bd_in_fseq;
    wire              bd_in_ss, bd_in_se, bd_in_fs, bd_in_fe;
    cdc_async_fifo #(.WIDTH(I2_W), .DEPTH(16)) u_i2_cdc (
        .wclk(clk), .wrst(rst), .w_valid(fd_out_valid), .w_ready(fd_out_ready),
        .w_data({fd_out_llr, fd_out_n, fd_out_sc, fd_out_sym_idx, fd_out_stype,
                 fd_out_fseq, fd_out_sym_start, fd_out_sym_end,
                 fd_out_frame_start, fd_out_frame_end}),
        .rclk(clk_bd), .rrst(rst_bd), .r_valid(bd_in_valid), .r_ready(bd_in_ready),
        .r_data({bd_in_llr, bd_in_n, bd_in_sc, bd_in_sym_idx, bd_in_stype,
                 bd_in_fseq, bd_in_ss, bd_in_se, bd_in_fs, bd_in_fe}));

    // C1 config: published on clk_bd, used on clk (FD, TD): atomic bundle
    wire       bd_cfg_valid, bd_cfg_err;
    wire [1:0] bd_cfg_fseq, bd_cfg_dmrs;
    wire [2:0] bd_cfg_mod;
    wire [7:0] bd_cfg_body, bd_cfg_c2;
    cdc_bundle #(.WIDTH(25)) u_c1_cdc (
        .src_clk(clk_bd), .src_rst(rst_bd),
        .src_data({bd_cfg_valid, bd_cfg_err, bd_cfg_fseq, bd_cfg_mod, bd_cfg_body,
                   bd_cfg_c2, bd_cfg_dmrs}),
        .dst_clk(clk), .dst_rst(rst),
        .dst_data({cfg_valid, cfg_err, cfg_fseq, cfg_mod, cfg_body_syms,
                   cfg_c2_syms, cfg_dmrs_period}));

    // =================================================================
    // Bit domain (clk_bd): FD->BIT stream in, C1 out, bytes out.
    // Its byte / header / status outputs are in the clk_bd domain.
    // =================================================================
    rx_bit_domain #(.LLR_W(LLR_W), .MAX_PAYLOAD_SYM(MAX_PAYLOAD_SYM)) u_bd (
        .clk(clk_bd), .rst(rst_bd),
        .in_valid(bd_in_valid), .in_ready(bd_in_ready), .in_llr(bd_in_llr),
        .in_n(bd_in_n), .in_sc(bd_in_sc), .in_sym_idx(bd_in_sym_idx),
        .in_stype(bd_in_stype), .in_fseq(bd_in_fseq),
        .in_sym_start(bd_in_ss), .in_sym_end(bd_in_se),
        .in_frame_start(bd_in_fs), .in_frame_end(bd_in_fe),
        .cfg_encoded_bits(cfg_encoded_bits), .cfg_di_units(cfg_di_units),
        .cfg_di_rows(cfg_di_rows), .cfg_di_cols(cfg_di_cols),
        .cfg_valid(bd_cfg_valid), .cfg_err(bd_cfg_err), .cfg_fseq(bd_cfg_fseq),
        .cfg_mod(bd_cfg_mod), .cfg_body_syms(bd_cfg_body),
        .cfg_c2_syms(bd_cfg_c2), .cfg_dmrs_period(bd_cfg_dmrs),
        .hdr_payload_len_bits(payload_len_bits), .hdr_mod_scheme(mod_scheme),
        .hdr_fec0(fec0_code), .hdr_fec1(fec1_code), .hdr_crc(crc_code),
        .out_valid(out_unit_valid), .out_byte(out_unit), .out_last(),
        .out_fseq(), .frame_done(frame_done),
        .st_cb_overflow(fifo_overflow), .st_unit_collision(bd_err[0]),
        .st_seq_err(bd_err[1]), .st_cb_hwm(bd_cb_hwm));
    assign hdr_valid = bd_cfg_valid;      // clk_bd domain, with the header fields
endmodule
