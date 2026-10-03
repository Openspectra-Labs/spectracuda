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
    output wire                       fifo_overflow
);
    // ---------------------------------------------------------------
    // Four stages, wired. No datapath lives at this level.
    //
    //   rx_time_domain  sync, CFO, frame phase FSM, CP strip + FFT
    //   rx_freq_domain  grid split, chan est, equalize, CPE, header bits
    //   rx_header       header decode -> scheme, payload length
    //   rx_bit_decoder  demap, Viterbi, deinterleave -> bytes
    // ---------------------------------------------------------------
    wire                     frame_start;
    wire signed [FFT_W-1:0]  fft_re, fft_im;
    wire                     fft_valid, fft_sof;
    wire [1:0]               fft_stype;
    wire [7:0]               fft_bin;

    wire                     hdr_bit, hdr_valid_bit, hdr_bit_sof;
    wire signed [EQ_W-1:0]   cpe_re, cpe_im;
    wire                     cpe_out_valid;

    wire                     hdr_done;
    wire [3:0]               hdr_bps;
    wire [1:0]               dm_scheme;
    wire [7:0]               n_pay_sym;

    rx_time_domain #(.SAMPLE_W(SAMPLE_W), .ACC_W(ACC_W), .FFT_W(FFT_W),
                     .ANGLE_W(ANGLE_W), .BUF_W(BUF_W),
                     .MAX_PAYLOAD_SYM(MAX_PAYLOAD_SYM)) u_td (
        .clk(clk), .rst(rst),
        .in_i(in_i), .in_q(in_q), .in_valid(in_valid),
        .n_pay_sym(n_pay_sym),
        .frame_start(frame_start),
        .fft_re(fft_re), .fft_im(fft_im), .fft_valid(fft_valid),
        .fft_sof(fft_sof), .fft_stype(fft_stype), .fft_bin(fft_bin));

    rx_freq_domain #(.FFT_W(FFT_W), .CE_IN_W(CE_IN_W), .EQ_W(EQ_W),
                     .ANGLE_W(ANGLE_W),
                     .SHIFT_FFT_TO_CE(SHIFT_FFT_TO_CE),
                     .SHIFT_FFT_TO_EQ(SHIFT_FFT_TO_EQ)) u_fd (
        .clk(clk), .rst(rst), .frame_start(frame_start),
        .fft_re(fft_re), .fft_im(fft_im), .fft_valid(fft_valid),
        .fft_sof(fft_sof), .fft_stype(fft_stype), .fft_bin(fft_bin),
        .hdr_bit(hdr_bit), .hdr_valid_bit(hdr_valid_bit),
        .hdr_bit_sof(hdr_bit_sof),
        .cpe_re(cpe_re), .cpe_im(cpe_im), .cpe_out_valid(cpe_out_valid));

    rx_header u_hd (
        .clk(clk), .rst(rst), .frame_start(frame_start),
        .hdr_bit(hdr_bit), .hdr_valid_bit(hdr_valid_bit),
        .hdr_bit_sof(hdr_bit_sof),
        .cfg_encoded_bits(cfg_encoded_bits),
        .hdr_valid(hdr_valid), .hdr_done(hdr_done),
        .payload_len_bits(payload_len_bits), .mod_scheme(mod_scheme),
        .hdr_bps(hdr_bps), .fec0_code(fec0_code), .fec1_code(fec1_code),
        .crc_code(crc_code),
        .dm_scheme(dm_scheme), .n_pay_sym(n_pay_sym));

    rx_bit_decoder #(.EQ_W(EQ_W), .MAX_PAYLOAD_SYM(MAX_PAYLOAD_SYM)) u_bd (
        .clk(clk), .rst(rst), .frame_start(frame_start),
        .cpe_re(cpe_re), .cpe_im(cpe_im), .cpe_out_valid(cpe_out_valid),
        .hdr_done(hdr_done), .hdr_bps(hdr_bps), .dm_scheme(dm_scheme),
        .cfg_encoded_bits(cfg_encoded_bits), .cfg_di_units(cfg_di_units),
        .cfg_di_rows(cfg_di_rows), .cfg_di_cols(cfg_di_cols),
        .out_unit(out_unit), .out_unit_valid(out_unit_valid),
        .frame_done(frame_done), .fifo_overflow(fifo_overflow));
endmodule
