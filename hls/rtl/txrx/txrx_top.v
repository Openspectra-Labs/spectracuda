// ============================================================
// txrx_top.v -- complete v3 OFDM modem: RX chain + TX chain on one part
//
// Two clocks, as on the Spectra M.2 board:
//   clk_sample  100 MHz  RX TD/FD, TX FD/TD (IFFT), the AD9361 sample side
//   clk_bit     125 MHz  RX bit domain (Viterbi), TX bit domain (encoder)
// One asynchronous reset input, synchronized per domain inside each chain
// (rx_top takes a clk-synchronous reset; tx_top synchronizes arst itself).
// Frame format: v3 (protected header, interleaver2, DMRS), golden_ref_v3.
// ============================================================
`timescale 1ns / 1ps
module txrx_top (
    input  wire               clk_sample,
    input  wire               clk_bit,
    input  wire               arst,
    // ---- RX: AD9361 samples in (clk_sample) ----
    input  wire signed [15:0] rx_i, rx_q,
    input  wire               rx_valid,
    // RX payload geometry, host-supplied (H8)
    input  wire [15:0]        rx_cfg_encoded_bits,
    input  wire [12:0]        rx_cfg_di_units, rx_cfg_di_rows, rx_cfg_di_cols,
    // ---- RX: decoded header + bytes out (clk_bit) ----
    output wire               rx_hdr_valid,
    output wire [15:0]        rx_payload_len_bits,
    output wire [7:0]         rx_mod_scheme,
    output wire [7:0]         rx_unit,
    output wire               rx_unit_valid,
    output wire               rx_frame_done,
    output wire               rx_fifo_overflow,
    output wire [5:0]         rx_fd_err,
    output wire [1:0]         rx_bd_err,
    // ---- TX: host frame in (clk_bit) ----
    input  wire               tx_cfg_valid,
    output wire               tx_cfg_ready,
    input  wire [15:0]        tx_cfg_payload_bits,
    input  wire [2:0]         tx_cfg_mod,
    input  wire [1:0]         tx_cfg_dmrs,
    input  wire [47:0]        tx_cfg_user,
    input  wire [1:0]         tx_cfg_fseq,
    input  wire               tx_in_valid,
    output wire               tx_in_ready,
    input  wire [7:0]         tx_in_byte,
    input  wire               tx_in_last,
    // ---- TX: AD9361 samples out (clk_sample) ----
    input  wire               tx_sample_ce,
    output wire               tx_out_valid,
    output wire signed [15:0] tx_out_i, tx_out_q,
    output wire               tx_out_active, tx_out_frame_start, tx_out_frame_end,
    output wire [7:0]         tx_status,
    output wire [15:0]        tx_abort_count
);
    // RX reset: asynchronous assert, release synchronized to clk_sample
    (* ASYNC_REG = "TRUE" *) reg [1:0] rx_rst_sync;
    always @(posedge clk_sample or posedge arst)
        if (arst) rx_rst_sync <= 2'b11; else rx_rst_sync <= {rx_rst_sync[0], 1'b0};

    wire [4:0] rx_fec0, rx_fec1; wire [2:0] rx_crc;
    wire [11:0] rx_b1_hwm; wire [9:0] rx_b2_hwm; wire [15:0] rx_cb_hwm;
    rx_top u_rx (
        .clk(clk_sample), .rst(rx_rst_sync[1]), .clk_bd(clk_bit),
        .cfg_encoded_bits(rx_cfg_encoded_bits), .cfg_di_units(rx_cfg_di_units),
        .cfg_di_rows(rx_cfg_di_rows), .cfg_di_cols(rx_cfg_di_cols),
        .in_i(rx_i), .in_q(rx_q), .in_valid(rx_valid),
        .hdr_valid(rx_hdr_valid), .payload_len_bits(rx_payload_len_bits),
        .mod_scheme(rx_mod_scheme), .fec0_code(rx_fec0), .fec1_code(rx_fec1), .crc_code(rx_crc),
        .out_unit(rx_unit), .out_unit_valid(rx_unit_valid), .frame_done(rx_frame_done),
        .fifo_overflow(rx_fifo_overflow), .fd_err(rx_fd_err), .fd_b1_hwm(rx_b1_hwm),
        .fd_b2_hwm(rx_b2_hwm), .bd_err(rx_bd_err), .bd_cb_hwm(rx_cb_hwm));

    wire [1:0] tx_fq_unused;
    wire tx_bit_done, st_be, st_fe, st_seq, st_und, st_clip, st_ifft;
    tx_top u_tx (
        .clk_bit(clk_bit), .clk_sample(clk_sample), .arst(arst), .sample_ce(tx_sample_ce),
        .cfg_valid(tx_cfg_valid), .cfg_ready(tx_cfg_ready), .cfg_payload_bits(tx_cfg_payload_bits),
        .cfg_mod(tx_cfg_mod), .cfg_dmrs(tx_cfg_dmrs), .cfg_user(tx_cfg_user), .cfg_fseq(tx_cfg_fseq),
        .in_valid(tx_in_valid), .in_ready(tx_in_ready), .in_byte(tx_in_byte), .in_last(tx_in_last),
        .out_valid(tx_out_valid), .out_i(tx_out_i), .out_q(tx_out_q), .out_active(tx_out_active),
        .out_frame_start(tx_out_frame_start), .out_frame_end(tx_out_frame_end), .out_fseq(tx_fq_unused),
        .bit_done(tx_bit_done), .st_bit_error(st_be), .st_freq_error(st_fe),
        .st_sequence_error(st_seq), .st_underrun(st_und), .st_clipped(st_clip),
        .st_ifft_error(st_ifft), .st_abort_count(tx_abort_count));
    // st_bit_error is clk_bit; the rest are clk_sample (sticky flags)
    assign tx_status = {1'b0, tx_bit_done, st_ifft, st_clip, st_und, st_seq, st_fe, st_be};
endmodule
