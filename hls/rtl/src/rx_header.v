`timescale 1ns / 1ps
`include "ofdm_params.vh"
`include "grid_params.vh"
`include "header_params.vh"
`include "demap_params.vh"

// ============================================================
// rx_header.v -- header bits in, frame geometry out
//
//   header bits -> header_decode -> fields
//                                -> demapper scheme for the payload
//                                -> payload length in symbols
//
// Control, not datapath: everything here happens once per frame. The
// payload symbol count goes back to rx_time_domain's phase FSM; the
// scheme and bits-per-symbol go to rx_bit_decoder.
//
// Moved verbatim out of rx_top.v (section 9, second half).
// ============================================================
module rx_header (
    input  wire        clk,
    input  wire        rst,
    input  wire        frame_start,

    input  wire        hdr_bit,
    input  wire        hdr_valid_bit,
    input  wire        hdr_bit_sof,

    // Host-supplied: encoded payload length (see rx_top.v).
    input  wire [15:0] cfg_encoded_bits,

    output wire        hdr_valid,
    output wire        hdr_done,
    output wire [15:0] payload_len_bits,
    output wire [7:0]  mod_scheme,
    output wire [3:0]  hdr_bps,
    output wire [4:0]  fec0_code,
    output wire [4:0]  fec1_code,
    output wire [2:0]  crc_code,

    output reg  [1:0]  dm_scheme,
    output reg  [7:0]  n_pay_sym
);
    localparam integer N_DATA = `N_DATA;

    wire [7:0] hdr_ver;
    wire [63:0] hdr_user;

    header_decode u_hdr (
        .clk(clk), .rst(rst),
        .in_bit(hdr_bit), .in_valid(hdr_valid_bit),
        .in_sof(hdr_bit_sof),
        .done(hdr_done), .fields_valid(hdr_valid),
        .protocol_version(hdr_ver), .payload_len_bits(payload_len_bits),
        .mod_scheme(mod_scheme), .bits_per_symbol(hdr_bps),
        .crc_code(crc_code), .fec0_code(fec0_code), .fec1_code(fec1_code),
        .user_data(hdr_user));

    // mod_scheme (header codes: bpsk0 qpsk1 qam16_2 qam64_3) -> demapper
    // codes (qpsk0 qam16_1 qam64_2). Header BPSK never reaches here.
    // Bound to a localparam: a bit-select cannot be applied to the
    // literal a macro expands to.
    localparam [1:0] DM_QPSK_C = 2'(`DM_QPSK);
    always @(posedge clk) begin
        if (rst) dm_scheme <= DM_QPSK_C;
        else if (hdr_done)
            dm_scheme <= (mod_scheme >= 8'd1) ? 2'(mod_scheme - 8'd1)
                                              : DM_QPSK_C;
    end

    // Payload symbol count = ceil(encoded_bits / (N_DATA * bps)).
    //
    // COMPUTED BY ACCUMULATION, NOT DIVISION. Writing this as `/` cost a
    // 96-deep CARRY4 chain and a 56.19 ns combinational path -- WNS
    // -46.18 ns on a 10 ns clock, by far the worst path in the design,
    // and most of the top level's LUTs. A divider is exactly what this
    // project pushes to the host everywhere else (section 2 of
    // hls/rundown.md); there was no reason for one here.
    //
    // The loop runs at most MAX_PAYLOAD_SYM iterations, one per clock,
    // starting at hdr_done. The result is not needed until the end of
    // the first payload symbol, ~288 clocks later, so it is always ready
    // in time.
    reg [31:0] sym_acc;
    reg        counting;
    wire [31:0] bits_per_sym = 32'(N_DATA) * 32'(hdr_bps);

    always @(posedge clk) begin
        if (rst || frame_start) begin
            n_pay_sym <= 8'd1;
            sym_acc   <= 32'd0;
            counting  <= 1'b0;
        end else if (hdr_done) begin
            n_pay_sym <= 8'd0;
            sym_acc   <= 32'd0;
            counting  <= (hdr_bps != 4'd0);
        end else if (counting) begin
            if (sym_acc >= 32'(cfg_encoded_bits)) begin
                counting <= 1'b0;
            end else begin
                sym_acc   <= sym_acc + bits_per_sym;
                n_pay_sym <= n_pay_sym + 1'b1;
            end
        end
    end
endmodule
