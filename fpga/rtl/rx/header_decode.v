// ============================================================
// header_decode.v -- 112-bit frame header: descramble and parse
//
// Port of spectracuda's HeaderCodec.decode_bits
// (spectracuda/framing/header.py:148):
//
//     unscrambled  = bits ^ scramble_mask
//     header_bytes = packbits(unscrambled)        MSB-first
//     ... field slicing ...
//
// THE SCRAMBLE MASK IS A ROM. It is numpy PCG64 output
// (default_rng(42).integers(0,2,112)); there is no LFSR that reproduces
// it, and one that merely looked random would decode every header to
// garbage. fpga/gen/emit_rtl.py dumps the live codec's own mask.
//
// Scrambling is not decoration either -- ofdm.py's class docstring
// records the unscrambled header's mostly-repeated content constructively
// interfering into a time-domain PAPR spike, found in development.
//
// VALIDITY IS PART OF THE JOB, not a nicety. decode_bits RAISES on an
// unknown mod/crc/fec code, because a false sync decodes noise into a
// plausible-looking header. Here that becomes `fields_valid`: the codes
// are checked against generated bitmaps of the known schemes. A consumer
// must not act on the fields unless fields_valid is set -- the same
// guard spectracuda's strict_fec_check exists to provide.
//
// Bits arrive MSB-first, matching np.packbits.
//
// INPUT IS THE WHOLE HEADER SYMBOL, not a pre-selected 112 bits. The
// header's content bits are SPREAD across the symbol's 216 data slots
// (np.linspace + unique, ofdm.py:443) for frequency diversity -- part of
// the same PAPR fix as the scrambling. This block owns that selection,
// driven by a generated per-slot map, so nothing upstream has to know
// the header's layout.
// ============================================================
`timescale 1ns / 1ps
`include "header_params.vh"

module header_decode (
    input  wire        clk,
    input  wire        rst,

    input  wire        in_bit,
    input  wire        in_valid,
    input  wire        in_sof,      // first bit of a header

    output reg         done,
    output reg         fields_valid,
    output reg  [7:0]  protocol_version,
    output reg  [15:0] payload_len_bits,
    output reg  [7:0]  mod_scheme,
    output reg  [3:0]  bits_per_symbol,
    output reg  [2:0]  crc_code,
    output reg  [4:0]  fec0_code,
    output reg  [4:0]  fec1_code,
    output reg  [63:0] user_data
);
    localparam integer N_BITS = `HDR_LEN_BITS;
    localparam integer CNT_W  = $clog2(N_BITS + 1);

    localparam integer N_SLOTS = `HDR_TOTAL_SLOTS;
    localparam integer SLOT_W  = $clog2(N_SLOTS);

    // EXPLICIT [0:0], not a bare `reg mask [...]`. Vivado's $readmemh
    // rejects a width-less array with "malformed $readmem task: invalid
    // memory name", silently leaves the ROM unloaded, and then optimises
    // the whole block away -- it synthesized to 0 LUT and looked like it
    // had simply been very efficient. Verilator loads it either way,
    // which is why simulation passed.
    reg [0:0]  mask [0:N_BITS-1];
    reg [0:0]  sel  [0:N_SLOTS-1];
    reg [7:0]  bps_rom [0:`HDR_MOD_CODE_MAX];
    initial begin
        $readmemh(`HDR_MASK_MEM, mask);
        $readmemh(`HDR_SEL_MEM, sel);
        $readmemh(`HDR_BPS_MEM, bps_rom);
    end

    // done_pre fires when the last content bit has been shifted in; the
    // fields are registered from sr on that cycle, so the OUTPUT done is
    // asserted one cycle later. That way `done` high always means the
    // field outputs are already valid -- a consumer that samples on done
    // gets the real header, not the previous cycle's zeros.
    reg              done_pre;
    reg [N_BITS-1:0] sr;      // MSB-first: bit 0 of the header ends up at the top
    reg [CNT_W-1:0]  cnt;     // content bits taken so far
    reg [SLOT_W-1:0] slot;    // position within the header symbol

    // Byte k of packbits() is bits [8k .. 8k+7], MSB first.
    function [7:0] hbyte(input integer k);
        integer j;
        begin
            for (j = 0; j < 8; j = j + 1)
                hbyte[7-j] = sr[N_BITS-1 - (8*k + j)];
        end
    endfunction

    wire [7:0]  b0 = hbyte(0);
    wire [7:0]  b1 = hbyte(1);
    wire [7:0]  b2 = hbyte(2);
    wire [7:0]  b3 = hbyte(3);
    wire [7:0]  b4 = hbyte(4);
    wire [7:0]  b5 = hbyte(5);

    wire [2:0]  w_crc  = b4[7:5];
    wire [4:0]  w_fec0 = b4[4:0];
    wire [4:0]  w_fec1 = b5[4:0];

    // Bound to wires first: a bit-select cannot be applied directly to
    // the literal a macro expands to.
    wire [31:0] MOD_VALID = `HDR_MOD_VALID;
    wire [7:0]  CRC_VALID = `HDR_CRC_VALID;
    wire [31:0] FEC_VALID = `HDR_FEC_VALID;

    wire mod_ok  = (b3 <= `HDR_MOD_CODE_MAX) && MOD_VALID[b3[4:0]];
    wire crc_ok  = CRC_VALID[w_crc];
    wire fec0_ok = FEC_VALID[w_fec0];
    wire fec1_ok = FEC_VALID[w_fec1];

    integer i;
    always @(posedge clk) begin
        if (rst) begin
            sr           <= {N_BITS{1'b0}};
            cnt          <= {CNT_W{1'b0}};
            slot         <= {SLOT_W{1'b0}};
            done_pre     <= 1'b0;
            done         <= 1'b0;
            fields_valid <= 1'b0;
        end else begin
            done_pre <= 1'b0;
            done     <= 1'b0;
            if (in_valid) begin
                if (in_sof) begin
                    // Slot 0 is itself a content bit in every layout we
                    // generate, but check it rather than assume.
                    slot <= SLOT_W'(1);
                    if (sel[0]) begin
                        sr  <= {sr[N_BITS-2:0], in_bit ^ mask[0]};
                        cnt <= CNT_W'(1);
                    end else begin
                        cnt <= {CNT_W{1'b0}};
                    end
                end else begin
                    slot <= (slot == SLOT_W'(N_SLOTS-1)) ? {SLOT_W{1'b0}}
                                                         : slot + 1'b1;
                    if (sel[slot]) begin
                        // mask[] is indexed by position within the HEADER,
                        // not within the symbol -- so it follows cnt.
                        sr  <= {sr[N_BITS-2:0], in_bit ^ mask[cnt]};
                        cnt <= (cnt == CNT_W'(N_BITS-1)) ? {CNT_W{1'b0}}
                                                         : cnt + 1'b1;
                        if (cnt == CNT_W'(N_BITS-1))
                            done_pre <= 1'b1;
                    end
                end
            end

            if (done_pre) begin
                done             <= 1'b1;
                protocol_version <= b0;
                payload_len_bits <= {b1, b2};
                mod_scheme       <= b3;
                bits_per_symbol  <= mod_ok ? bps_rom[b3[2:0]][3:0] : 4'd0;
                crc_code         <= w_crc;
                fec0_code        <= w_fec0;
                fec1_code        <= w_fec1;
                for (i = 0; i < 8; i = i + 1)
                    user_data[8*(7-i) +: 8] <= hbyte(6 + i);
                fields_valid     <= mod_ok && crc_ok && fec0_ok && fec1_ok;
            end
        end
    end
endmodule
