// ============================================================
// crc_gen.v -- CRC generator, byte at a time
//
// Port of spectracuda's CRC.generate_key() (spectracuda/fec/crc.py).
// Constants come from fpga/gen/emit_rtl.py, which DERIVES them from
// spectracuda's own tables and cross-checks one real message before
// emitting -- they are not transcribed from a datasheet, because these
// are not the standard named variants.
//
// THE REGISTER IS 32 BITS FOR EVERY SCHEME, including crc16. liquid's
// crc.c runs all widths through one 32-bit reflected reduction and masks
// only the RESULT to the scheme's width; crc.py's _build_table docstring
// calls that distinction "load-bearing here, not cosmetic". Sizing the
// register to the scheme instead yields a perfectly plausible, wrong
// CRC -- which is exactly what the first version of this file did
// (0xf726 where spectracuda gives 0x5c27), caught only because the
// generator cross-checks rather than trusts.
//
// NO TABLE. Python uses a 256-entry ROM because a lookup beats eight
// conditional shifts in an interpreter; in hardware the eight steps
// unroll into a plain XOR network costing no memory at all. The two are
// exactly equivalent -- the table is BUILT by running that same loop
// (crc.py:113-126).
// ============================================================
`timescale 1ns / 1ps
`include "generated/crc_params.vh"

module crc_gen (
    input  wire                    clk,
    input  wire                    rst,

    input  wire                    start,       // restart the register
    input  wire [7:0]              data,
    input  wire                    data_valid,

    output wire [`CRC_OUT_W-1:0]   crc_out      // complemented and masked
);

    localparam integer REG_W = `CRC_REG_W;
    localparam [REG_W-1:0] POLY = `CRC_POLY;
    localparam [REG_W-1:0] INIT = `CRC_INIT;

    reg [REG_W-1:0] crc;

    function [REG_W-1:0] crc_byte;
        input [REG_W-1:0] c_in;
        input [7:0]       d;
        reg [REG_W-1:0] c;
        integer i;
        begin
            c = c_in ^ {{(REG_W-8){1'b0}}, d};
            for (i = 0; i < 8; i = i + 1)
                c = c[0] ? ((c >> 1) ^ POLY) : (c >> 1);
            crc_byte = c;
        end
    endfunction

    always @(posedge clk) begin
        if (rst)             crc <= INIT;
        else if (start)      crc <= data_valid ? crc_byte(INIT, data) : INIT;
        else if (data_valid) crc <= crc_byte(crc, data);
    end

    // Complement the full 32-bit register, THEN narrow -- the order
    // matters, and it is what crc.py does: (~key) & _REG_MASK & width_mask.
    wire [REG_W-1:0] final_reg = ~crc;
    assign crc_out = final_reg[`CRC_OUT_W-1:0];
endmodule
