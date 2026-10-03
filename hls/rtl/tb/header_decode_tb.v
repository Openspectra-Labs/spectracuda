`timescale 1ns / 1ps
`include "build/hdr_tb_params.vh"

module header_decode_tb;
    localparam integer NSLOT = `HDRTB_NSLOT;

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg in_bit, in_valid, in_sof;
    wire done, fields_valid;
    wire [7:0]  protocol_version, mod_scheme;
    wire [15:0] payload_len_bits;
    wire [3:0]  bits_per_symbol;
    wire [2:0]  crc_code;
    wire [4:0]  fec0_code, fec1_code;
    wire [63:0] user_data;

    header_decode dut (
        .clk(clk), .rst(rst), .in_bit(in_bit), .in_valid(in_valid),
        .in_sof(in_sof), .done(done), .fields_valid(fields_valid),
        .protocol_version(protocol_version), .payload_len_bits(payload_len_bits),
        .mod_scheme(mod_scheme), .bits_per_symbol(bits_per_symbol),
        .crc_code(crc_code), .fec0_code(fec0_code), .fec1_code(fec1_code),
        .user_data(user_data));

    reg [3:0] stim [0:NSLOT-1];
    integer fd, i;
    reg fired = 0;

    always @(posedge clk) begin
        if (!rst && done && !fired) begin
            fired <= 1'b1;
            $fwrite(fd, "%0d\n", protocol_version);
            $fwrite(fd, "%0d\n", payload_len_bits);
            $fwrite(fd, "%0d\n", mod_scheme);
            $fwrite(fd, "%0d\n", bits_per_symbol);
            $fwrite(fd, "%0d\n", crc_code);
            $fwrite(fd, "%0d\n", fec0_code);
            $fwrite(fd, "%0d\n", fec1_code);
            $fwrite(fd, "%0d\n", user_data);
        end
    end

    // fields_valid lands one cycle after done, so sample it later.
    always @(posedge clk)
        if (fired) $fwrite(fd, "");

    initial begin
        for (i = 0; i < NSLOT; i = i + 1) stim[i] = 4'h0;
        $readmemh(`HDRTB_STIM_PATH, stim);
        fd = $fopen(`HDRTB_OUT_PATH, "w");
        in_valid = 0; in_sof = 0; in_bit = 0;
        repeat (4) @(posedge clk);
        rst = 0;
        @(posedge clk);
        for (i = 0; i < NSLOT; i = i + 1) begin
            @(negedge clk);
            in_bit   = stim[i][0];
            in_valid = 1'b1;
            in_sof   = (i == 0);
        end
        @(negedge clk);
        in_valid = 1'b0; in_sof = 1'b0;
        repeat (6) @(posedge clk);
        $fwrite(fd, "%0d\n", fields_valid);
        $fclose(fd);
        $display("header_decode_tb: %0d slots, fields_valid=%0d", NSLOT, fields_valid);
        $finish;
    end
endmodule
