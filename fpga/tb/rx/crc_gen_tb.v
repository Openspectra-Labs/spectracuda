// CRC generator: streams one message and dumps the resulting key.
// Bit-exact against spectracuda's CRC.generate_key().
`timescale 1ns / 1ps
`include "build/crc_tb_params.vh"

module crc_gen_tb;
    localparam integer N = `CR_N, NMEM = `CR_NMEM;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg        start = 0, data_valid = 0;
    reg  [7:0] data = 0;
    wire [`CRC_OUT_W-1:0] crc_out;

    crc_gen dut (.clk(clk), .rst(rst), .start(start),
                 .data(data), .data_valid(data_valid), .crc_out(crc_out));

    reg [7:0] stim [0:NMEM-1];
    integer fd, i;

    initial begin
        $readmemh(`CR_STIM_PATH, stim);
        fd = $fopen(`CR_OUT_PATH, "w");
        if (fd == 0) begin $display("FAIL: cannot open output"); $finish; end

        repeat (4) @(posedge clk); rst = 0; @(posedge clk);

        if (N == 0) begin
            // Empty message: just reset the register and read it back.
            @(negedge clk); start = 1'b1; data_valid = 1'b0;
            @(negedge clk); start = 1'b0;
        end else begin
            for (i = 0; i < N; i = i + 1) begin
                @(negedge clk);
                data = stim[i];
                data_valid = 1'b1;
                start = (i == 0);
            end
            @(negedge clk); data_valid = 1'b0; start = 1'b0;
        end

        @(posedge clk);
        $fwrite(fd, "%0d\n", crc_out);
        $fclose(fd);
        $display("crc_gen_tb: %0d bytes -> 0x%04x", N, crc_out);
        $finish;
    end
endmodule
