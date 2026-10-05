`timescale 1ns / 1ps
// Feeds 432-slot headers from +in=FILE (one line per header: 432 bits as
// 0/1 chars) and prints the decoded fields, one line per header.
module header_decode_v3_tb;
    reg clk = 0, rst = 1; always #4 clk = ~clk;
    reg b = 0, v = 0, sof = 0;
    wire done, fv, cok; wire [7:0] ver, mod; wire [15:0] len, c2; wire [3:0] bps;
    wire [2:0] crc; wire [4:0] f0, f1; wire [1:0] dm; wire [47:0] user;
    header_decode_v3 dut(.clk(clk), .rst(rst), .in_bit(b), .in_valid(v), .in_sof(sof),
        .done(done), .fields_valid(fv), .crc_ok(cok), .protocol_version(ver),
        .payload_len_bits(len), .mod_scheme(mod), .bits_per_symbol(bps), .crc_code(crc),
        .fec0_code(f0), .fec1_code(f1), .dmrs_code(dm), .c2_len_bytes(c2), .user_data(user));
    reg [8*432-1:0] line; reg [1023:0] fn; integer fd, i, n = 0, got = 0;
    always @(posedge clk) if (done) begin
        $display("HDR %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %012x", fv, cok, ver, len, mod, crc, f0, f1, dm, c2, user);
        got = got + 1;
    end
    initial begin
        if (!$value$plusargs("in=%s", fn)) $fatal(1, "need +in");
        fd = $fopen(fn, "r"); repeat (4) @(negedge clk); rst = 0; repeat (4) @(negedge clk);
        while ($fscanf(fd, "%s\n", line) == 1) begin
            for (i = 0; i < 432; i = i + 1) begin
                v = 1; sof = (i == 0); b = (line[8*(431-i) +: 8] == "1");
                @(negedge clk); v = 0; sof = 0;
                if (i % 3 == 0) @(negedge clk);   // gaps
            end
            n = n + 1;
            repeat (600) @(negedge clk);
        end
        repeat (1000) @(negedge clk);
        if (got != n) $fatal(1, "decoded %0d of %0d", got, n);
        $display("TB_DONE %0d", n); $finish;
    end
endmodule
