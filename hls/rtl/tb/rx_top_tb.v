`timescale 1ns / 1ps
`include "build/rxtop_tb_params.vh"
// Input rate, as a fraction: C = RXT_CPS_NUM / RXT_CPS_DEN clocks per
// sample (e.g. 5/2 = 2.5 for 40 Msps at 100 MHz). Older harnesses
// (exp_*.py) set neither: default 1 sample/clock.
`ifndef RXT_CPS_NUM
`define RXT_CPS_NUM 1
`endif
`ifndef RXT_CPS_DEN
`define RXT_CPS_DEN 1
`endif

module rx_top_tb;
    localparam integer NSAMP = `RXT_NSAMP;

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg signed [15:0] in_i, in_q;
    reg in_valid;

    wire hdr_valid, out_unit_valid, frame_done;
    wire fifo_ovf;
    wire [5:0]  fd_err;
    wire [11:0] fd_b1_hwm;
    wire [9:0]  fd_b2_hwm;
    wire [15:0] payload_len_bits;
    wire [7:0]  mod_scheme, out_unit;
    wire [4:0]  fec0_code, fec1_code;
    wire [2:0]  crc_code;

    rx_top dut (
        .cfg_encoded_bits(16'd`RXT_ENC_BITS), .cfg_di_units(13'd`RXT_DI_UNITS),
        .cfg_di_rows(13'd`RXT_DI_ROWS), .cfg_di_cols(13'd`RXT_DI_COLS),
        .clk(clk), .rst(rst), .in_i(in_i), .in_q(in_q), .in_valid(in_valid),
        .hdr_valid(hdr_valid), .payload_len_bits(payload_len_bits),
        .mod_scheme(mod_scheme), .fec0_code(fec0_code), .fec1_code(fec1_code),
        .crc_code(crc_code), .out_unit(out_unit),
        .out_unit_valid(out_unit_valid), .frame_done(frame_done),
        .fifo_overflow(fifo_ovf),
        .fd_err(fd_err), .fd_b1_hwm(fd_b1_hwm), .fd_b2_hwm(fd_b2_hwm));

    // Optional I1 dump (FFT output, one bin per line) for the
    // rate-invariance check: the same frame must give identical bins at
    // every clocks-per-sample ratio. Set by run_frame.py --dump-i1.
`ifdef RXT_I1_PATH
    integer fi1;
    initial fi1 = $fopen(`RXT_I1_PATH, "w");
    always @(posedge clk)
        if (!rst && dut.fft_valid)
            $fwrite(fi1, "%0d %0d\n", $signed(dut.fft_re), $signed(dut.fft_im));
`endif

`ifdef RXT_CAPTURE_DIR
`include "ofdm_params.vh"
`include "tb/capture_taps.vh"
`endif

    reg [31:0] stim [0:NSAMP-1];
    integer fd, fu, i, nunits;
    reg hdr_seen = 0;

    always @(posedge clk) begin
        if (!rst) begin
            if (hdr_valid && !hdr_seen) begin
                hdr_seen <= 1'b1;
                $fwrite(fd, "%0d %0d %0d %0d %0d\n", payload_len_bits,
                        mod_scheme, crc_code, fec0_code, fec1_code);
            end
            if (out_unit_valid) begin
                $fwrite(fu, "%0d\n", out_unit);
                nunits = nunits + 1;
            end
        end
    end

    initial begin
        nunits = 0;
        for (i = 0; i < NSAMP; i = i + 1) stim[i] = 32'h0;
        $readmemh(`RXT_STIM_PATH, stim);
        fd = $fopen(`RXT_HDR_PATH, "w");
        fu = $fopen(`RXT_UNIT_PATH, "w");
        in_valid = 0; in_i = 0; in_q = 0;
        repeat (8) @(posedge clk);
        rst = 0;
        @(posedge clk);
        // Sample i occupies floor((i+1)C) - floor(iC) clocks: one with
        // in_valid high, the rest idle. Integer C gives a fixed pattern,
        // fractional C (2.5) alternates 2 and 3.
        for (i = 0; i < NSAMP; i = i + 1) begin
            @(negedge clk);
            in_i     = stim[i][31:16];
            in_q     = stim[i][15:0];
            in_valid = 1'b1;
            repeat (((i+1)*`RXT_CPS_NUM)/`RXT_CPS_DEN - (i*`RXT_CPS_NUM)/`RXT_CPS_DEN - 1) begin
                @(negedge clk);
                in_valid = 1'b0;
            end
        end
        @(negedge clk);
        in_valid = 1'b0;
        repeat (`RXT_DRAIN) @(posedge clk);
`ifdef RXT_CAPTURE_DIR
        cap_write_c1;
`endif
        // Status lines BEFORE closing the file. (They used to be written
        // after $fclose, so run_frame.py never saw fifo_overflow at all.)
        $fwrite(fu, "OVF %0d\n", fifo_ovf);
        // frequency-domain stage status: error flags, B1 / B2 high-water
        $fwrite(fu, "FD %0d %0d %0d\n", fd_err, fd_b1_hwm, fd_b2_hwm);
        $fclose(fd); $fclose(fu);
        $display("rx_top_tb: %0d samples, hdr_seen=%0d units=%0d ovf=%0d", NSAMP, hdr_seen, nunits, fifo_ovf);
        $finish;
    end
endmodule
