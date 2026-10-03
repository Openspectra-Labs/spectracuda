`timescale 1ns / 1ps
`include "build/fs_tb_params.vh"

module frame_sync_tb;
    localparam integer SAMPLE_W = 16;
    localparam integer ACC_W    = 48;
    localparam integer LAG      = `FS_LAG;
    localparam integer NSAMP    = `FS_NSAMP;

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg signed [SAMPLE_W-1:0] in_i, in_q;
    reg in_valid;

    wire signed [ACC_W-1:0] p_re, p_im, r_sum;
    wire sc_v;

    sc_sync_rtl #(.SAMPLE_W(SAMPLE_W), .LAG(LAG), .ACC_W(ACC_W)) u_sync (
        .clk(clk), .rst(rst), .in_i(in_i), .in_q(in_q), .in_valid(in_valid),
        .p_re(p_re), .p_im(p_im), .r_sum(r_sum), .out_valid(sc_v));

    wire signed [SAMPLE_W-1:0] o_i, o_q;
    wire o_v, f_start, det;
    wire [11:0] sidx;

    frame_sync #(.SAMPLE_W(SAMPLE_W), .ACC_W(ACC_W), .BUF_W(12),
                 .FRAME_LEN(`FS_FRAME_LEN)) u_fs (
        .clk(clk), .rst(rst), .in_i(in_i), .in_q(in_q), .in_valid(in_valid),
        .p_re(p_re), .p_im(p_im), .r_sum(r_sum), .sc_valid(sc_v),
        .out_i(o_i), .out_q(o_q), .out_valid(o_v),
        .frame_start(f_start), .detected(det), .start_index(sidx));

    reg [31:0] stim [0:NSAMP-1];
    integer fd, fm, i;
    reg fired = 0;

    always @(posedge clk) begin
        if (!rst) begin
            if (det && !fired) begin
                fired <= 1'b1;
                $fwrite(fm, "%0d\n", sidx);
            end
            if (o_v) $fwrite(fd, "%0d %0d\n", $signed(o_i), $signed(o_q));
        end
    end

    initial begin
        for (i = 0; i < NSAMP; i = i + 1) stim[i] = 32'h0;
        $readmemh(`FS_STIM_PATH, stim);
        fd = $fopen(`FS_OUT_PATH, "w");
        fm = $fopen(`FS_META_PATH, "w");
        in_valid = 0; in_i = 0; in_q = 0;
        repeat (4) @(posedge clk);
        rst = 0;
        @(posedge clk);
        for (i = 0; i < NSAMP; i = i + 1) begin
            @(negedge clk);
            in_i     = stim[i][31:16];
            in_q     = stim[i][15:0];
            in_valid = 1'b1;
        end
        @(negedge clk);
        in_valid = 1'b0;
        // Let any accepted frame finish replaying.
        repeat (`FS_FRAME_LEN + 512) @(posedge clk);
        $fclose(fd); $fclose(fm);
        $display("frame_sync_tb: %0d samples, detected=%0d start_index=%0d",
                 NSAMP, fired, sidx);
        $finish;
    end
endmodule
