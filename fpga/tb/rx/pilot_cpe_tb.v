`timescale 1ns / 1ps
`include "build/cpe_tb_params.vh"

module pilot_cpe_tb;
    localparam integer DATA_W = 18;
    localparam integer NSYM   = `CPE_NSYM;
    localparam integer NDATA  = `CPE_NDATA;
    localparam integer NPIL   = `CPE_NPIL;

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg signed [DATA_W-1:0] d_re, d_im, p_re, p_im;
    reg d_v, p_v, sdone;
    wire signed [DATA_W-1:0] o_re, o_im;
    wire o_v, cpe_v;
    wire signed [15:0] cpe_a;

    pilot_cpe #(.DATA_W(DATA_W)) dut (
        .clk(clk), .rst(rst),
        .data_re(d_re), .data_im(d_im), .data_valid(d_v),
        .pilot_re(p_re), .pilot_im(p_im), .pilot_valid(p_v),
        .sym_done(sdone),
        .out_re(o_re), .out_im(o_im), .out_valid(o_v),
        .cpe_angle(cpe_a), .cpe_valid(cpe_v));

    reg [35:0] dstim [0:NSYM*NDATA-1];
    reg [35:0] pstim [0:NSYM*NPIL-1];
    integer fd, fa, s, i;

    always @(posedge clk) begin
        if (!rst) begin
            if (o_v)   $fwrite(fd, "%0d %0d\n", $signed(o_re), $signed(o_im));
            if (cpe_v) $fwrite(fa, "%0d\n", $signed(cpe_a));
        end
    end

    initial begin
        for (i = 0; i < NSYM*NDATA; i = i + 1) dstim[i] = 36'h0;
        for (i = 0; i < NSYM*NPIL;  i = i + 1) pstim[i] = 36'h0;
        $readmemh(`CPE_DSTIM_PATH, dstim);
        $readmemh(`CPE_PSTIM_PATH, pstim);
        fd = $fopen(`CPE_OUT_PATH, "w");
        fa = $fopen(`CPE_ANGLE_PATH, "w");
        d_v = 0; p_v = 0; sdone = 0; d_re = 0; d_im = 0; p_re = 0; p_im = 0;
        repeat (4) @(posedge clk);
        rst = 0;
        @(posedge clk);

        for (s = 0; s < NSYM; s = s + 1) begin
            for (i = 0; i < NDATA; i = i + 1) begin
                @(negedge clk);
                d_re = $signed(dstim[s*NDATA + i][35:18]);
                d_im = $signed(dstim[s*NDATA + i][17:0]);
                d_v  = 1'b1;
                if (i < NPIL) begin
                    p_re = $signed(pstim[s*NPIL + i][35:18]);
                    p_im = $signed(pstim[s*NPIL + i][17:0]);
                    p_v  = 1'b1;
                end else begin
                    p_v = 1'b0;
                end
            end
            @(negedge clk);
            d_v = 1'b0; p_v = 1'b0; sdone = 1'b1;
            @(negedge clk);
            sdone = 1'b0;
            // Wait out the angle CORDIC plus the whole rotation pass.
            repeat (NDATA + 8*16 + 64) @(posedge clk);
        end
        repeat (8) @(posedge clk);
        $fclose(fd); $fclose(fa);
        $display("pilot_cpe_tb: %0d symbols x %0d data, %0d pilots", NSYM, NDATA, NPIL);
        $finish;
    end
endmodule
