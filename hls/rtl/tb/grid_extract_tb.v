`timescale 1ns / 1ps
`include "build/ge_tb_params.vh"

module grid_extract_tb;
    localparam integer DATA_W = 32;
    localparam integer NSYM   = `GE_NSYM;
    localparam integer NFFT   = `GE_NFFT;

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg  signed [DATA_W-1:0] in_re, in_im;
    reg  in_valid, sof;
    wire signed [DATA_W-1:0] data_re, data_im, pilot_re, pilot_im;
    wire data_valid, pilot_valid, sym_done;

    grid_extract #(.DATA_W(DATA_W)) dut (
        .clk(clk), .rst(rst), .in_re(in_re), .in_im(in_im),
        .in_valid(in_valid), .sof(sof), .in_stype(2'd0),
        .data_re(data_re), .data_im(data_im), .data_valid(data_valid),
        .pilot_re(pilot_re), .pilot_im(pilot_im), .pilot_valid(pilot_valid),
        .sym_done(sym_done), .out_stype());

    reg [63:0] stim [0:NSYM*NFFT-1];
    integer fd_d, fd_p, i, s;

    always @(posedge clk) begin
        if (!rst) begin
            if (data_valid)  $fwrite(fd_d, "%0d %0d\n", $signed(data_re),  $signed(data_im));
            if (pilot_valid) $fwrite(fd_p, "%0d %0d\n", $signed(pilot_re), $signed(pilot_im));
        end
    end

    initial begin
        for (i = 0; i < NSYM*NFFT; i = i + 1) stim[i] = 64'h0;
        $readmemh(`GE_STIM_PATH, stim);
        fd_d = $fopen(`GE_DATA_PATH, "w");
        fd_p = $fopen(`GE_PILOT_PATH, "w");
        in_valid = 0; sof = 0; in_re = 0; in_im = 0;
        repeat (4) @(posedge clk);
        rst = 0;
        @(posedge clk);
        // Driven on negedge with blocking assignments, matching
        // sc_sync_tb.v -- keeps stimulus away from the sampling edge.
        for (s = 0; s < NSYM; s = s + 1)
            for (i = 0; i < NFFT; i = i + 1) begin
                @(negedge clk);
                in_re    = $signed(stim[s*NFFT + i][63:32]);
                in_im    = $signed(stim[s*NFFT + i][31:0]);
                in_valid = 1'b1;
                sof      = (i == 0);
            end
        @(negedge clk);
        in_valid = 1'b0; sof = 1'b0;
        repeat (4) @(posedge clk);
        $fclose(fd_d); $fclose(fd_p);
        $display("grid_extract_tb: %0d symbols x %0d bins", NSYM, NFFT);
        $finish;
    end
endmodule
