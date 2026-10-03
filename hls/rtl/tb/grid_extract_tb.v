`timescale 1ns / 1ps
`include "build/ge_tb_params.vh"
`include "grid_params.vh"

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

    reg  [7:0] in_bin, in_meta;
    wire [7:0] out_sc, out_meta;

    grid_extract #(.DATA_W(DATA_W), .META_W(8)) dut (
        .clk(clk), .rst(rst), .in_re(in_re), .in_im(in_im),
        .in_valid(in_valid), .in_bin(in_bin), .in_meta(in_meta),
        .data_re(data_re), .data_im(data_im), .data_valid(data_valid),
        .pilot_re(pilot_re), .pilot_im(pilot_im), .pilot_valid(pilot_valid),
        .out_sc(out_sc), .out_meta(out_meta), .sym_done(sym_done));

    // Metadata checks (step 3a): out_sc must equal the item's position in
    // its own stream within the symbol, and out_meta (driven with the
    // symbol number) must come out with that symbol's items.
    integer exp_d = 0, exp_p = 0, exp_sym = 0;
    always @(posedge clk) if (!rst) begin
        if (data_valid) begin
            if (out_sc != exp_d[7:0] || out_meta != exp_sym[7:0])
                $fatal(1, "grid_extract: data sc=%0d meta=%0d, want %0d/%0d", out_sc, out_meta, exp_d, exp_sym);
            exp_d = exp_d + 1;
        end
        if (pilot_valid) begin
            if (out_sc != exp_p[7:0] || out_meta != exp_sym[7:0])
                $fatal(1, "grid_extract: pilot sc=%0d meta=%0d, want %0d/%0d", out_sc, out_meta, exp_p, exp_sym);
            exp_p = exp_p + 1;
        end
        if (sym_done) begin
            if (exp_d != `GRID_N_DATA || exp_p != `GRID_N_PILOT)
                $fatal(1, "grid_extract: symbol ended with %0d data / %0d pilots", exp_d, exp_p);
            exp_d = 0; exp_p = 0; exp_sym = exp_sym + 1;
        end
    end

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
                in_bin   = i[7:0];
                in_meta  = s[7:0];
            end
        @(negedge clk);
        in_valid = 1'b0; sof = 1'b0;
        repeat (4) @(posedge clk);
        $fclose(fd_d); $fclose(fd_p);
        $display("grid_extract_tb: %0d symbols x %0d bins", NSYM, NFFT);
        $finish;
    end
endmodule
