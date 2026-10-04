`timescale 1ns / 1ps
`include "build/rxtop_tb_params.vh"
module rx_top_dbg;
    localparam integer NSAMP = `RXT_NSAMP;
    reg clk = 0, rst = 1;
    always #5 clk = ~clk;
    reg signed [15:0] in_i, in_q; reg in_valid;
    wire hdr_valid, out_unit_valid, frame_done;
    wire fifo_ovf;
    wire [15:0] plen; wire [7:0] mods, ounit;
    wire [4:0] f0, f1; wire [2:0] cc;
    rx_top dut (
        .cfg_encoded_bits(16'd`RXT_ENC_BITS), .cfg_di_units(13'd`RXT_DI_UNITS),
        .cfg_di_rows(13'd`RXT_DI_ROWS), .cfg_di_cols(13'd`RXT_DI_COLS),
        .clk(clk), .rst(rst), .clk_bd(clk), .in_i(in_i), .in_q(in_q), .in_valid(in_valid),
        .hdr_valid(hdr_valid), .payload_len_bits(plen), .mod_scheme(mods),
        .fec0_code(f0), .fec1_code(f1), .crc_code(cc), .out_unit(ounit),
        .out_unit_valid(out_unit_valid), .frame_done(frame_done),
        .fifo_overflow(fifo_ovf));
    reg [31:0] stim [0:NSAMP-1];
    integer i, n_det, n_fsv, n_cor, n_fft, n_gd, n_hv, n_ce, n_eqd, n_cpe, n_dm;
    integer n_push, n_vit, n_di, n_rdy, n_hdone, fq, fe, fu;
    always @(posedge clk) if (!rst) begin
        if (dut.u_td.fs_detected) n_det = n_det + 1;
        if (dut.u_td.fs_valid)    n_fsv = n_fsv + 1;
        if (dut.u_td.cor_valid)   n_cor = n_cor + 1;
        if (dut.u_td.fft_valid)   n_fft = n_fft + 1;
        if (dut.u_fd.gd_valid)    n_gd  = n_gd + 1;
        if (dut.u_fd.ch_valid)     n_ce  = n_ce + 1;
        if ((dut.fd_out_valid && dut.fd_out_stype == 3'd1)) n_hv = n_hv + 1;
        if (dut.u_fd.eqd_valid)   n_eqd = n_eqd + 1;
        if (dut.u_fd.cpe_ov) n_cpe = n_cpe + 1;
        if ((dut.fd_out_valid && dut.fd_out_stype == 3'd3))    n_dm  = n_dm + 1;
        if (out_unit_valid) $fwrite(fu, "%0d\n", ounit);
        if (dut.u_fd.eqd_valid && (dut.u_fd.ed_st == 3'd3)) $fwrite(fe, "%0d %0d\n", $signed(dut.u_fd.eqd_re), $signed(dut.u_fd.eqd_im));
        if (dut.u_fd.cpe_ov) $fwrite(fq, "%0d %0d\n", $signed(dut.u_fd.cpe_re), $signed(dut.u_fd.cpe_im));
        if (dut.u_bd.vit_push)    n_push = n_push + 1;
        if (dut.u_bd.vit_valid)   n_vit = n_vit + 1;
        if (dut.u_bd.di_pend)    n_di  = n_di + 1;
        if (dut.u_bd.vit_ready)   n_rdy = n_rdy + 1;
        if (dut.u_bd.hd_done)    n_hdone = n_hdone + 1;
    end
    initial begin
        n_det=0;n_fsv=0;n_cor=0;n_fft=0;n_gd=0;n_hv=0;n_ce=0;n_eqd=0;n_cpe=0;n_dm=0;
        n_push=0;n_vit=0;n_di=0;n_rdy=0;n_hdone=0;
        for (i=0;i<NSAMP;i=i+1) stim[i]=32'h0;
        $readmemh(`RXT_STIM_PATH, stim);
        fq = $fopen("build/rxtop_cpe.txt", "w");
        fe = $fopen("build/rxtop_eqd.txt", "w");
        fu = $fopen(`RXT_UNIT_PATH, "w");
        in_valid=0; in_i=0; in_q=0;
        repeat (8) @(posedge clk); rst=0; @(posedge clk);
        for (i=0;i<NSAMP;i=i+1) begin
            @(negedge clk); in_i=stim[i][31:16]; in_q=stim[i][15:0]; in_valid=1'b1;
        end
        @(negedge clk); in_valid=1'b0;
        repeat (60000) @(posedge clk);
        $display("det=%0d fft=%0d grid_d=%0d ce=%0d hdrbit=%0d eqd=%0d cpe=%0d dm=%0d",
                 n_det, n_fft, n_gd, n_ce, n_hv, n_eqd, n_cpe, n_dm);
        $display("hdr_done=%0d fifo_wr=%0d fifo_rd=%0d push=%0d vit_rdy=%0d vit_out=%0d di_in=%0d n_pay=%0d",
                 n_hdone, dut.u_bd.cb_level, dut.u_bd.cb_level, n_push, n_rdy, n_vit, n_di, dut.cfg_body_syms);
        $fclose(fq); $fclose(fe); $fclose(fu);
        $finish;
    end
endmodule
