// ============================================================
// rx_bit_domain_tb.v -- standalone test of the bit/FEC stage
//
// Built once (run_bd_stage.py); every scenario is a run with plusargs.
//
//   +i2=FILE    FD->BIT items: fseq sym sc stype n l0..l5 ss se fs fe
//   +o1=FILE    expected output bytes: byte last fseq
//   +c1=FILE    expected config publications, in order:
//               fseq valid err mod body len fec0 fec1 crc
//   +enc= +units= +rows= +cols=   host geometry (H8: one value per run)
//   +cps_num= +cps_den=  pacing: a symbol's 216 groups one per C clocks,
//               then the rest of the 288*C symbol period idle
//   +burst=1    groups back to back (worst case for the coded-bit FIFO)
//   +corrupt=1  invert every header bit (header must be rejected)
//
// Emulates FD's B1: a frame's DATA is held until BD has published a VALID
// config for that fseq; on cfg_err the frame's DATA is never sent.
// Checks every byte / out_last / out_fseq, every published C1 bundle, the
// status flags, and valid/ready (an item is held until accepted).
// ============================================================
`timescale 1ns / 1ps

module rx_bit_domain_tb;
`ifndef TB_LLR_W
`define TB_LLR_W 1
`endif
    // LLR_W = 4: the reference's hard bits are fed as +/-7 LLRs. With equal
    // magnitudes the soft metric is an order-preserving affine map of the
    // hard one, so the bytes must equal the hard reference exactly.
    localparam integer LLR_W = `TB_LLR_W;
    localparam integer MAXI  = 65536;
    localparam integer MAXO  = 16384;

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg         in_valid = 0;
    wire        in_ready;
    reg  [6*LLR_W-1:0] in_llr = 0;
    reg  [2:0]  in_n = 0, in_stype = 0;
    reg  [7:0]  in_sc = 0, in_sym = 0;
    reg  [1:0]  in_fseq = 0;
    reg         in_ss = 0, in_se = 0, in_fs = 0, in_fe = 0;
    reg  [15:0] enc = 0;
    reg  [12:0] units = 0, rows = 0, cols = 0;

    wire        cfg_valid, cfg_err;
    wire [1:0]  cfg_fseq, cfg_dmrs;
    wire [2:0]  cfg_mod;
    wire [7:0]  cfg_body, cfg_c2;
    wire [15:0] h_len;
    wire [7:0]  h_mod;
    wire [4:0]  h_f0, h_f1;
    wire [2:0]  h_crc;
    wire        out_valid, out_last, frame_done;
    wire [7:0]  out_byte;
    wire [1:0]  out_fseq;
    wire        cb_ovf, unit_col, seq_err;
    wire [15:0] cb_hwm;

    rx_bit_domain #(.LLR_W(LLR_W)) dut (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_ready(in_ready), .in_llr(in_llr), .in_n(in_n),
        .in_sc(in_sc), .in_sym_idx(in_sym), .in_stype(in_stype), .in_fseq(in_fseq),
        .in_sym_start(in_ss), .in_sym_end(in_se), .in_frame_start(in_fs),
        .in_frame_end(in_fe),
        .cfg_encoded_bits(enc), .cfg_di_units(units), .cfg_di_rows(rows),
        .cfg_di_cols(cols),
        .cfg_valid(cfg_valid), .cfg_err(cfg_err), .cfg_fseq(cfg_fseq),
        .cfg_mod(cfg_mod), .cfg_body_syms(cfg_body), .cfg_c2_syms(cfg_c2),
        .cfg_dmrs_period(cfg_dmrs),
        .hdr_payload_len_bits(h_len), .hdr_mod_scheme(h_mod),
        .hdr_fec0(h_f0), .hdr_fec1(h_f1), .hdr_crc(h_crc),
        .out_valid(out_valid), .out_byte(out_byte), .out_last(out_last),
        .out_fseq(out_fseq), .frame_done(frame_done),
        .st_cb_overflow(cb_ovf), .st_unit_collision(unit_col),
        .st_seq_err(seq_err), .st_cb_hwm(cb_hwm));

    // ---------------- files ----------------
    reg [1:0] i_fq [0:MAXI-1]; reg [7:0] i_sym [0:MAXI-1], i_sc [0:MAXI-1];
    reg [2:0] i_st [0:MAXI-1], i_n [0:MAXI-1]; reg [5:0] i_llr [0:MAXI-1];
    reg [3:0] i_mk [0:MAXI-1];
    integer ni = 0;
    reg [7:0] o_b [0:MAXO-1]; reg o_l [0:MAXO-1]; reg [1:0] o_fq [0:MAXO-1];
    integer no = 0;
    integer c_fq[0:15], c_v[0:15], c_e[0:15], c_md[0:15], c_bd[0:15],
            c_len[0:15], c_f0[0:15], c_f1[0:15], c_crc[0:15];
    integer nc = 0;
    integer cps_num = 1, cps_den = 1, burst = 0, corrupt = 0, tmp;
    reg [1023:0] fname;

    task load;
        integer fd, r, a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14;
        begin
            if (!$value$plusargs("i2=%s", fname)) $fatal(1, "need +i2=");
            fd = $fopen(fname, "r");
            while (!$feof(fd)) begin
                r = $fscanf(fd, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                            a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14);
                if (r == 15) begin
                    i_fq[ni] = a0; i_sym[ni] = a1; i_sc[ni] = a2; i_st[ni] = a3; i_n[ni] = a4;
                    i_llr[ni] = {a10[0], a9[0], a8[0], a7[0], a6[0], a5[0]};
                    i_mk[ni] = {a11[0], a12[0], a13[0], a14[0]};
                    ni = ni + 1;
                end
            end
            $fclose(fd);
            if (!$value$plusargs("o1=%s", fname)) $fatal(1, "need +o1=");
            fd = $fopen(fname, "r");
            while (!$feof(fd)) begin
                r = $fscanf(fd, "%d %d %d\n", a0, a1, a2);
                if (r == 3) begin o_b[no] = a0; o_l[no] = a1[0]; o_fq[no] = a2; no = no + 1; end
            end
            $fclose(fd);
            if (!$value$plusargs("c1=%s", fname)) $fatal(1, "need +c1=");
            fd = $fopen(fname, "r");
            while (!$feof(fd)) begin
                r = $fscanf(fd, "%d %d %d %d %d %d %d %d %d\n", a0,a1,a2,a3,a4,a5,a6,a7,a8);
                if (r == 9) begin
                    c_fq[nc] = a0; c_v[nc] = a1; c_e[nc] = a2; c_md[nc] = a3; c_bd[nc] = a4;
                    c_len[nc] = a5; c_f0[nc] = a6; c_f1[nc] = a7; c_crc[nc] = a8; nc = nc + 1;
                end
            end
            $fclose(fd);
            void'($value$plusargs("enc=%d", tmp)); enc = tmp;
            void'($value$plusargs("units=%d", tmp)); units = tmp;
            void'($value$plusargs("rows=%d", tmp)); rows = tmp;
            void'($value$plusargs("cols=%d", tmp)); cols = tmp;
            void'($value$plusargs("cps_num=%d", cps_num));
            void'($value$plusargs("cps_den=%d", cps_den));
            void'($value$plusargs("burst=%d", burst));
            void'($value$plusargs("corrupt=%d", corrupt));
            $display("TB: %0d input items, %0d expected bytes, %0d expected configs, C=%0d/%0d burst=%0d corrupt=%0d",
                     ni, no, nc, cps_num, cps_den, burst, corrupt);
        end
    endtask

    integer cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

    // ---------------- C1 monitor ----------------
    integer ci = 0, cfg_errors = 0, last_hdr_acc = -1, cfg_lat_max = 0;
    reg pv = 0, pe = 0; reg [1:0] pf = 0;
    always @(posedge clk) if (!rst) begin
        if ((cfg_valid && !(pv && pf == cfg_fseq)) || (cfg_err && !(pe && pf == cfg_fseq))) begin
            if (last_hdr_acc >= 0 && cyc - last_hdr_acc > cfg_lat_max) cfg_lat_max = cyc - last_hdr_acc;
            if (ci >= nc) begin
                cfg_errors = cfg_errors + 1; $display("TB: EXTRA config publication fseq=%0d", cfg_fseq);
            end else begin
                if (cfg_fseq != c_fq[ci] || cfg_valid != c_v[ci][0] || cfg_err != c_e[ci][0] ||
                    (c_v[ci] && (cfg_mod != c_md[ci] || cfg_body != c_bd[ci] || h_len != c_len[ci] ||
                                 h_f0 != c_f0[ci] || h_f1 != c_f1[ci] || h_crc != c_crc[ci]))) begin
                    cfg_errors = cfg_errors + 1;
                    $display("TB: CONFIG MISMATCH #%0d: got fq=%0d v=%0d e=%0d mod=%0d body=%0d len=%0d f0=%0d f1=%0d crc=%0d | want fq=%0d v=%0d e=%0d mod=%0d body=%0d len=%0d f0=%0d f1=%0d crc=%0d",
                             ci, cfg_fseq, cfg_valid, cfg_err, cfg_mod, cfg_body, h_len, h_f0, h_f1, h_crc,
                             c_fq[ci], c_v[ci], c_e[ci], c_md[ci], c_bd[ci], c_len[ci], c_f0[ci], c_f1[ci], c_crc[ci]);
                end
                ci = ci + 1;
            end
        end
        pv <= cfg_valid; pe <= cfg_err; pf <= cfg_fseq;
    end

    // ---------------- driver (valid/ready, FD-like DATA gating) ----------------
    function integer clocks_for(input integer n);
        clocks_for = ((n + 1) * cps_num) / cps_den - (n * cps_num) / cps_den;
    endfunction

    integer ii = 0, slot = 0, rdy_low = 0;
    reg     skip_frame = 0;
    task drive;
        integer w;
        begin
            while (ii < ni) begin
                // FD's B1: a frame's DATA waits for a valid config for it
                if (i_st[ii] == 3 && (ii == 0 || i_st[ii-1] != 3)) begin
                    while (!((cfg_valid && cfg_fseq == i_fq[ii]) || (cfg_err && cfg_fseq == i_fq[ii])))
                        @(negedge clk);
                    skip_frame = cfg_err && cfg_fseq == i_fq[ii];
                end
                if (i_st[ii] != 3) skip_frame = 0;
                if (skip_frame) begin ii = ii + 1; end
                else begin
                    @(negedge clk);
                    in_valid = 1; in_fq_set(ii);
                    // in_ready is sampled mid-cycle, where it is stable; the
                    // transfer then happens on the next rising edge.
                    while (!in_ready) begin rdy_low = rdy_low + 1; @(negedge clk); end
                    @(posedge clk);
                    if (i_st[ii] == 1 && i_mk[ii][2]) last_hdr_acc = cyc;
                    ii = ii + 1;
                    // pacing: groups one per C clocks, then a 72-group gap per symbol
                    if (!burst) begin
                        for (w = 1; w < clocks_for(slot); w = w + 1) begin @(negedge clk); in_valid = 0; end
                        slot = slot + 1;
                        if (i_mk[ii-1][2])
                            for (w = 0; w < 72; w = w + 1) begin
                                repeat (clocks_for(slot)) begin @(negedge clk); in_valid = 0; end
                                slot = slot + 1;
                            end
                    end
                end
            end
            @(negedge clk); in_valid = 0;
        end
    endtask

    task in_fq_set(input integer k);
        begin
            in_fseq = i_fq[k]; in_sym = i_sym[k]; in_sc = i_sc[k]; in_stype = i_st[k];
            in_n = i_n[k];
            begin : mk_llr
                integer b; reg [5:0] hb;
                hb = (i_st[k] == 1 && corrupt) ? ~i_llr[k] & 6'b000001 : i_llr[k];
                for (b = 0; b < 6; b = b + 1)
                    if (LLR_W == 1) in_llr[b] = hb[b];
                    else            in_llr[b*LLR_W +: LLR_W] = (b < i_n[k]) ? (hb[b] ? -7 : 7) : 0;
            end
            {in_ss, in_se, in_fs, in_fe} = i_mk[k];
        end
    endtask

    // ---------------- output checker ----------------
    integer oi = 0, errors = 0, extra = 0;
    always @(posedge clk) if (!rst && out_valid) begin
        if (oi >= no) begin
            extra = extra + 1;
        end else begin
            if (out_byte != o_b[oi] || out_last != o_l[oi] || out_fseq != o_fq[oi]) begin
                errors = errors + 1;
                if (errors <= 8)
                    $display("TB: BYTE MISMATCH #%0d at %0d: got %0d last=%0d fq=%0d want %0d last=%0d fq=%0d",
                             errors, oi, out_byte, out_last, out_fseq, o_b[oi], o_l[oi], o_fq[oi]);
            end
            oi = oi + 1;
        end
    end

    // +debug=1: trace the frame-start handshake
    integer dbg = 0;
    initial void'($value$plusargs("debug=%d", dbg));
    always @(posedge clk) if (dbg && !rst) begin
        if (dut.f_start) $display("DBG %0d f_start fq=%0d", cyc, dut.fq_head);
        if (dut.vit_done) $display("DBG %0d vit_done", cyc);
        if (out_valid && out_last) $display("DBG %0d out_last", cyc);
        if (dut.cb_hv && dut.h_first && dut.sub == 0 && dut.busy && (cyc % 50000 == 0))
            $display("DBG %0d first entry waiting: busy=%0d vit_running=%0d", cyc, dut.busy, dut.frame_done);
        if (dut.acc_dat && dut.in_first) $display("DBG %0d first DATA accepted fq=%0d", cyc, in_fseq);
    end

    integer t0;
    initial begin
        load;
        repeat (5) @(posedge clk);
        rst = 0;
        drive;
        t0 = cyc;
        while ((oi < no || ci < nc) && cyc < t0 + 3000000) @(posedge clk);
        repeat (5000) @(posedge clk);
        $display("TB: bytes %0d/%0d, byte mismatches %0d, extra %0d; configs %0d/%0d, config errors %0d",
                 oi, no, errors, extra, ci, nc, cfg_errors);
        $display("TB: flags cb_overflow=%0d unit_collision=%0d seq_err=%0d; in_ready low %0d clk",
                 cb_ovf, unit_col, seq_err, rdy_low);
        $display("TB: CB high-water %0d of %0d; config latency (last header item in -> publish) %0d clk",
                 cb_hwm, 128*216, cfg_lat_max);
        if (oi == no && errors == 0 && extra == 0 && ci == nc && cfg_errors == 0 &&
            !cb_ovf && !unit_col && !seq_err)
            $display("TB_RESULT PASS");
        else
            $display("TB_RESULT FAIL");
        $finish;
    end
endmodule
