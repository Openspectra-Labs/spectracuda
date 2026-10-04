// ============================================================
// rx_freq_domain_tb.v -- standalone test of the frequency-domain stage
//
// Built ONCE (run_fd_stage.py); every test is a run with different
// plusargs, so the DUT is the same binary for all of them.
//
//   +stim=FILE   TD->FD items, one per line:  fseq sym bin stype re im
//                (a new frame starts at sym 0, bin 0)
//   +exp=FILE    expected FD->BIT items, one per line:
//                fseq sym sc stype n l0 l1 l2 l3 l4 l5 ss se fs fe
//   +cfg=FILE    config events, one per line:
//                trig_frame delay valid err fseq mod body c2 dmrs
//                Event i drives the C1 bundle `delay` clocks after the
//                LAST header item of the trig_frame-th frame (0-based, in
//                arrival order -- fseq values repeat) was accepted on the
//                output (i.e. after the bit domain could have parsed it),
//                and holds it until event i+1. Events fire in file order.
//   +cps_num=N +cps_den=D  clocks per input sample = N/D (bins arrive one
//                per sample period, 256 bins then a 32-sample CP gap)
//   +stall=P     out_ready low with probability P percent (LFSR)
//   +gap=G       idle samples between frames
//
// Checks: every accepted output item against +exp (all fields); every
// field stable while out_valid && !out_ready; no extra outputs; B1/B2
// overflow, sequence errors and unsupported-config flags all 0; B1 empty
// at the end. Prints "TB_RESULT PASS" or "TB_RESULT FAIL".
// ============================================================
`timescale 1ns / 1ps

module rx_freq_domain_tb;
    localparam integer LLR_W = 1;
    localparam integer MAXS  = 131072;
    localparam integer MAXE  = 65536;
    localparam integer MAXC  = 16;

    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    // ---------------- DUT ----------------
    reg         in_valid = 0;
    reg  signed [31:0] in_re = 0, in_im = 0;
    reg  [7:0]  in_bin = 0, in_sym = 0;
    reg  [2:0]  in_stype = 0;
    reg  [1:0]  in_fseq = 0;
    reg         in_fs = 0;

    reg         cfg_valid = 0, cfg_err = 0;
    reg  [1:0]  cfg_fseq = 0, cfg_dmrs = 0;
    reg  [2:0]  cfg_mod = 0;
    reg  [7:0]  cfg_body = 0, cfg_c2 = 0;

    wire        out_valid;
    reg         out_ready = 1;
    wire [6*LLR_W-1:0] out_llr;
    wire [2:0]  out_n, out_stype;
    wire [7:0]  out_sc, out_sym;
    wire [1:0]  out_fseq;
    wire        out_ss, out_se, out_fs, out_fe;
    wire        b1_ovf, b2_ovf, seq_err, cfg_uns, fseq_col, hdr_no_tr, hq_held;
    wire [11:0] b1_level, b1_hwm;
    wire [9:0]  b2_hwm;

    rx_freq_domain #(.LLR_W(LLR_W)) dut (
        .clk(clk), .rst(rst),
        .in_valid(in_valid), .in_re(in_re), .in_im(in_im), .in_bin(in_bin),
        .in_sym_idx(in_sym), .in_stype(in_stype), .in_fseq(in_fseq),
        .in_frame_start(in_fs),
        .cfg_valid(cfg_valid), .cfg_err(cfg_err), .cfg_fseq(cfg_fseq),
        .cfg_mod(cfg_mod), .cfg_body_syms(cfg_body), .cfg_c2_syms(cfg_c2),
        .cfg_dmrs_period(cfg_dmrs),
        .out_valid(out_valid), .out_ready(out_ready), .out_llr(out_llr),
        .out_n(out_n), .out_sc(out_sc), .out_sym_idx(out_sym),
        .out_stype(out_stype), .out_fseq(out_fseq),
        .out_sym_start(out_ss), .out_sym_end(out_se),
        .out_frame_start(out_fs), .out_frame_end(out_fe),
        .st_b1_overflow(b1_ovf), .st_b2_overflow(b2_ovf),
        .st_seq_err(seq_err), .st_cfg_unsupported(cfg_uns),
        .st_fseq_collision(fseq_col), .st_hdr_no_train(hdr_no_tr), .st_hq_held(hq_held),
        .st_b1_level(b1_level), .st_b1_hwm(b1_hwm), .st_b2_hwm(b2_hwm));

    // ---------------- files ----------------
    reg  [1:0]  s_fq  [0:MAXS-1];
    reg  [7:0]  s_sym [0:MAXS-1], s_bin [0:MAXS-1];
    reg  [2:0]  s_st  [0:MAXS-1];
    reg  signed [31:0] s_re [0:MAXS-1], s_im [0:MAXS-1];
    integer ns = 0;

    reg  [1:0]  e_fq [0:MAXE-1];
    reg  [7:0]  e_sym[0:MAXE-1], e_sc[0:MAXE-1];
    reg  [2:0]  e_st [0:MAXE-1], e_n [0:MAXE-1];
    reg  [5:0]  e_llr[0:MAXE-1];
    reg  [3:0]  e_mk [0:MAXE-1];          // {ss, se, fs, fe}
    integer ne = 0;

    integer c_tf[0:MAXC-1], c_dl[0:MAXC-1], c_v[0:MAXC-1], c_e[0:MAXC-1],
            c_fq[0:MAXC-1], c_md[0:MAXC-1], c_bd[0:MAXC-1], c_c2[0:MAXC-1],
            c_dp[0:MAXC-1];
    integer nc = 0;

    integer cps_num = 1, cps_den = 1, stall = 0, gap = 64;
    // Negative tests: +expect=1 (fseq collision) or +expect=2 (header
    // waiting for an H with no training) -- PASS means that flag fired.
    integer expect_flag = 0;
    reg [1023:0] fname;

    task load;
        integer fd, r, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14;
        begin
            if (!$value$plusargs("stim=%s", fname)) $fatal(1, "need +stim=");
            fd = $fopen(fname, "r");
            while (!$feof(fd)) begin
                r = $fscanf(fd, "%d %d %d %d %d %d\n", a0, a1, a2, a3, a4, a5);
                if (r == 6) begin
                    s_fq[ns] = a0; s_sym[ns] = a1; s_bin[ns] = a2; s_st[ns] = a3;
                    s_re[ns] = a4; s_im[ns] = a5; ns = ns + 1;
                end
            end
            $fclose(fd);
            if (!$value$plusargs("exp=%s", fname)) $fatal(1, "need +exp=");
            fd = $fopen(fname, "r");
            while (!$feof(fd)) begin
                r = $fscanf(fd, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14);
                if (r == 15) begin
                    e_fq[ne] = a0; e_sym[ne] = a1; e_sc[ne] = a2; e_st[ne] = a3; e_n[ne] = a4;
                    e_llr[ne] = {a10[0], a9[0], a8[0], a7[0], a6[0], a5[0]};
                    e_mk[ne]  = {a11[0], a12[0], a13[0], a14[0]};
                    ne = ne + 1;
                end
            end
            $fclose(fd);
            if (!$value$plusargs("cfg=%s", fname)) $fatal(1, "need +cfg=");
            fd = $fopen(fname, "r");
            while (!$feof(fd)) begin
                r = $fscanf(fd, "%d %d %d %d %d %d %d %d %d\n", a0, a1, a2, a3, a4, a5, a6, a7, a8);
                if (r == 9) begin
                    c_tf[nc] = a0; c_dl[nc] = a1; c_v[nc] = a2; c_e[nc] = a3; c_fq[nc] = a4;
                    c_md[nc] = a5; c_bd[nc] = a6; c_c2[nc] = a7; c_dp[nc] = a8; nc = nc + 1;
                end
            end
            $fclose(fd);
            void'($value$plusargs("cps_num=%d", cps_num));
            void'($value$plusargs("cps_den=%d", cps_den));
            void'($value$plusargs("stall=%d", stall));
            void'($value$plusargs("gap=%d", gap));
            void'($value$plusargs("expect=%d", expect_flag));
            $display("TB: %0d input items, %0d expected outputs, %0d cfg events, C=%0d/%0d, stall=%0d%%",
                     ns, ne, nc, cps_num, cps_den, stall);
        end
    endtask

    // ---------------- I1 driver: one bin per sample period ----------------
    integer cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

    integer si = 0, samp = 0, in_done = 0;
    integer in_first_hdr_cyc = -1, in_last_cyc = 0;

    // clocks occupied by sample number `samp` at rate num/den
    function integer clocks_for(input integer n);
        clocks_for = ((n + 1) * cps_num) / cps_den - (n * cps_num) / cps_den;
    endfunction

    task drive_inputs;
        integer j, w;
        begin
            while (si < ns) begin
                // gap between frames, CP gap before every symbol
                // Idle sample periods drive in_valid LOW on every clock: at
                // C=1 no clock is otherwise idle, and a gap that merely waits
                // repeats the previous bin (found that way).
                if (si > 0 && s_sym[si] == 0 && s_bin[si] == 0)
                    for (j = 0; j < gap; j = j + 1) begin
                        for (w = 0; w < clocks_for(samp); w = w + 1) begin
                            @(negedge clk); in_valid = 1'b0;
                        end
                        samp = samp + 1;
                    end
                if (s_bin[si] == 0)
                    for (j = 0; j < 32; j = j + 1) begin
                        for (w = 0; w < clocks_for(samp); w = w + 1) begin
                            @(negedge clk); in_valid = 1'b0;
                        end
                        samp = samp + 1;
                    end
                @(negedge clk);
                in_valid = 1'b1;
                in_re = s_re[si]; in_im = s_im[si]; in_bin = s_bin[si];
                in_sym = s_sym[si]; in_stype = s_st[si]; in_fseq = s_fq[si];
                in_fs = (s_sym[si] == 0 && s_bin[si] == 0);
                if (s_st[si] == 1 && in_first_hdr_cyc < 0) in_first_hdr_cyc = cyc;
                in_last_cyc = cyc;
                for (w = 1; w < clocks_for(samp); w = w + 1) begin
                    @(negedge clk); in_valid = 1'b0;
                end
                samp = samp + 1;
                si = si + 1;
            end
            @(negedge clk); in_valid = 1'b0;
            in_done = 1;
        end
    endtask

    // ---------------- C1 driver ----------------
    integer hdr_end_cyc [0:63];          // by frame ordinal
    integer n_hdr_end = 0;
    integer ci = 0, ev_cyc = -1, cfg_out_cyc = -1, first_hdr_out_cyc = -1;
    integer last_fe_cyc = -1;

    always @(posedge clk) if (!rst && ci < nc) begin
        if (n_hdr_end > c_tf[ci] && cyc >= hdr_end_cyc[c_tf[ci]] + c_dl[ci]) begin
            cfg_valid <= c_v[ci][0]; cfg_err <= c_e[ci][0]; cfg_fseq <= c_fq[ci];
            cfg_mod <= c_md[ci]; cfg_body <= c_bd[ci]; cfg_c2 <= c_c2[ci];
            cfg_dmrs <= c_dp[ci];
            if (cfg_out_cyc < 0) cfg_out_cyc = cyc;
            ci <= ci + 1;
        end
    end

    // ---------------- output checker + ready stalls ----------------
    reg [15:0] lfsr = 16'hACE1;
    always @(posedge clk) begin
        lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};
        out_ready <= rst ? 1'b1 : ((lfsr % 100) >= stall);
    end

    integer ei = 0, errors = 0, stab_err = 0, extra = 0;
    reg  held = 0;
    reg  [6*LLR_W-1:0] h_llr; reg [2:0] h_n, h_st; reg [7:0] h_sc, h_sym;
    reg  [1:0] h_fq; reg [3:0] h_mk;

    always @(posedge clk) if (!rst) begin
        // stability: what was presented and not taken must still be there
        if (held) begin
            if (!out_valid || out_llr != h_llr || out_n != h_n || out_stype != h_st ||
                out_sc != h_sc || out_sym != h_sym || out_fseq != h_fq ||
                {out_ss, out_se, out_fs, out_fe} != h_mk) begin
                stab_err = stab_err + 1;
                if (stab_err <= 5) $display("TB: STABILITY violated at cyc %0d", cyc);
            end
        end
        held <= out_valid && !out_ready;
        h_llr <= out_llr; h_n <= out_n; h_st <= out_stype; h_sc <= out_sc;
        h_sym <= out_sym; h_fq <= out_fseq; h_mk <= {out_ss, out_se, out_fs, out_fe};

        if (out_valid && out_ready) begin
            if (out_stype == 3'd1 && first_hdr_out_cyc < 0) first_hdr_out_cyc = cyc;
            if (out_fe) last_fe_cyc = cyc;
            if (out_stype == 3'd1 && out_se) begin
                hdr_end_cyc[n_hdr_end] = cyc;
                n_hdr_end = n_hdr_end + 1;
            end
            if (ei >= ne) begin
                extra = extra + 1;
                if (extra <= 5) $display("TB: EXTRA output fseq=%0d sym=%0d sc=%0d st=%0d", out_fseq, out_sym, out_sc, out_stype);
            end else begin
                if (out_fseq != e_fq[ei] || out_sym != e_sym[ei] || out_sc != e_sc[ei] ||
                    out_stype != e_st[ei] || out_n != e_n[ei] || out_llr != e_llr[ei] ||
                    {out_ss, out_se, out_fs, out_fe} != e_mk[ei]) begin
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("TB: MISMATCH #%0d item %0d: got fq=%0d sym=%0d sc=%0d st=%0d n=%0d llr=%b mk=%b | want fq=%0d sym=%0d sc=%0d st=%0d n=%0d llr=%b mk=%b",
                                 errors, ei, out_fseq, out_sym, out_sc, out_stype, out_n, out_llr,
                                 {out_ss, out_se, out_fs, out_fe}, e_fq[ei], e_sym[ei], e_sc[ei],
                                 e_st[ei], e_n[ei], e_llr[ei], e_mk[ei]);
                end
                ei = ei + 1;
            end
        end
    end

    // ---------------- run ----------------
    integer t_end;
    initial begin
        load;
        repeat (5) @(posedge clk);
        rst = 0;
        drive_inputs;
        // drain: wait for everything expected, then a quiet period
        t_end = cyc;
        if (expect_flag != 0) begin
            repeat (50000) @(posedge clk);
            $display("TB: negative test, expect=%0d: fseq_collision=%0d hdr_no_train=%0d",
                     expect_flag, fseq_col, hdr_no_tr);
            if ((expect_flag == 1 && fseq_col) || (expect_flag == 2 && hdr_no_tr))
                $display("TB_RESULT PASS");
            else
                $display("TB_RESULT FAIL");
            $finish;
        end
        while (ei < ne && cyc < t_end + 2000000) @(posedge clk);
        repeat (20000) @(posedge clk);
        $display("TB: outputs %0d/%0d, mismatches %0d, extra %0d, stability %0d",
                 ei, ne, errors, extra, stab_err);
        $display("TB: flags b1_ovf=%0d b2_ovf=%0d seq_err=%0d cfg_unsupported=%0d fseq_collision=%0d hdr_no_train=%0d b1_level_end=%0d",
                 b1_ovf, b2_ovf, seq_err, cfg_uns, fseq_col, hdr_no_tr, b1_level);
        $display("TB: B1 high-water %0d of %0d, B2 high-water %0d of 512, hq_held=%0d",
                 b1_hwm, 5*256, b2_hwm, hq_held);
        $display("TB: latency first header bin in -> first header group out: %0d clk; last input bin -> last frame_end out: %0d clk",
                 first_hdr_out_cyc - in_first_hdr_cyc, last_fe_cyc - in_last_cyc);
        if (ei == ne && errors == 0 && extra == 0 && stab_err == 0 &&
            !b1_ovf && !b2_ovf && !seq_err && !cfg_uns && !fseq_col && !hdr_no_tr &&
            b1_level == 0)
            $display("TB_RESULT PASS");
        else
            $display("TB_RESULT FAIL");
        $finish;
    end
endmodule
