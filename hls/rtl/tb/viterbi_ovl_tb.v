// ============================================================
// viterbi_ovl_tb.v -- equivalence: viterbi_dec_ovl vs viterbi_dec
//
// Both decoders get the same frames of coded symbol pairs, each through
// its own valid/ready handshake (their stalls differ), and their decoded
// bits must be identical, frame by frame.
//
//   +sym=FILE   one frame per line: "<n_sym> <s0> <s1> ..." (s = 2-bit pair)
//   +gap=P      sparse input: a symbol is offered with probability (100-P)%
//               per clock (0 = continuous, the throughput case)
//
// Prints decoded bits/clock for the overlapped decoder while input was
// continuously available, and "TB_RESULT PASS"/"FAIL".
// ============================================================
`timescale 1ns / 1ps

module viterbi_ovl_tb;
    localparam integer MAXS = 400000;
    reg clk = 0, rst = 1;
    always #5 clk = ~clk;

    reg  [1:0] syms [0:MAXS-1];
    integer flen [0:255], fbase [0:255], nf = 0, ns = 0, gap = 0;

    // ---- two decoders, two drivers ----
    reg  st_a = 0, st_b = 0;
    reg  [1:0] sa, sb;  reg va = 0, vb = 0, la = 0, lb = 0;
    wire ra, rb, oba, obb, ova, ovb, fda, fdb;

    viterbi_dec     u_a (.clk(clk), .rst(rst), .start(st_a), .sym(sa), .in_valid(va),
                         .in_ready(ra), .last(la), .out_bit(oba), .out_valid(ova),
                         .frame_done(fda));
    viterbi_dec_ovl u_b (.clk(clk), .rst(rst), .start(st_b), .sym(sb), .in_valid(vb),
                         .in_ready(rb), .last(lb), .out_bit(obb), .out_valid(ovb),
                         .frame_done(fdb));

    // collect outputs
    reg  bits_a [0:MAXS-1]; reg bits_b [0:MAXS-1];
    integer na = 0, nb = 0, done_a = 0, done_b = 0;
    always @(posedge clk) begin
        if (ova) begin bits_a[na] = oba; na = na + 1; end
        if (ovb) begin bits_b[nb] = obb; nb = nb + 1; end
        if (fda) done_a = done_a + 1;
        if (fdb) done_b = done_b + 1;
    end

    // DBGOVL: trace the overlapped decoder's queue/engines (+dbg=1)
    integer dbg = 0;
    initial void'($value$plusargs("dbg=%d", dbg));
    always @(posedge clk) if (dbg && !rst && $time < 200000)
        if ((u_b.acs_go && (u_b.trig_grp || u_b.trig_fl)) || u_b.dispatch || u_b.frame_done || u_a.frame_done)
            $display("DBG t=%0t grp=%0d fl=%0d disp=%0d q_v=%0d fq_v=%0d best_v=%0d e0=%0d e1=%0d o0=%0d o1=%0d s_n=%0d fdB=%0d fdA=%0d pend=%0d run=%0d flush=%0d",
                $time, u_b.trig_grp, u_b.trig_fl, u_b.dispatch, u_b.q_v, u_b.fq_v, u_b.best_v,
                u_b.e_st[0], u_b.e_st[1], u_b.o_v[0], u_b.o_v[1], u_b.s_n, u_b.frame_done, u_a.frame_done,
                u_b.pending, u_b.running, u_b.flushing);

    reg [31:0] lfsr_a = 32'h1234567, lfsr_b = 32'h89abcde;
    integer busy_cyc = 0, b_bits_start = 0;

    // automatic: two fork branches run this task at once; a static task
    // would share f/k between them.
    task automatic run_dec(input integer which);
        integer f, k, wait_done;
        begin
            for (f = 0; f < nf; f = f + 1) begin
                // start, then feed the frame
                @(negedge clk);
                if (which == 0) st_a = 1; else st_b = 1;
                @(negedge clk);
                if (which == 0) st_a = 0; else st_b = 0;
                k = 0;
                while (k < flen[f]) begin
                    if (which == 0) begin
                        lfsr_a = {lfsr_a[30:0], lfsr_a[31] ^ lfsr_a[21] ^ lfsr_a[1] ^ lfsr_a[0]};
                        va = (gap == 0) || ((lfsr_a % 100) >= gap);
                        sa = syms[fbase[f] + k]; la = (k == flen[f] - 1);
                    end else begin
                        lfsr_b = {lfsr_b[30:0], lfsr_b[31] ^ lfsr_b[21] ^ lfsr_b[1] ^ lfsr_b[0]};
                        vb = (gap == 0) || ((lfsr_b % 100) >= gap);
                        sb = syms[fbase[f] + k]; lb = (k == flen[f] - 1);
                    end
                    @(posedge clk);
                    if (which == 0) begin if (va && ra) k = k + 1; end
                    else begin
                        if (vb && rb) k = k + 1;
                        if (gap == 0) busy_cyc = busy_cyc + 1;
                    end
                    @(negedge clk);
                end
                if (which == 0) begin va = 0; la = 0; end else begin vb = 0; lb = 0; end
                // wait for this frame's flush before the next start
                while (((which == 0) ? done_a : done_b) < f + 1) @(posedge clk);
            end
        end
    endtask

    reg [8191:0] fname;
    integer fd, r, n, v, f2, i, errs;
    initial begin
        if (!$value$plusargs("sym=%s", fname)) $fatal(1, "need +sym=");
        void'($value$plusargs("gap=%d", gap));
        fd = $fopen(fname, "r");
        while (!$feof(fd) && nf < 256) begin
            r = $fscanf(fd, "%d", n);
            if (r == 1) begin
                flen[nf] = n; fbase[nf] = ns;
                for (i = 0; i < n; i = i + 1) begin r = $fscanf(fd, "%d", v); syms[ns] = v; ns = ns + 1; end
                nf = nf + 1;
            end
        end
        $fclose(fd);
        repeat (5) @(posedge clk);
        rst = 0;
        fork
            run_dec(0);
            run_dec(1);
        join
        repeat (2000) @(posedge clk);
        errs = 0;
        for (i = 0; i < na && i < nb; i = i + 1) if (bits_a[i] !== bits_b[i]) errs = errs + 1;
        // +dump=PREFIX: write both decoders' bits for an error-rate check
        if ($value$plusargs("dump=%s", fname)) begin
            fd = $fopen({fname, ".old"}, "w");
            for (i = 0; i < na; i = i + 1) $fwrite(fd, "%0d", bits_a[i]);
            $fclose(fd);
            fd = $fopen({fname, ".new"}, "w");
            for (i = 0; i < nb; i = i + 1) $fwrite(fd, "%0d", bits_b[i]);
            $fclose(fd);
        end
        $display("TB: %0d frames, %0d symbols; decoded bits old=%0d new=%0d, mismatches=%0d, frames done old=%0d new=%0d",
                 nf, ns, na, nb, errs, done_a, done_b);
        if (gap == 0)
            $display("TB: overlapped decoder: %0d decoded bits in %0d input-busy clocks = %0d.%03d bits/clock",
                     nb, busy_cyc, (nb * 1000 / busy_cyc) / 1000, (nb * 1000 / busy_cyc) % 1000);
        if (na == nb && errs == 0 && done_a == nf && done_b == nf) $display("TB_RESULT PASS");
        else $display("TB_RESULT FAIL");
        $finish;
    end
endmodule
