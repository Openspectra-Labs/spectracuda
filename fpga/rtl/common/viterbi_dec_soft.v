// ============================================================
// viterbi_dec_soft.v -- SOFT-decision K=7 rate-1/2 Viterbi, overlapped
//                       ACS/traceback (~1 decoded bit/clock)
//
// EVALUATION MODULE for the bit-domain clock decision (125 vs 150 MHz).
// Same overlapped traceback / engines / serializer as viterbi_dec_ovl.v;
// only the ACS front end differs. Merged into one decoder once chosen.
//
// INPUT: one coded-bit pair per symbol, as SIGNED SYM_W-bit LLRs,
// sym = {l1, l0}, l0 for the G1 (0o171) bit. LLR > 0 means "0".
//
// BRANCH METRIC (to minimise), correlation form:
//     bm(e0, e1) = (e0 ? +l0 : -l0) + (e1 ? +l1 : -l1)   in [-2L, 2L]
// The two branches into a state carry complementary expected bits, so
//     bm_b = -bm_a              (NORM = 0, signed)
//     bm_b = 4L - bm_a          (NORM != 0, offset form bm + 2L in [0, 4L])
// The constant offset adds the same amount to both candidates, so it
// changes no decision.
//
// OVERFLOW PROOF (L = 2^(SYM_W-1)-1 = 7 for SYM_W = 4; memory m = 6):
//   * per step, branch metrics differ by at most B = 2(|l0|+|l1|) <= 4L = 28
//   * every state at t is reachable from the best state at t-6, so for
//     t >= 6:  max PM - min PM <= m*B = 168
//   * steps t < 6: the non-zero states start at X = m*B + 1 = 169 (more
//     than any path can gain in 6 steps, so an impossible start never
//     wins); spread <= X + m*B = 337
//   * ACS compares candidates differing by <= 337 + B = 365
//   NORM = 0 (two's-complement wrap, sign of difference): ordering is
//   preserved iff every compared difference < 2^(PM_W-1): PM_W = 10
//   (365 < 512); 9 would not be (256 < 365). Validated, not proven, by
//   study_pm_spread.py: measured max 140 steady / 197 early.
//   NORM = 1, window offset: PMs (offset form) never decrease; subtract
//   the snapshot's best metric once per traceback window (<= stale min,
//   so never negative): <= 2521 before the first window, <= 1568 after:
//   PM_W = 12, unsigned.
//   NORM = 2, threshold: when every PM >= T = 2^(PM_W-2), subtract T:
//   <= T + 337 + 3*28 = 677 < 1024: PM_W = 10, unsigned.
//   Wide, no normalization: NORM = 0 with PM_W = 22 (82,944 steps max
//   frame x 14 + 169 < 2^21) -- also the equivalence reference.
//
// OPT = 1 (required for NORM != 0): the four possible branch metrics of a
// step (and the four candidate differences) are computed and REGISTERED
// one clock ahead, with any normalization subtracted there. The ACS
// recursion is then only  pm -> {add, add, ternary-difference sign} ->
// select -> pm : one carry chain to the decision instead of two in
// series, and normalization never enters the recursion.
// OPT = 0: the straightforward form (branch metric -> add -> compare ->
// select), kept to measure what soft width does to the plain ACS path.
// ============================================================
`timescale 1ns / 1ps

module viterbi_dec_soft #(
    parameter integer TB_DISCARD = 42,
    parameter integer TB_GROUP   = 42,
    parameter integer RAM_AW     = 8,
    parameter integer SYM_W      = 4,     // LLR width per coded bit
    parameter integer PM_W       = 10,
    parameter integer NORM       = 0,     // 0 signed/modulo, 1 window, 2 threshold
    parameter integer OPT        = 1      // 1 = registered BMs + ternary decision
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [2*SYM_W-1:0] sym,       // {l1, l0}, signed
    input  wire        in_valid,
    output wire        in_ready,
    input  wire        last,
    output reg         out_bit,
    output reg         out_valid,
    output reg         frame_done
);
    localparam integer NS       = 64;
    localparam integer TB_TOTAL = TB_DISCARD + TB_GROUP;
    localparam integer RAM_D    = (1 << RAM_AW);
    localparam integer CW       = 8;            // job counters (<= TB_TOTAL)
    localparam integer LMAX     = (1 << (SYM_W - 1)) - 1;
    localparam integer BM_OFF   = (NORM != 0) ? 2 * LMAX : 0;
    localparam integer X_INIT   = 6 * 4 * LMAX + 1;          // 169
    localparam integer BMW      = SYM_W + 4;                 // signed BM width

    // ---- branch metric tables (as viterbi_dec) ----------------------
    function automatic parity7(input [6:0] x);
        begin parity7 = ^x; end
    endfunction
    wire [NS-1:0] o1a, o2a;
    genvar g;
    generate
        for (g = 0; g < NS; g = g + 1) begin : gen_tbl
            assign o1a[g] = parity7(g[6:0] & 7'o171);
            assign o2a[g] = parity7(g[6:0] & 7'o133);
        end
    endgenerate

    // ---- branch metrics --------------------------------------------------
    wire signed [SYM_W-1:0] l0 = sym[SYM_W-1:0];
    wire signed [SYM_W-1:0] l1 = sym[2*SYM_W-1:SYM_W];
    // bmv[c], c = {e0, e1}: the four possible branch metrics of this symbol
    wire signed [BMW-1:0] l0x = l0, l1x = l1;
    wire signed [BMW-1:0] bmv0 = -l0x - l1x + BM_OFF;     // e = 00
    wire signed [BMW-1:0] bmv1 = -l0x + l1x + BM_OFF;     // e = 01
    wire signed [BMW-1:0] bmv2 =  l0x - l1x + BM_OFF;     // e = 10
    wire signed [BMW-1:0] bmv3 =  l0x + l1x + BM_OFF;     // e = 11

    // ---- path metrics ------------------------------------------------------
    reg  [PM_W-1:0] pm   [0:NS-1];
    reg  [PM_W-1:0] pm_n [0:NS-1];
    reg  [NS-1:0]   dec_vec;
    integer ia, ib;

    // OPT=1 registered stage: branch metrics (normalization folded in) and
    // the candidate differences d[c] = bm(~c) - bm(c), one step ahead.
    // Each value feeds 32 ACS units x ~10 bits: let synthesis replicate
    // the registers so no copy drives more than 16 loads (routing, not
    // logic depth, was the 125 MHz limit).
    (* max_fanout = 16 *) reg  signed [PM_W:0] rv [0:3];
    (* max_fanout = 16 *) reg  signed [PM_W:0] rd [0:3];
    reg                  s_v, s_last;     // a registered step is ready

    always @* begin
        for (ia = 0; ia < NS; ia = ia + 1) begin : acs
            reg [1:0] c;
            reg signed [PM_W:0] bma, bmb, dl;
            reg [PM_W-1:0] ca, cb;
            reg dec;
            c = {o1a[ia], o2a[ia]};
            if (OPT != 0) begin
                bma = rv[c];  bmb = rv[~c];  dl = rd[c];
            end else begin
                case (c)
                    2'd0: bma = bmv0; 2'd1: bma = bmv1; 2'd2: bma = bmv2; default: bma = bmv3;
                endcase
                bmb = (NORM != 0) ? (4 * LMAX - bma) : -bma;
                dl  = bmb - bma;
            end
            ca = pm[ia >> 1]        + bma[PM_W-1:0];
            cb = pm[(ia >> 1) + 32] + bmb[PM_W-1:0];
            if (OPT != 0) begin
                // one carry chain: sign of (pm_b - pm_a + (bm_b - bm_a))
                if (NORM == 0) dec = ($signed(pm[(ia >> 1) + 32] - pm[ia >> 1] + dl[PM_W-1:0]) < 0);
                else           dec = ($signed({1'b0, pm[(ia >> 1) + 32]} - {1'b0, pm[ia >> 1]} + dl) < 0);
            end else begin
                if (NORM == 0) dec = ($signed(cb - ca) < 0);
                else           dec = (cb < ca);
            end
            pm_n[ia]    = dec ? cb : ca;
            dec_vec[ia] = dec;
        end
    end

    // ---- survivor RAM: two copies, one per engine ----------------------
    (* ram_style = "distributed" *) reg [NS-1:0] sram0 [0:RAM_D-1];
    (* ram_style = "distributed" *) reg [NS-1:0] sram1 [0:RAM_D-1];
    // The write pointer addresses two 256 x 64-bit LUT-RAM copies (~1,000
    // loads): at 125 MHz its fan-out was the whole-BD critical path (0 logic
    // levels, 93% routing). Let synthesis replicate it.
    (* max_fanout = 64 *) reg [RAM_AW-1:0] wptr;

    // ---- control -------------------------------------------------------
    reg         running;          // between start and the flush job
    reg         flushing;         // flush job issued, waiting for its bits
    reg [CW:0]  pending;          // steps not yet assigned to a job

    // job queue (1 entry) and the job being handed to an engine
    reg              q_v;
    reg              q_flush;
    reg [RAM_AW-1:0] q_head;      // tb_ptr: newest slice is at q_head-1
    reg [CW:0]       q_emit;
    reg              q_st_v;      // start state known
    reg [5:0]        q_st;
    // The flush job's own slot: `last` may arrive while a group job still
    // waits in q. (Making in_ready depend on `last` instead would close a
    // combinational loop through the caller's skid buffer.)
    reg              fq_v;
    reg [RAM_AW-1:0] fq_head;
    reg [CW:0]       fq_emit;

    // Stall only if THIS step would trigger a group job while q is still
    // occupied. Depends on registered state only.
    reg    lastseen;                 // `last` accepted; no more input this frame
    wire   step_last;
    wire   acs_go;                   // a trellis step executes this cycle
    // OPT=1: an accepted symbol executes one clock later (registered BMs),
    // so look one step further ahead before accepting.
    // (OPT=0 has no in-flight step, and acs_go = acc there: keep it out)
    wire   q_busy   = q_v || ((OPT != 0) && acs_go && trig_grp_c);
    wire   trig_grp_c;
    assign in_ready = running && !flushing && !lastseen &&
                      !(q_busy && (pending + (OPT != 0 ? {{CW{1'b0}}, s_v} : 0) + 1'b1 >= TB_TOTAL));
    wire   acc      = in_valid && in_ready;
    assign acs_go   = (OPT != 0) ? s_v : acc;
    assign step_last = (OPT != 0) ? s_last : last;
    assign trig_grp_c = !step_last && (pending + 1'b1 >= TB_TOTAL);
    wire   trig_grp = acs_go && trig_grp_c;
    wire   trig_fl  = acs_go && step_last;

    // ---- best state of the trigger step: snapshot + 3-stage tree -------
    reg  [PM_W-1:0] snap [0:NS-1];
    reg             snap_v;
    function automatic [5:0] sidx(input integer n);
        sidx = n;
    endfunction
    function automatic [PM_W+5:0] pick(input [PM_W+5:0] a, input [PM_W+5:0] b);
        // {metric, state}: smaller metric wins, ties keep `a` (lower index),
        // compared modulo like viterbi_dec's tree
        if (NORM == 0) pick = ($signed(b[PM_W+5:6] - a[PM_W+5:6]) < 0) ? b : a;
        else           pick = (b[PM_W+5:6] < a[PM_W+5:6]) ? b : a;
    endfunction
    reg            trig_q;
    reg [PM_W+5:0] t1 [0:15];     // after 2 levels: 64 -> 16
    reg [PM_W+5:0] t2 [0:3];      // after 4 levels: 16 -> 4
    reg            t1_v, t2_v;
    reg [5:0]      best;
    reg [PM_W-1:0] best_m;           // its metric (window normalization)
    reg            best_v;
    integer it;
    always @(posedge clk) begin
        if (rst) begin
            trig_q <= 1'b0; snap_v <= 1'b0; t1_v <= 1'b0; t2_v <= 1'b0; best_v <= 1'b0;
        end else begin
            // Snapshot the REGISTERED metrics one clock after the trigger
            // step: pm then holds exactly that step's result (a further
            // step on this clock updates pm on the same edge, so the old --
            // trigger -- value is what is captured). Taking pm_n here put
            // 640 far-away flops at the end of every ACS path: at 125 MHz
            // the critical path was 83% routing into snap.
            trig_q <= trig_grp;
            snap_v <= trig_q;
            if (trig_q) for (it = 0; it < NS; it = it + 1) snap[it] <= pm[it];
            t1_v <= snap_v;
            for (it = 0; it < 16; it = it + 1)
                t1[it] <= pick(pick({snap[4*it],   sidx(4*it)},   {snap[4*it+1], sidx(4*it+1)}),
                               pick({snap[4*it+2], sidx(4*it+2)}, {snap[4*it+3], sidx(4*it+3)}));
            t2_v <= t1_v;
            for (it = 0; it < 4; it = it + 1)
                t2[it] <= pick(pick(t1[4*it], t1[4*it+1]), pick(t1[4*it+2], t1[4*it+3]));
            best_v <= t2_v;
            {best_m, best} <= pick(pick(t2[0], t2[1]), pick(t2[2], t2[3]));
        end
    end

    // ---- traceback engines ----------------------------------------------
    localparam [1:0] E_IDLE = 2'd0, E_DISC = 2'd1, E_EMIT = 2'd2, E_HAND = 2'd3;
    reg [1:0]          e_st   [0:1];
    // e_ptr holds the survivor-RAM address the engine reads NEXT, i.e. the
    // traceback position minus one, pre-decremented so the RAM address comes
    // straight from a register (the subtractor in the traceback loop cost
    // 0.26 ns at 125 MHz in the full receiver). Same addresses as before.
    reg [RAM_AW-1:0]   e_ptr  [0:1];
    reg [5:0]          e_ts   [0:1];
    reg [CW:0]         e_cnt  [0:1];
    reg [CW:0]         e_ndis [0:1];
    reg [CW:0]         e_nemt [0:1];
    reg                e_fl   [0:1];
    reg [TB_TOTAL-1:0] e_lifo [0:1];

    // per-engine output buffers, drained by the serializer
    reg                o_v    [0:1];
    reg                o_fl   [0:1];
    reg [CW:0]         o_n    [0:1];
    reg [TB_TOTAL-1:0] o_bits [0:1];

    reg                disp;        // engine the next job goes to
    reg                serv;        // engine the serializer takes next
    reg [CW:0]         s_n;         // bits left in the serializer
    reg [TB_TOTAL-1:0] s_bits;
    reg                s_fl;

    // next traceback state, read from each engine's own RAM copy
    wire [RAM_AW-1:0] rp0 = e_ptr[0];
    wire [RAM_AW-1:0] rp1 = e_ptr[1];
    wire [5:0] nx0 = {sram0[rp0][e_ts[0]], e_ts[0][5:1]};
    wire [5:0] nx1 = {sram1[rp1][e_ts[1]], e_ts[1][5:1]};

    wire q_ready = q_v && (q_flush || q_st_v || best_v);   // start state known
    wire [5:0] q_start = q_flush ? 6'd0 : (q_st_v ? q_st : best);
    wire dispatch = q_ready && (e_st[disp] == E_IDLE);

    integer e;
    always @(posedge clk) begin
        if (rst) begin
            running <= 1'b0; flushing <= 1'b0; pending <= 0; wptr <= 0; lastseen <= 1'b0;
            q_v <= 1'b0; q_st_v <= 1'b0; fq_v <= 1'b0; disp <= 1'b0; serv <= 1'b0;
            s_n <= 0; s_fl <= 1'b0;
            out_valid <= 1'b0; frame_done <= 1'b0;
            for (e = 0; e < 2; e = e + 1) begin
                e_st[e] <= E_IDLE; o_v[e] <= 1'b0;
            end
        end else begin
            out_valid  <= 1'b0;
            frame_done <= 1'b0;

            // ---- start of a frame ----
            if (start && !running && !flushing) begin
                pm[0] <= {PM_W{1'b0}};
                for (ib = 1; ib < NS; ib = ib + 1) pm[ib] <= X_INIT;
                wptr <= 0; pending <= 0; running <= 1'b1; lastseen <= 1'b0;
            end

            if (acc && last) lastseen <= 1'b1;

            // ---- ACS step ----
            if (acs_go) begin
                for (ib = 0; ib < NS; ib = ib + 1) pm[ib] <= pm_n[ib];
                sram0[wptr] <= dec_vec;
                sram1[wptr] <= dec_vec;
                wptr <= wptr + 1'b1;
                if (trig_fl) begin
                    // into q if it is (or becomes) free, else the flush slot
                    if (!q_v || dispatch) begin
                        q_v <= 1'b1; q_flush <= 1'b1; q_head <= wptr + 1'b1;
                        q_emit <= pending + 1'b1; q_st_v <= 1'b1; q_st <= 6'd0;
                    end else begin
                        fq_v <= 1'b1; fq_head <= wptr + 1'b1; fq_emit <= pending + 1'b1;
                    end
                    pending <= 0; running <= 1'b0; flushing <= 1'b1;
                end else if (trig_grp) begin
                    q_v <= 1'b1; q_flush <= 1'b0; q_head <= wptr + 1'b1;
                    q_emit <= TB_GROUP; q_st_v <= 1'b0;
                    pending <= pending + 1'b1 - TB_GROUP;
                end else begin
                    pending <= pending + 1'b1;
                end
            end
            if (q_v && !q_flush && !q_st_v && best_v) begin
                q_st_v <= 1'b1; q_st <= best;
            end

            // ---- dispatch the queued job to the next engine in turn ----
            if (dispatch) begin
                e_st[disp]   <= q_flush ? E_EMIT : E_DISC;
                e_ptr[disp]  <= q_head - 1'b1;
                e_ts[disp]   <= q_start;
                e_cnt[disp]  <= 0;
                e_ndis[disp] <= q_flush ? 0 : TB_DISCARD;
                e_nemt[disp] <= q_emit;
                e_fl[disp]   <= q_flush;
                disp <= ~disp;
                // refill q from the flush slot, else empty it (an ACS
                // trigger this same cycle overrides below via its own write)
                if (fq_v) begin
                    q_v <= 1'b1; q_flush <= 1'b1; q_head <= fq_head;
                    q_emit <= fq_emit; q_st_v <= 1'b1; q_st <= 6'd0;
                    fq_v <= 1'b0;
                end else if (!(acs_go && (trig_grp || trig_fl))) begin
                    q_v <= 1'b0;
                end
            end

            // ---- engines walk back (same steps as viterbi_dec) ----
            for (e = 0; e < 2; e = e + 1) begin
                case (e_st[e])
                E_DISC: if (e_cnt[e] >= e_ndis[e]) begin
                            e_cnt[e] <= 0; e_st[e] <= E_EMIT;
                        end else begin
                            e_ptr[e] <= e_ptr[e] - 1'b1;
                            e_ts[e]  <= (e == 0) ? nx0 : nx1;
                            e_cnt[e] <= e_cnt[e] + 1'b1;
                        end
                E_EMIT: if (e_cnt[e] >= e_nemt[e]) begin
                            e_st[e] <= E_HAND;
                        end else begin
                            e_lifo[e] <= {e_lifo[e][TB_TOTAL-2:0], e_ts[e][0]};
                            e_ptr[e]  <= e_ptr[e] - 1'b1;
                            e_ts[e]   <= (e == 0) ? nx0 : nx1;
                            e_cnt[e]  <= e_cnt[e] + 1'b1;
                        end
                E_HAND: if (!o_v[e]) begin
                            o_v[e] <= 1'b1; o_fl[e] <= e_fl[e];
                            o_n[e] <= e_cnt[e]; o_bits[e] <= e_lifo[e];
                            e_st[e] <= E_IDLE;
                        end
                default: ;
                endcase
            end

            // ---- serializer: buffers in job order, oldest bit first ----
            if (s_n == 0) begin
                if (o_v[serv]) begin
                    s_bits <= o_bits[serv]; s_n <= o_n[serv]; s_fl <= o_fl[serv];
                    o_v[serv] <= 1'b0;
                    serv <= ~serv;
                    if (o_n[serv] == 0 && o_fl[serv]) begin
                        frame_done <= 1'b1; flushing <= 1'b0;
                    end
                end
            end else begin
                // pushed newest-first, so bit 0 is the oldest
                out_bit   <= s_bits[0];
                out_valid <= 1'b1;
                s_bits    <= s_bits >> 1;
                s_n       <= s_n - 1'b1;
                if (s_n == 1 && s_fl) begin
                    frame_done <= 1'b1; flushing <= 1'b0;
                end
            end
        end
    end
    // ---- OPT=1 input stage + normalization (outside the recursion) -----
    // Normalization subtracts a common value N from every candidate of ONE
    // step by subtracting it from that step's four registered branch
    // metrics -- the recursion itself never sees it.
    reg              n_req;          // a normalization is due
    reg  [PM_W-1:0]  n_val;
    reg  [2:0]       n_hold;         // wait until an applied N reached pm
    reg              all_ge;         // NORM=2: every PM >= T last clock
    integer in2;
    always @(posedge clk) begin
        if (rst) begin
            s_v <= 1'b0; s_last <= 1'b0; n_req <= 1'b0; n_val <= 0; n_hold <= 0; all_ge <= 1'b0;
        end else begin
            s_v <= acc;
            if (acc) begin
                s_last <= last;
                rv[0] <= bmv0 - (n_req ? $signed({1'b0, n_val}) : 0);
                rv[1] <= bmv1 - (n_req ? $signed({1'b0, n_val}) : 0);
                rv[2] <= bmv2 - (n_req ? $signed({1'b0, n_val}) : 0);
                rv[3] <= bmv3 - (n_req ? $signed({1'b0, n_val}) : 0);
                rd[0] <= bmv3 - bmv0; rd[1] <= bmv2 - bmv1;
                rd[2] <= bmv1 - bmv2; rd[3] <= bmv0 - bmv3;
            end
            // Hold-off after an application: the adjusted step executes one
            // clock later, pm shows it a clock after that, and all_ge one
            // more -- so ignore all_ge for 4 clocks, counted EVERY clock.
            // (It used to count only idle clocks, so with continuous input
            // it never expired, normalization stopped and the metrics
            // overflowed: caught by the wide-reference equivalence test.)
            if (acc && n_req) begin n_req <= 1'b0; n_hold <= 3'd4; end
            else if (n_hold != 0) n_hold <= n_hold - 1'b1;
            if (start && !running && !flushing) begin n_req <= 1'b0; n_hold <= 3'd4; end

            // NORM = 2: every PM has bit W-1 or W-2 set => all >= T
            all_ge <= 1'b1;
            for (in2 = 0; in2 < NS; in2 = in2 + 1)
                if (!(pm[in2][PM_W-1] | pm[in2][PM_W-2])) all_ge <= 1'b0;
            if (NORM == 2 && all_ge && !n_req && n_hold == 0) begin
                n_req <= 1'b1; n_val <= (1 << (PM_W - 2));
            end
            // NORM = 1: the best metric of each traceback snapshot
            if (NORM == 1 && best_v && !n_req) begin
                n_req <= 1'b1; n_val <= best_m;
            end
        end
    end
endmodule
