// ============================================================
// viterbi_dec_ovl.v -- hard-decision Viterbi, rate 1/2, K=7, with ACS
//                      and traceback OVERLAPPED (~1 decoded bit/clock)
//
// Drop-in for viterbi_dec.v (same ports, same trellis, same sliding
// window). viterbi_dec pauses ACS while it traces back: 42 useful clocks
// in every ~171, 0.246 decoded bits/clock. Here ACS never pauses:
//
//   * Every TB_GROUP (42) new trellis steps, once TB_DISCARD+TB_GROUP (84)
//     are held, a traceback JOB is queued -- exactly the windows
//     viterbi_dec uses (walk back 42 discarded, then emit the next 42).
//     The final flush (on `last`) starts from state 0, discards nothing
//     and emits everything pending -- also as viterbi_dec.
//   * Two traceback ENGINES take jobs in strict turn (0,1,0,1...). Each
//     has its own copy of the survivor RAM (write broadcast, one read port
//     each), so they walk concurrently with ACS and with each other. A job
//     costs ~86 clocks and arrives every >= 42, so two engines keep up
//     (a 1-entry job queue absorbs the rest; ACS stalls only if a job
//     would have to wait behind a full queue).
//   * Engines hand their bits (newest first) to an output SERIALIZER,
//     again in strict turn, which emits them oldest first at 1 bit/clock
//     -- so output order is job order regardless of engine timing.
//
// START STATE of a group traceback: the arg-min of the path metrics AFTER
// the trigger step, from a snapshot taken on that step and a 3-stage
// tree (two compare levels per stage). viterbi_dec uses a best state that
// is two CLOCKS stale, i.e. a different trellis step depending on input
// timing; TB_DISCARD exists so that the survivors have merged and the
// emitted bits do not depend on the start state -- verified, not assumed,
// by the equivalence test against viterbi_dec (tb/rx/viterbi_ovl_tb.v).
//
// Survivor RAM depth 2**RAM_AW = 256 >= the widest span in flight
// (a job's oldest slice to ACS's newest while two engines and the queue
// are busy, ~214).
// ============================================================
`timescale 1ns / 1ps

module viterbi_dec_ovl #(
    parameter integer TB_DISCARD = 42,
    parameter integer TB_GROUP   = 42,
    parameter integer RAM_AW     = 8,
    parameter integer PM_W       = 8
)(
    input  wire        clk,
    input  wire        rst,
    input  wire        start,
    input  wire [1:0]  sym,
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

    // ---- path metrics and ACS (identical arithmetic to viterbi_dec) --
    reg  [PM_W-1:0] pm   [0:NS-1];
    reg  [PM_W-1:0] pm_n [0:NS-1];
    reg  [NS-1:0]   dec_vec;
    integer ia, ib;
    always @* begin
        for (ia = 0; ia < NS; ia = ia + 1) begin : acs
            reg [1:0]      bm_a, bm_b;
            reg [PM_W-1:0] ca, cb;
            bm_a = (sym[0] ^ o1a[ia]) + (sym[1] ^ o2a[ia]);
            bm_b = 2'd2 - bm_a;
            ca   = pm[ia >> 1]        + bm_a;
            cb   = pm[(ia >> 1) + 32] + bm_b;
            if ($signed(cb - ca) < 0) begin
                pm_n[ia] = cb;  dec_vec[ia] = 1'b1;
            end else begin
                pm_n[ia] = ca;  dec_vec[ia] = 1'b0;
            end
        end
    end

    // ---- survivor RAM: two copies, one per engine ----------------------
    (* ram_style = "distributed" *) reg [NS-1:0] sram0 [0:RAM_D-1];
    (* ram_style = "distributed" *) reg [NS-1:0] sram1 [0:RAM_D-1];
    reg [RAM_AW-1:0] wptr;

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
    assign in_ready = running && !flushing && !(q_v && (pending + 1'b1 >= TB_TOTAL));
    wire   acs_go   = in_valid && in_ready;
    wire   trig_grp = acs_go && !last && (pending + 1'b1 >= TB_TOTAL);
    wire   trig_fl  = acs_go && last;

    // ---- best state of the trigger step: snapshot + 3-stage tree -------
    reg  [PM_W-1:0] snap [0:NS-1];
    reg             snap_v;
    function automatic [5:0] sidx(input integer n);
        sidx = n;
    endfunction
    function automatic [PM_W+5:0] pick(input [PM_W+5:0] a, input [PM_W+5:0] b);
        // {metric, state}: smaller metric wins, ties keep `a` (lower index),
        // compared modulo like viterbi_dec's tree
        pick = ($signed(b[PM_W+5:6] - a[PM_W+5:6]) < 0) ? b : a;
    endfunction
    reg [PM_W+5:0] t1 [0:15];     // after 2 levels: 64 -> 16
    reg [PM_W+5:0] t2 [0:3];      // after 4 levels: 16 -> 4
    reg            t1_v, t2_v;
    reg [5:0]      best;
    reg            best_v;
    integer it;
    always @(posedge clk) begin
        if (rst) begin
            snap_v <= 1'b0; t1_v <= 1'b0; t2_v <= 1'b0; best_v <= 1'b0;
        end else begin
            snap_v <= trig_grp;
            if (trig_grp) for (it = 0; it < NS; it = it + 1) snap[it] <= pm_n[it];
            t1_v <= snap_v;
            for (it = 0; it < 16; it = it + 1)
                t1[it] <= pick(pick({snap[4*it],   sidx(4*it)},   {snap[4*it+1], sidx(4*it+1)}),
                               pick({snap[4*it+2], sidx(4*it+2)}, {snap[4*it+3], sidx(4*it+3)}));
            t2_v <= t1_v;
            for (it = 0; it < 4; it = it + 1)
                t2[it] <= pick(pick(t1[4*it], t1[4*it+1]), pick(t1[4*it+2], t1[4*it+3]));
            best_v <= t2_v;
            best   <= pick(pick(t2[0], t2[1]), pick(t2[2], t2[3]));
        end
    end

    // ---- traceback engines ----------------------------------------------
    localparam [1:0] E_IDLE = 2'd0, E_DISC = 2'd1, E_EMIT = 2'd2, E_HAND = 2'd3;
    reg [1:0]          e_st   [0:1];
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
    wire [RAM_AW-1:0] rp0 = e_ptr[0] - 1'b1;
    wire [RAM_AW-1:0] rp1 = e_ptr[1] - 1'b1;
    wire [5:0] nx0 = {sram0[rp0][e_ts[0]], e_ts[0][5:1]};
    wire [5:0] nx1 = {sram1[rp1][e_ts[1]], e_ts[1][5:1]};

    wire q_ready = q_v && (q_flush || q_st_v || best_v);   // start state known
    wire [5:0] q_start = q_flush ? 6'd0 : (q_st_v ? q_st : best);
    wire dispatch = q_ready && (e_st[disp] == E_IDLE);

    integer e;
    always @(posedge clk) begin
        if (rst) begin
            running <= 1'b0; flushing <= 1'b0; pending <= 0; wptr <= 0;
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
                for (ib = 1; ib < NS; ib = ib + 1) pm[ib] <= 8'd60;
                wptr <= 0; pending <= 0; running <= 1'b1;
            end

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
                e_ptr[disp]  <= q_head;
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
endmodule
