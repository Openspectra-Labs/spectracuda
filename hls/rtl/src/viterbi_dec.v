// ============================================================
// viterbi_dec.v -- hard-decision Viterbi decoder, rate 1/2, K=7
//
// Port of spectracuda's ConvolutionalCode (spectracuda/fec/viterbi.py),
// the "conv_v27" scheme that flies on the two-Pluto link. Written rather
// than instantiated: the Xilinx Viterbi LogiCORE is licensed, and the
// license available here is License_Type:Hardware_Evaluation, which
// self-expires in hardware. openofdm instantiates that core
// (verilog/viterbi.v -> viterbi_v7_0), so it is a reference for how the
// block sits in a receiver, not for the decoder itself.
//
// TRELLIS CONVENTION -- from viterbi.py:_build_transition_tables().
// The input bit is bit0 and the register shifts LEFT, which is not the
// more common shift-into-MSB form, so it is taken from the golden model
// rather than from a textbook:
//     reg        = (state << 1) | b      (7 bits: bit0 = input b)
//     next_state = reg & 63
//     out1       = parity(reg & 0o171)   emitted FIRST
//     out2       = parity(reg & 0o133)   emitted SECOND
//
// Reading that backwards gives the butterfly this module is built on.
// For a next-state ns, the input bit that produced it is ns[0], and its
// two predecessors are (ns>>1) and (ns>>1)+32. Substituting:
//     branch from pred_a:  reg_a = ns
//     branch from pred_b:  reg_b = ns | 64
//
// ONE BRANCH METRIC PER STATE, NOT TWO. Bit 6 is set in both 0o171
// (0b1111001) and 0o133 (0b1011011), so reg_b's outputs are reg_a's
// outputs with BOTH bits inverted. The two branches entering a state
// therefore always carry complementary symbol pairs, giving
//     bm_b = 2 - bm_a
// which halves the branch-metric logic. This is a property of these
// specific polynomials, so it is derived above rather than assumed --
// a punctured or different-K code would not have it.
//
// MODULO PATH METRICS, NO RENORMALIZATION SUBTRACT. Hard-decision
// branch metrics are 0..2, and the spread between the best and worst
// survivor on this trellis is bounded well under 128. Carrying metrics
// as PM_W=8 two's-complement and comparing via the sign of (a-b) is
// then exact, and costs nothing -- the usual alternative, subtracting
// the running minimum every step, needs a 64-way min tree per step.
//
// TRACEBACK: sliding, not whole-frame. viterbi.py allocates
// survivor_prev_state[T][64] and traces the entire frame, which fabric
// cannot do for a runtime payload length. libcorrect -- the C model the
// shipping x86/ARM kernels reproduce bit-exactly -- already uses a fixed
// sliding window (src/convolutional/decode.c:271 passes 5*order=30 and
// 15*order=90), so a window is what the golden model does too, and only
// the depth is a choice here.
//
// ACS PAUSES DURING TRACEBACK, deliberately. It avoids a dual-port
// survivor RAM and the ACS/traceback concurrency that goes with it.
//
// THE COST, counted from the state machine below (TB_GROUP=TB_DISCARD=42):
//     S_ACS       42 clocks   ingesting, 2 coded bits/clk
//     S_TB_DISC   42 + 1      ingesting nothing
//     S_TB_EMIT   42 + 1      ingesting nothing
//     S_POP       42 + 1      ingesting nothing
//     ------------------------------------------------
//     171 clocks for 84 coded bits / 42 decoded bits
//
// Duty cycle 42/171 = 24.6%, so at 100 MHz:
//     49.1 Mbit/s CODED ingest, 24.6 Mbit/s decoded, 4.07 clocks/bit.
//
// An earlier version of this comment claimed "~3 clocks per decoded bit,
// ~33 Mbit/s decoded" -- 34% optimistic, and it was the only throughput
// figure anyone had. Recomputed above.
//
// WHAT THAT COVERS. Coded demand is fs * (N_DATA/SLOT_LEN) * bps
// = fs * 0.75 * bps, so at 10 Msps: QPSK 15, QAM16 30, QAM64 45 Mbit/s.
// All three fit, but QAM64 has only 1.09x margin -- the tightest number
// in the receiver, and a CALCULATED one: simulation runs 1 sample/clock,
// which is nothing like the real rate relationship, so no test here
// exercises it.
//
// Ceilings per MCS (49.1 / (0.75*bps)): QPSK 32.7, QAM16 16.4,
// QAM64 10.9 Msps. 20 Msps needs more for QAM16 and QAM64.
//
// TWO WAYS UP, in increasing order of work:
//   * TB_GROUP 42 -> 168 amortises the fixed TB_DISCARD overhead:
//     2G/(3G+D+3) goes 0.491 -> 0.612 (61.2 Mbit/s), ceiling 2/3.
//     Costs one BRAM (RAM_AW 7->8) and latency. NO BER CHANGE -- quality
//     is set by the discard depth D, not by how many bits a pass
//     harvests. This is the cheap fix for the QAM64 margin.
//   * Overlap ACS and traceback (ping-pong the survivor RAM) for a
//     continuous 2 bits/clk = 200 Mbit/s, which covers QAM64 to 44 Msps.
//     Radix-4 is only needed above that.
// ============================================================
`timescale 1ns / 1ps

module viterbi_dec #(
    parameter integer TB_DISCARD = 42,   // traced and thrown away (depth)
    parameter integer TB_GROUP   = 42,   // traced and emitted per burst
    parameter integer RAM_AW     = 7,    // 2**RAM_AW >= TB_DISCARD+TB_GROUP
    parameter integer PM_W       = 8
)(
    input  wire        clk,
    input  wire        rst,

    // Pulse before the first symbol of a frame. Resets the trellis to
    // state 0, which the zero-tail termination in viterbi.py guarantees
    // is where the encoder started.
    input  wire        start,

    // One rate-1/2 symbol pair. sym[0] is the G1 (0o171) output bit,
    // sym[1] the G2 (0o133) bit -- encode() interleaves them
    // encoded[0::2]=out1, encoded[1::2]=out2, so this is the wire order.
    input  wire [1:0]  sym,
    input  wire        in_valid,
    output wire        in_ready,
    // Marks the last symbol pair of the frame. Triggers the final
    // traceback from state 0 (termination) and flushes what is left.
    input  wire        last,

    output reg         out_bit,
    output reg         out_valid,
    output reg         frame_done
);
    localparam integer NS       = 64;
    localparam integer TB_TOTAL = TB_DISCARD + TB_GROUP;
    localparam integer RAM_D    = (1 << RAM_AW);

    // ---- branch metric tables -------------------------------------
    // out1_a[ns] = parity(ns & 0o171), out2_a[ns] = parity(ns & 0o133).
    // Built at elaboration from the same expressions as the Python.
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

    // ---- path metrics ---------------------------------------------
    reg  [PM_W-1:0] pm   [0:NS-1];
    reg  [PM_W-1:0] pm_n [0:NS-1];
    reg  [NS-1:0]   dec_vec;        // per next-state: 0 = pred_a, 1 = pred_b

    // One loop variable PER always block. Sharing a single `integer i`
    // between the combinational ACS and the sequential block lets the
    // two clobber each other's loop index mid-iteration -- the ACS
    // reads pm[i>>1] while the sequential block is advancing i through
    // its own loop, which produces plausible-looking garbage.
    integer ia;   // ACS (combinational)
    integer ib;   // state update / init (sequential)

    // ---- survivor RAM ---------------------------------------------
    (* ram_style = "distributed" *)
    reg  [NS-1:0] sram [0:RAM_D-1];
    reg  [RAM_AW-1:0] wptr;
    reg  [RAM_AW:0]   fill;         // how many valid slices are stored

    // ---- control ---------------------------------------------------
    localparam [2:0] S_IDLE = 3'd0, S_ACS = 3'd1, S_TB_DISC = 3'd2,
                     S_TB_EMIT = 3'd3, S_POP = 3'd4, S_DONE = 3'd5;
    reg [2:0]  state;
    reg [5:0]  tb_state;            // current state during traceback
    reg [RAM_AW-1:0] tb_ptr;
    reg [RAM_AW:0]   tb_cnt;
    reg [RAM_AW:0]   lifo_n;
    // Sized TB_TOTAL, not TB_GROUP: the final flush emits everything
    // still pending, which is up to TB_TOTAL-1 steps, not one group.
    reg [TB_TOTAL-1:0] lifo;
    reg               saw_last;
    reg [RAM_AW:0]    pending;      // steps accumulated since last burst

    assign in_ready = (state == S_ACS);

    // Best state for a mid-stream traceback: arg-min over the 64 path
    // metrics, as a BALANCED TREE, registered.
    //
    // Written as a for-loop over 64 entries this synthesizes to a 63-deep
    // priority chain: measured 193 logic levels and 126 CARRY4 on the
    // pm -> tb_state path, WNS -110 ns at 100 MHz. A 6-level tree turns
    // the same function into log2(64) compare/mux stages.
    //
    // Registering it costs nothing in accuracy: the value is only read
    // when a burst starts, and a one-cycle-stale starting state is
    // irrelevant by construction -- TB_DISCARD exists precisely so that
    // the survivor paths have merged before any emitted bit depends on
    // where the walk began.
    // Split into TWO pipeline stages of three levels each. One flat
    // 6-level tree measured 15.6 ns (12 CARRY4), still 5.6 ns over at
    // 100 MHz; three levels of 8-bit compare fit with room.
    //
    // best_s is now two cycles stale, which remains harmless for the same
    // reason the single register was: TB_DISCARD guarantees the survivor
    // paths have merged before any EMITTED bit depends on the state the
    // walk started from. The starting state only has to be a good guess,
    // not the current-cycle optimum.
    wire [PM_W-1:0] a_m [0:NS-1];
    wire [5:0]      a_s [0:NS-1];
    reg  [PM_W-1:0] mid_m [0:7];
    reg  [5:0]      mid_s [0:7];
    wire [PM_W-1:0] b_m [0:7];
    wire [5:0]      b_s [0:7];

    genvar lv, gn;

    // --- stage A: 64 -> 8, combinational over three levels ------------
    generate
        for (gn = 0; gn < NS; gn = gn + 1) begin : gen_leaf
            assign a_m[gn] = pm[gn];
            assign a_s[gn] = gn[5:0];
        end
    endgenerate

    // Levels are written out rather than looped so each stage's width is
    // explicit and the pipeline cut is visible in the source.
    wire [PM_W-1:0] a1_m [0:31];  wire [5:0] a1_s [0:31];
    wire [PM_W-1:0] a2_m [0:15];  wire [5:0] a2_s [0:15];
    wire [PM_W-1:0] a3_m [0:7];   wire [5:0] a3_s [0:7];
    generate
        for (gn = 0; gn < 32; gn = gn + 1) begin : gen_a1
            wire hi = ($signed(a_m[2*gn+1] - a_m[2*gn]) < 0);
            assign a1_m[gn] = hi ? a_m[2*gn+1] : a_m[2*gn];
            assign a1_s[gn] = hi ? a_s[2*gn+1] : a_s[2*gn];
        end
        for (gn = 0; gn < 16; gn = gn + 1) begin : gen_a2
            wire hi = ($signed(a1_m[2*gn+1] - a1_m[2*gn]) < 0);
            assign a2_m[gn] = hi ? a1_m[2*gn+1] : a1_m[2*gn];
            assign a2_s[gn] = hi ? a1_s[2*gn+1] : a1_s[2*gn];
        end
        for (gn = 0; gn < 8; gn = gn + 1) begin : gen_a3
            wire hi = ($signed(a2_m[2*gn+1] - a2_m[2*gn]) < 0);
            assign a3_m[gn] = hi ? a2_m[2*gn+1] : a2_m[2*gn];
            assign a3_s[gn] = hi ? a2_s[2*gn+1] : a2_s[2*gn];
        end
    endgenerate

    integer ip;
    always @(posedge clk)
        for (ip = 0; ip < 8; ip = ip + 1) begin
            mid_m[ip] <= a3_m[ip];
            mid_s[ip] <= a3_s[ip];
        end

    // --- stage B: 8 -> 1, combinational over three levels --------------
    wire [PM_W-1:0] b1_m [0:3];   wire [5:0] b1_s [0:3];
    wire [PM_W-1:0] b2_m [0:1];   wire [5:0] b2_s [0:1];
    generate
        for (gn = 0; gn < 4; gn = gn + 1) begin : gen_b1
            wire hi = ($signed(mid_m[2*gn+1] - mid_m[2*gn]) < 0);
            assign b1_m[gn] = hi ? mid_m[2*gn+1] : mid_m[2*gn];
            assign b1_s[gn] = hi ? mid_s[2*gn+1] : mid_s[2*gn];
        end
        for (gn = 0; gn < 2; gn = gn + 1) begin : gen_b2
            wire hi = ($signed(b1_m[2*gn+1] - b1_m[2*gn]) < 0);
            assign b2_m[gn] = hi ? b1_m[2*gn+1] : b1_m[2*gn];
            assign b2_s[gn] = hi ? b1_s[2*gn+1] : b1_s[2*gn];
        end
    endgenerate
    wire b_hi = ($signed(b2_m[1] - b2_m[0]) < 0);
    assign b_m[0] = b_hi ? b2_m[1] : b2_m[0];
    assign b_s[0] = b_hi ? b2_s[1] : b2_s[0];

    reg [5:0] best_s;
    always @(posedge clk) best_s <= b_s[0];

    // ---- ACS combinational -----------------------------------------
    // For next-state ns: pred_a = ns>>1, pred_b = pred_a + 32.
    always @* begin
        for (ia = 0; ia < NS; ia = ia + 1) begin : acs
            reg [1:0]      bm_a, bm_b;
            reg [PM_W-1:0] ca, cb;
            bm_a = (sym[0] ^ o1a[ia]) + (sym[1] ^ o2a[ia]);
            bm_b = 2'd2 - bm_a;                       // complementary branch
            ca   = pm[ia >> 1]        + bm_a;
            cb   = pm[(ia >> 1) + 32] + bm_b;
            if ($signed(cb - ca) < 0) begin
                pm_n[ia]    = cb;
                dec_vec[ia] = 1'b1;
            end else begin
                pm_n[ia]    = ca;
                dec_vec[ia] = 1'b0;
            end
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state      <= S_IDLE;
            out_valid  <= 1'b0;
            frame_done <= 1'b0;
        end else begin
            out_valid  <= 1'b0;
            frame_done <= 1'b0;

            case (state)
            S_IDLE: begin
                if (start) begin
                    // Encoder starts at state 0 (zero-tail termination).
                    // Other states get a large-but-safe offset: big enough
                    // to lose against any real path, small enough that the
                    // modulo compare above stays unambiguous (<< 128).
                    pm[0] <= {PM_W{1'b0}};
                    for (ib = 1; ib < NS; ib = ib + 1) pm[ib] <= 8'd60;
                    wptr     <= {RAM_AW{1'b0}};
                    fill     <= {(RAM_AW+1){1'b0}};
                    pending  <= {(RAM_AW+1){1'b0}};
                    saw_last <= 1'b0;
                    state    <= S_ACS;
                end
            end

            S_ACS: begin
                if (in_valid) begin
                    for (ib = 0; ib < NS; ib = ib + 1) pm[ib] <= pm_n[ib];
                    sram[wptr] <= dec_vec;
                    wptr       <= wptr + 1'b1;
                    if (fill != TB_TOTAL) fill <= fill + 1'b1;
                    pending    <= pending + 1'b1;

                    if (last) begin
                        saw_last <= 1'b1;
                        // Terminated: the encoder is back at state 0.
                        tb_state <= 6'd0;
                        // wptr+1, not wptr: this cycle is still WRITING
                        // sram[wptr], so the newest valid slice is at
                        // wptr and the traceback head is one past it.
                        tb_ptr   <= wptr + 1'b1;
                        tb_cnt   <= {(RAM_AW+1){1'b0}};
                        state    <= S_TB_DISC;
                    // TB_TOTAL, not TB_GROUP. Emitting a group means
                    // tracing back TB_DISCARD from the head FIRST, so a
                    // burst needs TB_DISCARD+TB_GROUP unemitted steps
                    // behind it. Triggering on TB_GROUP re-emits a window
                    // that overlaps the previous burst.
                    end else if (pending + 1'b1 >= TB_TOTAL) begin
                        tb_state <= best_s;
                        // wptr+1, not wptr: this cycle is still WRITING
                        // sram[wptr], so the newest valid slice is at
                        // wptr and the traceback head is one past it.
                        tb_ptr   <= wptr + 1'b1;
                        tb_cnt   <= {(RAM_AW+1){1'b0}};
                        state    <= S_TB_DISC;
                    end
                end
            end

            // Walk back TB_DISCARD steps without emitting: this is the
            // depth that makes the survivor paths merge, so the bits
            // emitted after it no longer depend on which state we started
            // the walk from.
            S_TB_DISC: begin
                if (saw_last || tb_cnt >= TB_DISCARD) begin
                    tb_cnt <= {(RAM_AW+1){1'b0}};
                    lifo_n <= {(RAM_AW+1){1'b0}};
                    state  <= S_TB_EMIT;
                end else begin
                    tb_ptr   <= tb_ptr - 1'b1;
                    tb_state <= {sram[tb_ptr - 1'b1][tb_state], tb_state[5:1]};
                    tb_cnt   <= tb_cnt + 1'b1;
                end
            end

            // Keep walking back, now pushing the decoded bit for each
            // step. The decoded input bit at a step IS the low bit of the
            // state reached at that step (reg bit0 = input b).
            S_TB_EMIT: begin
                if (tb_cnt >= (saw_last ? pending : TB_GROUP)) begin
                    state <= S_POP;
                end else begin
                    lifo     <= {lifo[TB_TOTAL-2:0], tb_state[0]};
                    lifo_n   <= lifo_n + 1'b1;
                    tb_ptr   <= tb_ptr - 1'b1;
                    tb_state <= {sram[tb_ptr - 1'b1][tb_state], tb_state[5:1]};
                    tb_cnt   <= tb_cnt + 1'b1;
                end
            end

            // Traceback produced bits newest-first; the LIFO reverses them
            // back into transmission order.
            S_POP: begin
                if (lifo_n == 0) begin
                    pending <= pending - tb_cnt;
                    if (saw_last) begin
                        frame_done <= 1'b1;
                        state      <= S_IDLE;
                    end else begin
                        state <= S_ACS;
                    end
                end else begin
                    // Traceback pushed newest-first, so the shift register's
                    // bit 0 holds the OLDEST bit -- pop from there and shift
                    // right to emit in transmission order.
                    out_bit   <= lifo[0];
                    out_valid <= 1'b1;
                    lifo      <= lifo >> 1;
                    lifo_n    <= lifo_n - 1'b1;
                end
            end

            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
