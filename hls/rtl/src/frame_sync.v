// ============================================================
// frame_sync.v -- frame detection and sample replay
//
// The piece that turns sc_sync's per-candidate P/R stream into "a frame
// starts HERE, and here are its samples from the start".
//
// Models spectracuda's rx_streaming() SEEKING state
// (pipeline/ofdm.py:1395), not the batch rx_process() path: batch mode
// is a plain argmax over a whole captured buffer, which a streaming
// receiver cannot do.
//
// NO DIVIDER AND NO SQUARING. The metric is |P|^2/R^2 and the decisions
// on it are a threshold and an argmax -- both orderings, and ordering
// survives a monotonic map, so
//
//     |P|^2 / R^2 > T      <=>      |P| > sqrt(T) * R
//
// Working in MAGNITUDE rather than magnitude-squared is what makes this
// block small. The squared form needed three 48x48 multiplies, a 96-bit
// dynamic normaliser (leading-one search + barrel shift, because |P|^2
// swings with signal level) and two cross-multiplies -- 31 DSPs and a
// 33 ns path, WNS -22.8 ns. openofdm does exactly this instead
// (sync_short.v:254): a pipelined magnitude, then a threshold built from
// SHIFTS of the energy, compared register-to-register.
//
// sqrt(0.3) = 0.54772, and the CORDIC returns K*|P| with
// K = 1.6468, so the constant folds to K*sqrt(T) = 0.9020, which is
//     1/2 + 1/4 + 1/8 + 1/32 = 0.90625
// three shifts and three adds, 0.5% high. Irrelevant: pure noise scores
// <= 0.21 on this metric and real signal >= 0.74 (Ofdm's own
// DEFAULT_SYNC_THRESHOLD note), so the 0.3 decision sits in a wide gap.
//
// TRAILING-EDGE GUARD -- this is load-bearing, not caution.
// ofdm.py:1420 records the measurement: a preamble only PARTIALLY in the
// buffer (k of fft_size samples, k > L) already scores (2(k-L)/k)^2 at
// the last candidate offset -- 0.44 at 3/4 present, 0.73 at 7/8, both
// over the 0.3 threshold -- with a start_index that is (fft_size - k)
// samples EARLY. Early by more than the CP means ISI and a failed
// decode: up to ~30% of alignments lost at chunk=64. A partial preamble
// can never out-score the full one, so a detection is not accepted until
// L further candidates have been seen past the current best. Costs L
// samples of latency, which is free -- the header needs more than that
// anyway.
// ============================================================
`timescale 1ns / 1ps
`include "ofdm_params.vh"

module frame_sync #(
    parameter integer SAMPLE_W = 16,
    parameter integer ACC_W    = 48,
    parameter integer BUF_W    = 12,   // ring buffer depth = 2**BUF_W samples
    parameter integer FRAME_LEN = 1984,
    // Threshold as a ratio, default 0.3 = 3/10 (DEFAULT_SYNC_THRESHOLD).
    parameter integer THRESH_NUM = 3,
    parameter integer THRESH_DEN = 10,
    // sc_sync_rtl is a 3-stage pipeline (s0 fetch / s1 multiply /
    // s2 accumulate), so its out_valid for candidate d lands 3 cycles
    // after the sample that completes d. Without this the replay starts
    // 3 samples late -- measured, not assumed: it showed up as a
    // start_index of 203 against Python's 200.
    parameter integer SC_LATENCY = 3,
    parameter integer STAGES     = 16   // cordic_vec depth for |P|
)(
    input  wire                        clk,
    input  wire                        rst,

    // Raw sample stream (same stream sc_sync is fed).
    input  wire signed [SAMPLE_W-1:0]  in_i,
    input  wire signed [SAMPLE_W-1:0]  in_q,
    input  wire                        in_valid,

    // sc_sync's per-candidate output. Candidate d corresponds to the
    // sample written 2*LAG cycles earlier.
    input  wire signed [ACC_W-1:0]     p_re,
    input  wire signed [ACC_W-1:0]     p_im,
    input  wire signed [ACC_W-1:0]     r_sum,
    input  wire                        sc_valid,

    output reg  signed [SAMPLE_W-1:0]  out_i,
    output reg  signed [SAMPLE_W-1:0]  out_q,
    output reg                         out_valid,
    output reg                         frame_start,   // with the first sample
    output reg                         detected,      // pulses on acceptance
    output reg  [BUF_W-1:0]            start_index
);
    localparam integer LAG   = `SC_LAG;          // fft_size/2
    localparam integer DEPTH = 1 << BUF_W;

    // ---- sample ring ----------------------------------------------------
    reg signed [SAMPLE_W-1:0] buf_i [0:DEPTH-1];
    reg signed [SAMPLE_W-1:0] buf_q [0:DEPTH-1];
    reg [BUF_W-1:0] wr;

    always @(posedge clk) begin
        if (rst) wr <= {BUF_W{1'b0}};
        else if (in_valid) begin
            buf_i[wr] <= in_i;
            buf_q[wr] <= in_q;
            wr        <= wr + 1'b1;
        end
    end

    // ---- magnitude of P, and the energy it is judged against --------
    localparam integer VEC_LAT = STAGES + 1;   // 1 load cycle + STAGES stages

    wire signed [ACC_W+1:0] mag;
    wire                    mag_valid;

    cordic_vec #(.DATA_W(ACC_W), .ANGLE_W(16), .STAGES(STAGES)) u_mag (
        .clk(clk), .rst(rst),
        .in_x(p_re), .in_y(p_im), .in_valid(sc_valid),
        .out_angle(), .out_mag(mag), .out_valid(mag_valid));

    // R and the candidate index have to arrive with the magnitude, so
    // they ride a delay line of the same depth, advanced by the same
    // enable that feeds the CORDIC.
    wire signed [ACC_W-1:0] r_half = r_sum >>> 1;
    reg  signed [ACC_W-1:0] r_pipe   [0:VEC_LAT-1];
    reg         [BUF_W-1:0] idx_pipe [0:VEC_LAT-1];

    // Candidate d's first sample sits 2*LAG writes back, plus sc_sync's
    // own 3-stage pipeline.
    wire [BUF_W-1:0] cand_idx_now = wr - BUF_W'(2*LAG + SC_LATENCY);

    integer k;
    always @(posedge clk) begin
        if (rst) begin
            for (k = 0; k < VEC_LAT; k = k + 1) begin
                r_pipe[k]   <= {ACC_W{1'b0}};
                idx_pipe[k] <= {BUF_W{1'b0}};
            end
        end else if (sc_valid) begin
            r_pipe[0]   <= r_half;
            idx_pipe[0] <= cand_idx_now;
            for (k = 1; k < VEC_LAT; k = k + 1) begin
                r_pipe[k]   <= r_pipe[k-1];
                idx_pipe[k] <= idx_pipe[k-1];
            end
        end
    end

    wire signed [ACC_W-1:0] r_now   = r_pipe[VEC_LAT-1];
    wire        [BUF_W-1:0] cand_idx = idx_pipe[VEC_LAT-1];

    // K*sqrt(0.3)*R as shifts: 1/2 + 1/4 + 1/8 + 1/32.
    //
    // EVERY SHIFT IS PARENTHESISED. Verilog binds binary + TIGHTER than
    // >>>, so `r >>> 1 + r >>> 2` parses as `r >>> (1 + r) >>> ...` --
    // a shift by an enormous amount, i.e. zero. That bug survived the
    // functional test: with the threshold stuck at zero every candidate
    // passed, and the argmax on |P| still picked the right peak. It only
    // showed up as unexplained timing loss.
    //
    // REGISTERED, like openofdm's prod_thres (sync_short.v:254), so the
    // compare below is register-to-register instead of dragging three
    // 50-bit adds into the same path as the CORDIC output.
    wire signed [ACC_W+1:0] r_ext = {{2{r_now[ACC_W-1]}}, r_now};
    wire signed [ACC_W+1:0] thresh_c = (r_ext >>> 1) + (r_ext >>> 2)
                                     + (r_ext >>> 3) + (r_ext >>> 5);

    reg  signed [ACC_W+1:0] thresh, mag_q;
    reg                     mag_valid_q;
    reg         [BUF_W-1:0] cand_idx_q;

    always @(posedge clk) begin
        if (rst) begin
            thresh      <= {(ACC_W+2){1'b0}};
            mag_q       <= {(ACC_W+2){1'b0}};
            mag_valid_q <= 1'b0;
            cand_idx_q  <= {BUF_W{1'b0}};
        end else begin
            thresh      <= thresh_c;
            mag_q       <= mag;
            mag_valid_q <= mag_valid;
            cand_idx_q  <= cand_idx;
        end
    end

    wire over_thresh = mag_valid_q && (mag_q > thresh);

    // Argmax on |P| alone. R is the received energy over a 2L window and
    // is essentially flat across the preamble, so argmax(|P|/R) and
    // argmax(|P|) pick the same candidate there -- VALIDATED against
    // Python's start_index on the real frames rather than assumed. It
    // removes the last two multiplies.
    wire better = mag_q > best_mag;

    // ---- SEEKING / REPLAY ------------------------------------------------
    localparam S_SEEK = 1'b0, S_REPLAY = 1'b1;
    reg state;

    reg signed [ACC_W+1:0] best_mag;
    reg [BUF_W-1:0] best_idx;
    reg             have_best;
    reg [BUF_W-1:0] since_best;     // candidates seen since the current best
    reg [BUF_W-1:0] rd;
    reg [15:0]      replay_cnt;

    always @(posedge clk) begin
        if (rst) begin
            state       <= S_SEEK;
            best_mag    <= {(ACC_W+2){1'b0}};
            best_idx    <= {BUF_W{1'b0}};
            have_best   <= 1'b0;
            since_best  <= {BUF_W{1'b0}};
            rd          <= {BUF_W{1'b0}};
            replay_cnt  <= 16'd0;
            out_valid   <= 1'b0;
            frame_start <= 1'b0;
            detected    <= 1'b0;
            start_index <= {BUF_W{1'b0}};
        end else begin
            out_valid   <= 1'b0;
            frame_start <= 1'b0;
            detected    <= 1'b0;

            case (state)
            S_SEEK: begin
                if (mag_valid_q) begin
                    if (over_thresh && (!have_best || better)) begin
                        best_mag   <= mag_q;
                        best_idx   <= cand_idx_q;
                        have_best  <= 1'b1;
                        since_best <= {BUF_W{1'b0}};
                    end else if (have_best) begin
                        since_best <= since_best + 1'b1;
                        // Trailing-edge guard: L further candidates with
                        // nothing better means the peak is real and whole.
                        if (since_best == BUF_W'(LAG-1)) begin
                            detected    <= 1'b1;
                            start_index <= best_idx;
                            rd          <= best_idx;
                            replay_cnt  <= 16'd0;
                            state       <= S_REPLAY;
                        end
                    end
                end
            end

            S_REPLAY: begin
                out_i       <= buf_i[rd];
                out_q       <= buf_q[rd];
                out_valid   <= 1'b1;
                frame_start <= (replay_cnt == 16'd0);
                rd          <= rd + 1'b1;
                if (replay_cnt == 16'(FRAME_LEN-1)) begin
                    have_best  <= 1'b0;
                    best_mag   <= {(ACC_W+2){1'b0}};
                    since_best <= {BUF_W{1'b0}};
                    state      <= S_SEEK;
                end else begin
                    replay_cnt <= replay_cnt + 1'b1;
                end
            end
            endcase
        end
    end
endmodule
