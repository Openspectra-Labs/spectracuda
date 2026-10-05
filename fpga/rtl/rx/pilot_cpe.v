// ============================================================
// pilot_cpe.v -- per-symbol common phase error correction
//
// Port of spectracuda's CPE step
// (pipeline/ofdm.py:_decode_payload_from_header):
//
//     pilot_ratio      = equalized_pilots / pilot_values
//     pilot_ratio_mean = mean(pilot_ratio)          over n_pilot
//     cpe              = angle(pilot_ratio_mean)
//     cpe              = 0  if |pilot_ratio_mean| <= 0.05   (see below)
//     data             = data * exp(-1j * cpe)
//
// THREE THINGS THE PYTHON DOES THAT COST NOTHING HERE:
//
// 1. The divide by pilot_values is skipped. Every pilot is +1 for this
//    configuration; emit_rtl.py asserts that and sets GRID_PILOT_IS_ONE
//    rather than letting it be an assumption.
//
// 2. The mean is never formed. angle(sum/N) == angle(sum) for real N>0,
//    so the raw sum goes into the CORDIC. No divider, no rounding step.
//
// 3. The reliability gate is done on the SQUARED magnitude of the sum,
//    so it is integer-only and needs no CORDIC magnitude output (which
//    would have carried the CORDIC gain K and needed compensating).
//    |mean| > T  <=>  |sum|^2 > (T*N)^2, and emit_rtl.py emits that
//    constant so it tracks n_pilot.
//
// The gate matters: ofdm.py's comment records why. |pilot_ratio_mean|
// collapses toward 0 when the individual pilot phases disagree (deep
// fade, local corruption), and averaging near-random phases gives a
// confident-looking but meaningless angle. Such a symbol is left
// UNCORRECTED rather than rotated by noise.
//
// Data is buffered for a whole symbol because the correction for a
// symbol is derived from that same symbol's pilots -- the phase is not
// known until the last pilot has arrived.
//
// PING-PONG BUFFERED, and that is not an optimisation. A single buffer
// cannot accept symbol N+1 while it is still rotating symbol N, and in a
// real receiver the symbols arrive back to back: measured in rx_top,
// payload symbol 0 came out correlating 1.0000 with Python and symbols
// 1-3 correlated 0.04-0.09, because their data was being dropped while
// the rotator was busy. The standalone testbench had missed it entirely
// by leaving a gap between symbols -- a test being kinder than the
// system it stands in for.
// ============================================================
`timescale 1ns / 1ps
`include "grid_params.vh"

module pilot_cpe #(
    parameter integer DATA_W  = 18,
    parameter integer ANGLE_W = 16,
    parameter integer STAGES  = 16
)(
    input  wire                      clk,
    input  wire                      rst,

    // Equalized data subcarriers for one symbol.
    input  wire signed [DATA_W-1:0]  data_re,
    input  wire signed [DATA_W-1:0]  data_im,
    input  wire                      data_valid,

    // Equalized pilot subcarriers for the SAME symbol.
    input  wire signed [DATA_W-1:0]  pilot_re,
    input  wire signed [DATA_W-1:0]  pilot_im,
    input  wire                      pilot_valid,

    input  wire                      sym_done,   // all bins of this symbol seen

    output wire signed [DATA_W-1:0]  out_re,
    output wire signed [DATA_W-1:0]  out_im,
    output wire                      out_valid,
    output reg  signed [ANGLE_W-1:0] cpe_angle,  // applied rotation, in turns
    output reg                       cpe_valid
);
    localparam integer N_DATA  = `GRID_N_DATA;
    localparam integer N_PILOT = `GRID_N_PILOT;
    localparam integer ADDR_W  = $clog2(N_DATA);
    localparam integer SUM_W   = DATA_W + 4;          // headroom for N_PILOT adds
    localparam [ADDR_W-1:0] LAST_ADDR = ADDR_W'(N_DATA - 1);

    // ---- symbol buffers (ping-pong) ------------------------------------
    reg signed [DATA_W-1:0] buf_re [0:2*N_DATA-1];
    reg signed [DATA_W-1:0] buf_im [0:2*N_DATA-1];
    reg [ADDR_W-1:0] wr, rd;
    reg              fill_bank, proc_bank;

    // Bank base is N_DATA, NOT 2**ADDR_W. Indexing as {bank, addr} put
    // bank 1 at offset 256 in a 432-entry array, so the last 40 writes
    // of every odd symbol fell off the end -- which is exactly the 40
    // corrupted subcarriers that showed up at the top level. ADDR_W+1
    // bits, because N_DATA + (N_DATA-1) does not fit in ADDR_W.
    wire [ADDR_W:0] fill_addr =
        fill_bank ? ({1'b0, wr} + (ADDR_W+1)'(N_DATA)) : {1'b0, wr};
    wire [ADDR_W:0] proc_addr =
        proc_bank ? ({1'b0, rd} + (ADDR_W+1)'(N_DATA)) : {1'b0, rd};

    // ---- pilot accumulator ---------------------------------------------
    reg signed [SUM_W-1:0] sum_re, sum_im;
    reg signed [SUM_W-1:0] lat_re, lat_im;   // snapshot for the bank being rotated
    // One symbol of queue. The rotator needs ~216 + CORDIC latency ~= 249
    // cycles against a ~256-cycle symbol period, so it can still be
    // running when the next symbol completes. Preempting it there cost
    // the LAST 40 of 216 subcarriers on alternate symbols -- measured,
    // and invisible at block level because a standalone testbench feeds
    // symbols with gaps. So a finished symbol waits its turn instead.
    reg                    pend;
    reg                    pend_bank;
    reg signed [SUM_W-1:0] pend_re, pend_im;

    // ---- rotate-side state ----------------------------------------------
    localparam S_IDLE = 2'd0, S_ANGLE = 2'd1, S_ROT = 2'd2;
    reg [1:0] state;

    // Reliability gate on the CORDIC's OWN magnitude, not on a square.
    //
    // u_vec below already runs a vectoring CORDIC on exactly this vector
    // to get the angle, and vectoring produces K*|sum| as a by-product.
    // Squaring it instead cost two 22x22 multiplies (4 DSP), a 44-bit
    // compare and an 11.3 ns path -- for an ordering test, which a
    // magnitude answers just as well. emit_rtl.py folds the CORDIC gain
    // K into the constant.
    //
    // The magnitude arrives registered and in the same cycle as the
    // angle it gates, so the compare is register-to-register.
    wire signed [SUM_W+1:0] vec_mag;
    localparam signed [SUM_W+1:0] THRESH_MAG = (SUM_W+2)'(`GRID_CPE_THRESH_MAG);
    wire reliable = vec_mag > THRESH_MAG;

    reg                        vec_valid;
    wire signed [ANGLE_W-1:0]  vec_angle;
    wire                       vec_done;

    cordic_vec #(.DATA_W(SUM_W), .ANGLE_W(ANGLE_W), .STAGES(STAGES)) u_vec (
        .clk(clk), .rst(rst),
        .in_x(lat_re), .in_y(lat_im), .in_valid(vec_valid),
        .out_angle(vec_angle), .out_mag(vec_mag), .out_valid(vec_done));

    // TWO stages between the symbol buffer and the rotator. One was not
    // enough: Vivado merged the single register with cordic_rot's own
    // input register, so the BRAM's clock-to-out fed the CORDIC's
    // quadrant pre-rotation directly -- a 10.6 ns path. The buffer read
    // and the CORDIC input now sit in separate cycles.
    reg                       rot_valid, rot_valid_d;
    reg signed [DATA_W-1:0]   rot_x, rot_y, rot_x_d, rot_y_d;

    always @(posedge clk) begin
        if (rst) begin
            rot_x_d     <= {DATA_W{1'b0}};
            rot_y_d     <= {DATA_W{1'b0}};
            rot_valid_d <= 1'b0;
        end else begin
            rot_x_d     <= rot_x;
            rot_y_d     <= rot_y;
            rot_valid_d <= rot_valid;
        end
    end

    cordic_rot #(.DATA_W(DATA_W), .ANGLE_W(ANGLE_W), .STAGES(STAGES)) u_rot (
        .clk(clk), .rst(rst),
        .in_x(rot_x_d), .in_y(rot_y_d), .in_z(-cpe_angle), .in_valid(rot_valid_d),
        .out_x(out_re), .out_y(out_im), .out_valid(out_valid));

    // ---- FILL side: always accepting, whatever the rotator is doing ------
    always @(posedge clk) begin
        if (rst) begin
            wr <= {ADDR_W{1'b0}};
            sum_re <= {SUM_W{1'b0}};
            sum_im <= {SUM_W{1'b0}};
            fill_bank <= 1'b0;
        end else begin
            if (data_valid) begin
                buf_re[fill_addr] <= data_re;
                buf_im[fill_addr] <= data_im;
                wr <= (wr == LAST_ADDR) ? {ADDR_W{1'b0}} : wr + 1'b1;
            end
            if (pilot_valid) begin
                sum_re <= sum_re + {{(SUM_W-DATA_W){pilot_re[DATA_W-1]}}, pilot_re};
                sum_im <= sum_im + {{(SUM_W-DATA_W){pilot_im[DATA_W-1]}}, pilot_im};
            end
            if (sym_done) begin
                wr        <= {ADDR_W{1'b0}};
                sum_re    <= {SUM_W{1'b0}};
                sum_im    <= {SUM_W{1'b0}};
                fill_bank <= ~fill_bank;
            end
        end
    end

    // ---- ROTATE side: works on the bank the filler just finished ---------
    always @(posedge clk) begin
        if (rst) begin
            state     <= S_IDLE;
            rd        <= {ADDR_W{1'b0}};
            vec_valid <= 1'b0;
            rot_valid <= 1'b0;
            cpe_angle <= {ANGLE_W{1'b0}};
            cpe_valid <= 1'b0;
            proc_bank <= 1'b0;
            pend      <= 1'b0;
            pend_bank <= 1'b0;
            lat_re    <= {SUM_W{1'b0}};
            lat_im    <= {SUM_W{1'b0}};
        end else begin
            vec_valid <= 1'b0;
            rot_valid <= 1'b0;
            cpe_valid <= 1'b0;

            if (sym_done) begin
                // Snapshot this symbol's pilot sum before the filler
                // clears it for the next one.
                pend_re   <= sum_re + (pilot_valid
                             ? {{(SUM_W-DATA_W){pilot_re[DATA_W-1]}}, pilot_re}
                             : {SUM_W{1'b0}});
                pend_im   <= sum_im + (pilot_valid
                             ? {{(SUM_W-DATA_W){pilot_im[DATA_W-1]}}, pilot_im}
                             : {SUM_W{1'b0}});
                pend_bank <= fill_bank;
                pend      <= 1'b1;
            end

            case (state)
            S_IDLE: if (pend) begin
                lat_re    <= pend_re;
                lat_im    <= pend_im;
                proc_bank <= pend_bank;
                vec_valid <= 1'b1;
                pend      <= 1'b0;
                state     <= S_ANGLE;
            end
            S_ANGLE: if (vec_done) begin
                cpe_angle <= reliable ? vec_angle : {ANGLE_W{1'b0}};
                cpe_valid <= 1'b1;
                rd        <= {ADDR_W{1'b0}};
                state     <= S_ROT;
            end
            S_ROT: begin
                rot_x     <= buf_re[proc_addr];
                rot_y     <= buf_im[proc_addr];
                rot_valid <= 1'b1;
                if (rd == LAST_ADDR) begin
                    rd <= {ADDR_W{1'b0}};
                    // Straight into a queued symbol if one is waiting.
                    if (pend) begin
                        lat_re    <= pend_re;
                        lat_im    <= pend_im;
                        proc_bank <= pend_bank;
                        vec_valid <= 1'b1;
                        pend      <= 1'b0;
                        state     <= S_ANGLE;
                    end else state <= S_IDLE;
                end else rd <= rd + 1'b1;
            end
            default: ;
            endcase
        end
    end
endmodule
