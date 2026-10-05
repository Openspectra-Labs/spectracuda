// ============================================================
// il2_deint.v -- inner (frequency) de-interleaver, "interleaver2"
//
// Port of spectracuda's Ofdm interleaver2 = BlockInterleaver(n_bits =
// bits per OFDM symbol, unit_bits = 1), applied per symbol AFTER padding
// (pipeline/ofdm.py _apply_interleaver2). Encode writes the symbol's n
// coded bits row-major into M x N (M = 1 + floor(sqrt(n)), N = ceil(n/M))
// and reads column-major, dropping virtual cells. So de-interleaved bit
// j = r*N + c comes from interleaved position
//
//     inv(j) = base(c) + r,  base(c) = c*M                 for c <  F
//                                   = F*M + (c-F)*(M-1)    for c >= F
//     F = n - (M-1)*N  (valid cells in the last row)
//
// n = 216*bps: QPSK 432 (21x21, F 12), 16-QAM 864 (30x29, F 23),
// 64-QAM 1296 (37x36, F 0).
//
// THROUGHPUT: writes are linear, one LLR group (one subcarrier, bps LLRs)
// per clock, into a ping-pong pair of symbol buffers; reads are permuted,
// TWO LLRs per clock (j = 2p, 2p+1), i.e. one Viterbi pair per clock, so
// the decoder still runs at ~1 pair/clock. A position maps to the stored
// group and lane by divmod(inv, bps) (bps = 2/4/6; /6 as (i*43691)>>18,
// exact for i < 4096).
//
// Interface: a FWFT entry stream in (one subcarrier per entry, the coded-
// bit FIFO's format) and a FWFT entry stream out with ONE PAIR per entry
// ({first, n/2 = 1, {l1, l0}}), so the existing unpack / padding-drop
// logic after it is unchanged. Python pads before interleaving, so after
// de-interleaving the padding is back at the end of the frame.
// ============================================================
`timescale 1ns / 1ps

module il2_deint #(
    parameter integer LLR_W  = 4,
    parameter integer N_DATA = 216
)(
    input  wire                 clk,
    input  wire                 rst,
    // in: {first, n/2 (1..3), llr[6*LLR_W]}, one subcarrier per entry
    input  wire                 in_hv,
    input  wire [6*LLR_W+2:0]   in_head,
    output wire                 in_pop,
    // out: same format, exactly one pair per entry
    output reg                  out_hv,
    output reg  [6*LLR_W+2:0]   out_head,
    input  wire                 out_pop
);
    localparam integer GW = 6 * LLR_W;

    // ---- two symbol buffers ------------------------------------------
    (* ram_style = "distributed" *) reg [GW-1:0] buf0 [0:N_DATA-1];
    (* ram_style = "distributed" *) reg [GW-1:0] buf1 [0:N_DATA-1];
    reg        full  [0:1];
    reg [1:0]  bh    [0:1];           // n/2 of that symbol
    reg        bfst  [0:1];           // symbol begins a frame
    reg        wb, rb;                // write bank, read bank
    reg [7:0]  wcnt;

    wire       in_first = in_head[GW+2];
    wire [1:0] in_half  = in_head[GW+1:GW];
    assign in_pop = in_hv && !full[wb];

    // ---- read side: walk (r, c) row-major, two cells per clock --------
    // Three registered stages so the walk, the divmod-by-bps and the buffer
    // read are each one cycle at 125 MHz (the single-cycle version missed
    // by 6.4 ns once IL2 was turned on):
    //   S0 walk -> inv0/inv1   S1 -> (group, lane)   S2 buffer read -> out FIFO
    // Credit: the walk only advances while the output FIFO can absorb
    // everything already in flight. The bank is released when its LAST
    // pair leaves S2, not when the walk ends, so the writer can never
    // overwrite a bank S2 is still reading.
    reg  [5:0]  M, Nn, F;
    reg  [10:0] npairs, pcnt;
    reg  [5:0]  r, c;
    reg  [10:0] base;                 // base(c)
    reg         rd_act;

    function [10:0] nbase(input [10:0] b, input [5:0] cc, input [5:0] FF, input [5:0] MM);
        nbase = b + ((cc < FF) ? MM : MM - 1);
    endfunction

    // second cell of the pair: (r, c+1), or (r+1, 0) on wrap
    wire        wrap1 = (c == Nn - 1);
    wire [10:0] inv0  = base + r;
    wire [10:0] inv1  = wrap1 ? (r + 1) : (nbase(base, c, F, M) + r);

    function [8:0] gdiv(input [10:0] i, input [2:0] b);
        case (b)
            3'd2:    gdiv = i[10:1];
            3'd4:    gdiv = i[10:2];
            // 32-bit product: Verilog sizes a product to its widest OPERAND,
            // so i * 18'd43691 overflowed 18 bits (64-QAM failed).
            default: gdiv = ({21'd0, i} * 32'd43691) >> 18;
        endcase
    endfunction

    // output FIFO + credit
    localparam integer OF_D = 8;
    wire            of_hv;
    wire [GW+2:0]   of_head;
    wire [$clog2(OF_D):0] of_level;
    reg             s1_v, s2_v;
    wire [3:0]      inflight = {3'd0, s1_v} + {3'd0, s2_v};
    wire            room = (of_level + inflight) < OF_D - 1;
    wire            emit = rd_act && room;

    // S0 -> S1
    reg  [10:0] s1_i0, s1_i1;
    reg         s1_bank, s1_first, s1_last;
    reg  [2:0]  s1_bps;
    // S1 -> S2
    reg  [8:0]  s2_g0, s2_g1;
    reg  [2:0]  s2_l0, s2_l1;
    reg         s2_bank, s2_first, s2_last;
    wire [8:0]  g0 = gdiv(s1_i0, s1_bps), g1 = gdiv(s1_i1, s1_bps);
    wire [2:0]  l0i = s1_i0 - g0 * s1_bps, l1i = s1_i1 - g1 * s1_bps;
    // S2: buffer read
    wire [GW-1:0] w0 = s2_bank ? buf1[s2_g0] : buf0[s2_g0];
    wire [GW-1:0] w1 = s2_bank ? buf1[s2_g1] : buf0[s2_g1];
    wire [LLR_W-1:0] ll0 = w0[s2_l0*LLR_W +: LLR_W];
    wire [LLR_W-1:0] ll1 = w1[s2_l1*LLR_W +: LLR_W];

    sync_fifo_fwft #(.WIDTH(GW+3), .DEPTH(OF_D)) u_of (
        .clk(clk), .rst(rst),
        .push(s2_v), .push_data({s2_first, 2'd1, {(GW - 2*LLR_W){1'b0}}, ll1, ll0}),
        .head_valid(of_hv), .head_data(of_head), .pop(of_hv && out_pop),
        .level(of_level), .full(), .overflow(), .underflow());
    always @* begin out_hv = of_hv; out_head = of_head; end

    always @(posedge clk) begin
        if (in_pop) begin
            if (wb) buf1[wcnt] <= in_head[GW-1:0];
            else    buf0[wcnt] <= in_head[GW-1:0];
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            full[0] <= 1'b0; full[1] <= 1'b0; wb <= 1'b0; rb <= 1'b0; wcnt <= 0;
            rd_act <= 1'b0; s1_v <= 1'b0; s2_v <= 1'b0;
        end else begin
            // ---- write a symbol into bank wb ----
            if (in_pop) begin
                if (wcnt == 0) begin bh[wb] <= in_half; bfst[wb] <= in_first; end
                if (wcnt == N_DATA - 1) begin
                    full[wb] <= 1'b1; wb <= ~wb; wcnt <= 0;
                end else wcnt <= wcnt + 1'b1;
            end

            // ---- start walking a full bank (not one S2 is still draining) ----
            if (!rd_act && full[rb] && !(s1_v && s1_bank == rb) && !(s2_v && s2_bank == rb)) begin
                rd_act <= 1'b1; r <= 0; c <= 0; base <= 0; pcnt <= 0;
                case (bh[rb])
                    2'd1:    begin M <= 21; Nn <= 21; F <= 12; npairs <= 216; end
                    2'd2:    begin M <= 30; Nn <= 29; F <= 23; npairs <= 432; end
                    default: begin M <= 37; Nn <= 36; F <= 0;  npairs <= 648; end
                endcase
            end

            // ---- S0: walk ----
            s1_v <= emit;
            if (emit) begin
                s1_i0 <= inv0; s1_i1 <= inv1; s1_bank <= rb; s1_bps <= {bh[rb], 1'b0};
                s1_first <= (pcnt == 0) && bfst[rb]; s1_last <= (pcnt == npairs - 1);
                if (wrap1) begin
                    r <= r + 1; c <= 1; base <= nbase(0, 0, F, M);
                end else if (c + 1 == Nn - 1) begin
                    r <= r + 1; c <= 0; base <= 0;
                end else begin
                    c <= c + 2; base <= nbase(nbase(base, c, F, M), c + 1, F, M);
                end
                if (pcnt == npairs - 1) begin rd_act <= 1'b0; rb <= ~rb; end
                pcnt <= pcnt + 1'b1;
            end
            // ---- S1: divmod ----
            s2_v <= s1_v;
            if (s1_v) begin
                s2_g0 <= g0; s2_g1 <= g1; s2_l0 <= l0i; s2_l1 <= l1i;
                s2_bank <= s1_bank; s2_first <= s1_first; s2_last <= s1_last;
            end
            // ---- S2: read + push; release the bank after its last pair ----
            if (s2_v && s2_last) full[s2_bank] <= 1'b0;
        end
    end
endmodule
