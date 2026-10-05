// TX bit domain, protected-header (v3) profile with interleaver2 and DMRS.
// clk_bit is 125 MHz. rst is synchronous. CRC/RS of the PAYLOAD stay
// host-side. T1 bits[0] is FIRST.
//
// Frame on T1:   TRAIN token (sym 0)
//                HEADER  sym 1..2, 216 one-bit groups each (432 BPSK slots)
//                DATA    sym 3.., 216 groups of bps bits, with a DMRS token
//                        (stype 4, n=0) after every `interval` data symbols
//                        except after the last (spectracuda framing/dmrs.py)
//
// Header (hls/gen/hdr_v3_model.py == spectracuda HeaderCodec): 14 info
// bytes + crc16 (2 bytes, big-endian) -> conv_v27 with 6-bit tail = 268
// bits -> XOR scramble mask -> 268 of the 432 slots (select ROM), filler
// bits in the other 164.
//
// Payload: bytes -> outer byte block interleaver -> conv_v27 + tail ->
// padding filler to whole symbols -> interleaver2 per symbol (block, M x N,
// write row-major / read column-major, virtual cells dropped) -> groups.
//
// Structure: capture side fills one of two packet banks (and computes the
// frame's geometry + coded header) while the emit side sends the other.
// On the emit side the ENCODER writes coded bit pairs, one pair per clock,
// into a ping-pong pair of symbol buffers; the IL2 READER takes two bits
// per clock out of a full buffer in interleaved order and packs groups.
`timescale 1ns/1ps
module tx_bit_domain #(
    parameter integer MAX_BYTES = 4095,
    parameter H3_MASK_FILE = "reference/hdr3_mask.mem",
    parameter H3_SEL_FILE  = "reference/hdr3_sel.mem",
    parameter H3_FILL_FILE = "reference/hdr3_fill.mem",
    parameter PAYLOAD_FILL_FILE = "reference/payload_filler.mem"
)(
    input wire clk_bit, rst,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [15:0] cfg_payload_bits,
    input wire [2:0] cfg_mod,      // header codes: QPSK=1,16QAM=2,64QAM=3
    input wire [1:0] cfg_dmrs,     // header dmrs_period code: 0 off, 1/2/3 = every 16/32/64
    input wire [47:0] cfg_user,    // six user bytes, byte 0 in [47:40]
    input wire [1:0] cfg_fseq,
    input wire in_valid,
    output wire in_ready,
    input wire [7:0] in_byte,
    input wire in_last,
    output reg out_valid,
    input wire out_ready,
    output reg [5:0] out_bits,
    output reg [2:0] out_n,
    output reg [7:0] out_sc, out_sym_idx,
    output reg [2:0] out_stype,
    output reg [1:0] out_fseq,
    output reg out_frame_start, out_frame_end,
    output reg frame_done,
    output reg st_error
);
    localparam AW=$clog2(MAX_BYTES);
    localparam [2:0] ST_TRAIN=0, ST_HEADER=1, ST_DATA=3, ST_DMRS=4;
    // capture side
    localparam C_IDLE=0, C_RECV=1, C_SQRT=2, C_COLS=3, C_COUNT=4, C_CRC=5, C_CONV=6;
    // emit side
    localparam E_IDLE=0, E_TRAIN=1, E_HEADER=2, E_DATA=3, E_DMRS=4, E_FINISH=5;

    (* ram_style="block" *) reg [7:0] packet [0:(2<<AW)-1];
    reg [0:0] h3_mask [0:267], h3_sel [0:431], h3_fill [0:163];
    // Three modulation-specific 1296-bit sequences, zero-filled beyond bps*216.
    reg [0:0] payload_fill [0:3887];
    initial begin
        $readmemh(H3_MASK_FILE, h3_mask); $readmemh(H3_SEL_FILE, h3_sel);
        $readmemh(H3_FILL_FILE, h3_fill);
        $readmemh(PAYLOAD_FILL_FILE, payload_fill);
    end

    // ---- per-bank frame descriptors (written by capture, read by emit) ----
    reg        dvalid [0:1];
    reg [267:0] d_hwire [0:1];       // coded, UNscrambled header; bit k = k-th wire bit
    reg [2:0]  d_bps [0:1];
    reg [1:0]  d_frame [0:1], d_dmrs [0:1];
    reg [12:0] d_units [0:1], d_rows [0:1], d_cols [0:1];
    reg [17:0] d_encoded [0:1];
    reg [7:0]  d_ndata [0:1];

    // =====================================================================
    // capture side
    // =====================================================================
    reg [2:0]  cstate;
    reg        cbank;
    reg [12:0] c_units, received, c_rows, c_cols, col_acc;
    reg [25:0] square;
    reg [17:0] c_encoded, sym_acc;
    reg [2:0]  c_bps;
    reg [7:0]  c_ndata;
    reg [111:0] c_info;
    reg [31:0] c_crc;
    reg [3:0]  c_byte;
    reg [7:0]  c_step;               // conv step 0..133
    reg [127:0] c_cbits;             // info||crc, consumed MSB-first
    reg [5:0]  c_cst;
    reg [267:0] c_wire;
    wire [17:0] enc_request = ({2'b0,cfg_payload_bits}+18'd6)<<1;
    wire [2:0] request_bps = {cfg_mod[1:0],1'b0};
    // MAX_PAYLOAD_SYMBOLS = 128 counts data + DMRS: data ceiling per code
    wire [7:0] max_data = cfg_dmrs==0 ? 8'd128 : (cfg_dmrs==1 ? 8'd121 : (cfg_dmrs==2 ? 8'd125 : 8'd127));
    wire [17:0] capacity = 18'(max_data) * 18'd216 * {15'd0,request_bps};
    wire legal = cfg_payload_bits != 0 && cfg_payload_bits[2:0]==0 &&
        32'(cfg_payload_bits[15:3])<=MAX_BYTES && cfg_mod>=1 && cfg_mod<=3 &&
        enc_request<=18'd65535 && enc_request<=capacity;
    assign cfg_ready = !rst && cstate==C_IDLE && !dvalid[cbank];
    assign in_ready = !rst && cstate==C_RECV;

    function [31:0] crc_byte(input [31:0] c_in, input [7:0] d);
        integer i; reg [31:0] c;
        begin
            c = c_in ^ {24'd0, d};
            for (i = 0; i < 8; i = i + 1) c = c[0] ? ((c >> 1) ^ 32'h0000a001) : (c >> 1);
            crc_byte = c;
        end
    endfunction
    wire [15:0] c_key = ~c_crc[15:0];
    wire        c_in_bit = c_step < 128 ? c_cbits[127] : 1'b0;   // 6-bit zero tail
    wire [6:0]  c_reg = {c_cst, c_in_bit};

    // =====================================================================
    // emit side: descriptor in use
    // =====================================================================
    reg [2:0]  estate;
    reg        ebank;
    reg [267:0] hwire;
    reg [2:0]  bps;
    reg [1:0]  frame, dmrs;
    reg [12:0] units, rows, cols;
    reg [17:0] encoded;
    reg [7:0]  ndata;
    reg [8:0]  hs;                   // header slot 0..431
    reg [8:0]  hcontent;
    reg [7:0]  hfill;
    reg [7:0]  sc, sym, dsym;        // group in symbol, sym_idx, data symbols sent

    // ---- ENCODER: outer-interleaved bytes -> conv_v27 pairs -> bit buffers ----
    reg        enc_on, nxt_ok, fetching, ebk, byte_q_ok;
    reg [12:0] orr, occ, orow_base, fetched, byte_count;
    reg [7:0]  byte_rd, byte_q, esym;
    reg [2:0]  bit_idx;
    reg [5:0]  conv_state;
    reg [17:0] produced;
    reg [10:0] filler_idx;
    reg [9:0]  waddr;                // pair index within the symbol
    reg [1:0]  bfull;
    wire [12:0] address = orow_base+occ;
    wire source_bit = byte_count < units ? byte_q[7-bit_idx] : 1'b0;
    wire [6:0] trellis = {conv_state,source_bit};
    wire first_code = ^(trellis & 7'o171);
    wire second_code = ^(trellis & 7'o133);
    wire [11:0] filler_base = (bps==2) ? 12'd0 : (bps==4 ? 12'd1296 : 12'd2592);
    wire code0 = produced<encoded ? first_code : payload_fill[filler_base+{1'b0,filler_idx}];
    wire code1 = produced<encoded ? second_code : payload_fill[filler_base+{1'b0,filler_idx}+12'd1];
    wire [9:0] pairs_m1 = bps==2 ? 10'd215 : (bps==4 ? 10'd431 : 10'd647);
    wire need_byte = produced<encoded && byte_count<units && bit_idx==7 && byte_count!=units-1;
    wire enc_step = enc_on && !bfull[ebk] && (!need_byte || nxt_ok);
    (* ram_style="distributed" *) reg [1:0] bitbuf [0:2047];   // {bank, pair}
    always @(posedge clk_bit) if (enc_step) bitbuf[{ebk,waddr}] <= {code1,code0};

    // memory: capture write, prefetch read
    always @(posedge clk_bit)
        if (cstate==C_RECV && in_valid && !rst && in_last==(received==c_units-1))
            packet[{cbank,received[AW-1:0]}] <= in_byte;
    wire fetch = fetching && !nxt_ok && address<units;
    always @(posedge clk_bit)
        if (fetch) byte_rd <= packet[{ebank,address[AW-1:0]}];

    // ---- IL2 READER: column-major walk of the M x N grid, 2 cells/clock ----
    reg        rbk;
    // The pair being read is two REGISTERED cells, A = (ic, ir, ij) and
    // B = (bc, brw, bj), so both buffer reads use registered addresses (the
    // combinational next-cell -> read -> group path missed 125 MHz by
    // 0.37 ns in the full modem). Each step loads A' = next(B), B' = next(A').
    reg [5:0]  ic, ir;               // cell A (column, row)
    reg [10:0] ij;                   // its row-major index r*N + c
    reg [5:0]  bc, brw;              // cell B
    reg [10:0] bj;
    reg [2:0]  fill_count;
    reg [5:0]  group;
    wire [5:0] gM = bps==2 ? 6'd21 : (bps==4 ? 6'd30 : 6'd37);
    wire [5:0] gN = bps==2 ? 6'd21 : (bps==4 ? 6'd29 : 6'd36);
    wire [5:0] gF = bps==2 ? 6'd12 : (bps==4 ? 6'd23 : 6'd0);
    function [22:0] next_cell(input [5:0] c, input [5:0] r, input [10:0] j,
                              input [5:0] M, input [5:0] N, input [5:0] F);
        reg [5:0] len;
        begin
            len = (c < F) ? M : M - 6'd1;
            if (r + 6'd1 < len) next_cell = {c, r + 6'd1, j + 11'(N)};
            else                next_cell = {c + 6'd1, 6'd0, 11'(c) + 11'd1};
        end
    endfunction
    wire [22:0] cell_n  = next_cell(bc, brw, bj, gM, gN, gF);                       // A'
    wire [22:0] cell_nb = next_cell(cell_n[22:17], cell_n[16:11], cell_n[10:0], gM, gN, gF); // B'
    wire [1:0] wa = bitbuf[{rbk, ij[10:1]}];
    wire [1:0] wb = bitbuf[{rbk, bj[10:1]}];
    wire bit_a = wa[ij[0]], bit_b = wb[bj[0]];
    reg [5:0] next_group;
    always @* begin
        next_group = group;
        next_group[fill_count] = bit_a;
        next_group[fill_count+3'd1] = bit_b;
    end
    wire advance = !out_valid || out_ready;
    wire emit_group = fill_count+2==bps;
    wire rd_step = estate==E_DATA && bfull[rbk] && (!emit_group || advance);
    wire last_data = dsym==ndata-1;
    // DMRS after data symbol number dsym+1 (1-based) when it is a multiple
    // of the interval and not the last one (trailing DMRS suppressed)
    wire [7:0] dn = dsym + 8'd1;
    wire dmrs_after = dmrs==1 ? dn[3:0]==0 : (dmrs==2 ? dn[4:0]==0 : (dmrs==3 ? dn[5:0]==0 : 1'b0));

    always @(posedge clk_bit) begin
        if (rst) begin
            cstate<=C_IDLE; cbank<=0; estate<=E_IDLE; ebank<=0;
            dvalid[0]<=0; dvalid[1]<=0;
            out_valid<=0; frame_done<=0; st_error<=0;
            out_bits<=0; out_n<=0; out_sc<=0; out_sym_idx<=0;
            out_stype<=0; out_fseq<=0; out_frame_start<=0; out_frame_end<=0;
            received<=0; c_units<=0; c_rows<=0; c_cols<=0; col_acc<=0; square<=0;
            c_encoded<=0; sym_acc<=0; c_bps<=0; c_ndata<=0; c_info<=0; c_crc<=0;
            c_byte<=0; c_step<=0; c_cbits<=0; c_cst<=0; c_wire<=0;
            hwire<=0; bps<=0; frame<=0; dmrs<=0; units<=0; rows<=0; cols<=0;
            encoded<=0; ndata<=0; hs<=0; hcontent<=0; hfill<=0; sc<=0; sym<=0; dsym<=0;
            enc_on<=0; byte_q_ok<=0; nxt_ok<=0; fetching<=0; ebk<=0; orr<=0; occ<=0; orow_base<=0;
            fetched<=0; byte_count<=0; byte_q<=0; esym<=0; bit_idx<=0; conv_state<=0;
            produced<=0; filler_idx<=0; waddr<=0; bfull<=0;
            rbk<=0; ic<=0; ir<=0; ij<=0; bc<=0; brw<=0; bj<=0; fill_count<=0; group<=0;
        end else begin
            frame_done<=0;
            if (out_valid && out_ready) out_valid<=0;

            // ================= capture =================
            case (cstate)
            C_IDLE: if (cfg_valid && cfg_ready) begin
                if (!legal) st_error<=1;
                else begin
                    d_frame[cbank]<=cfg_fseq; d_dmrs[cbank]<=cfg_dmrs;
                    c_bps<=request_bps; d_bps[cbank]<=request_bps;
                    c_units<=cfg_payload_bits[15:3]; received<=0;
                    c_encoded<=enc_request;
                    // v3: version 3, length, mod, crc none(1)|fec0 none(0),
                    // dmrs|fec1 conv_v27(1), c2_len 0, six user bytes
                    c_info<={8'd3,cfg_payload_bits,{5'd0,cfg_mod},8'h20,
                             {1'b0,cfg_dmrs,5'd1},16'd0,cfg_user};
                    c_ndata<=0; sym_acc<=0;
                    cstate<=C_RECV;
                end
            end
            C_RECV: if (in_valid) begin
                if (in_last != (received==c_units-1)) begin
                    st_error<=1; cstate<=C_IDLE; // abort before any RF-visible output
                end else begin
                    received<=received+1;
                    if (received==c_units-1) begin
                        c_rows<=1; square<=1; cstate<=C_SQRT;
                    end
                end
            end
            // At exit rows=1+floor(sqrt(units)), exactly the Python rule.
            C_SQRT: if (square<= {13'd0,c_units}) begin
                square<=square+26'(2*c_rows+1); c_rows<=c_rows+1;
            end else begin c_cols<=0; col_acc<=0; cstate<=C_COLS; end
            C_COLS: if (col_acc>=c_units) cstate<=C_COUNT;
                else begin col_acc<=col_acc+c_rows; c_cols<=c_cols+1; end
            C_COUNT: if (sym_acc>=c_encoded) begin
                c_crc<=32'hffffffff; c_byte<=0; cstate<=C_CRC;
            end else begin sym_acc<=sym_acc+18'd216*{15'd0,c_bps}; c_ndata<=c_ndata+1; end
            // header crc16 over the 14 info bytes, one byte per clock
            C_CRC: if (c_byte==14) begin
                c_cbits<={c_info,~c_crc[15:0]}; c_step<=0; c_cst<=0; cstate<=C_CONV;
            end else begin
                c_crc<=crc_byte(c_crc, c_info[111-8*c_byte -: 8]); c_byte<=c_byte+1;
            end
            // header conv_v27 with tail: 134 steps, 2 wire bits each
            C_CONV: begin
                c_wire[2*c_step]   <= ^(c_reg & 7'o171);
                c_wire[2*c_step+1] <= ^(c_reg & 7'o133);
                c_cst<=c_reg[5:0]; c_cbits<={c_cbits[126:0],1'b0};
                if (c_step==133) begin
                    d_hwire[cbank]<={^(c_reg & 7'o133), ^(c_reg & 7'o171), c_wire[265:0]};
                    d_units[cbank]<=c_units; d_rows[cbank]<=c_rows; d_cols[cbank]<=c_cols;
                    d_encoded[cbank]<=c_encoded; d_ndata[cbank]<=c_ndata;
                    dvalid[cbank]<=1; cbank<=!cbank; cstate<=C_IDLE;
                end else c_step<=c_step+1;
            end
            default: begin cstate<=C_IDLE; st_error<=1; end
            endcase

            // ================= byte prefetch (outer interleaver) =================
            if (fetching && !nxt_ok) begin
                if (address<units) begin
                    nxt_ok<=1; fetched<=fetched+1;
                    if (fetched==units-1) fetching<=0;
                end
                if (orr==rows-1) begin orr<=0; orow_base<=0; occ<=occ+1; end
                else begin orr<=orr+1; orow_base<=orow_base+cols; end
            end

            // ================= encoder =================
            if (enc_on && produced==0 && byte_count==0 && bit_idx==0 && !byte_q_ok) begin
                // first byte of the frame
                if (nxt_ok) begin byte_q<=byte_rd; nxt_ok<=0; byte_q_ok<=1; end
            end else if (enc_step) begin
                produced<=produced+2;
                if (produced<encoded) begin
                    conv_state<=trellis[5:0];
                    if (byte_count<units) begin
                        if (bit_idx==7) begin
                            byte_count<=byte_count+1; bit_idx<=0;
                            if (need_byte) begin byte_q<=byte_rd; nxt_ok<=0; end
                        end else bit_idx<=bit_idx+1;
                    end
                end else filler_idx<=filler_idx+2;
                if (waddr==pairs_m1) begin
                    waddr<=0; bfull[ebk]<=1; ebk<=!ebk; esym<=esym+1;
                    if (esym==ndata-1) enc_on<=0;
                end else waddr<=waddr+1;
            end

            // ================= emit =================
            case (estate)
            E_IDLE: if (dvalid[ebank]) begin
                hwire<=d_hwire[ebank]; bps<=d_bps[ebank]; frame<=d_frame[ebank];
                dmrs<=d_dmrs[ebank]; units<=d_units[ebank]; rows<=d_rows[ebank];
                cols<=d_cols[ebank]; encoded<=d_encoded[ebank]; ndata<=d_ndata[ebank];
                orr<=0; occ<=0; orow_base<=0; fetched<=0; nxt_ok<=0; fetching<=1;
                enc_on<=1; byte_q_ok<=0; produced<=0; byte_count<=0; bit_idx<=0;
                conv_state<=0; filler_idx<=0; waddr<=0; esym<=0; ebk<=0; rbk<=0; bfull<=0;
                estate<=E_TRAIN;
            end
            E_TRAIN: if (advance) begin
                out_valid<=1; out_bits<=0; out_n<=0; out_sc<=0;
                out_sym_idx<=0; out_stype<=ST_TRAIN; out_fseq<=frame;
                out_frame_start<=1; out_frame_end<=0;
                hs<=0; sc<=0; hcontent<=0; hfill<=0; estate<=E_HEADER;
            end
            E_HEADER: if (advance) begin
                out_valid<=1; out_n<=1; out_sc<=sc; out_sym_idx<=hs<216 ? 8'd1 : 8'd2;
                out_stype<=ST_HEADER; out_fseq<=frame; out_frame_start<=0; out_frame_end<=0;
                if (h3_sel[hs]) begin
                    out_bits<={5'd0,hwire[hcontent]^h3_mask[hcontent]};
                    hcontent<=hcontent+1;
                end else begin
                    out_bits<={5'd0,h3_fill[hfill]}; hfill<=hfill+1;
                end
                sc<= sc==215 ? 8'd0 : sc+1;
                if (hs==431) begin
                    sym<=3; dsym<=0; ic<=0; ir<=0; ij<=0; bc<=0; brw<=6'd1; bj<=11'(gN); fill_count<=0; group<=0;
                    estate<=E_DATA;
                end
                hs<=hs+1;
            end
            E_DATA: if (rd_step) begin
                group<=next_group;
                {ic, ir, ij} <= cell_n; {bc, brw, bj} <= cell_nb;
                if (emit_group) begin
                    out_valid<=1; out_bits<=next_group; out_n<=bps;
                    out_sc<=sc; out_sym_idx<=sym; out_stype<=ST_DATA; out_fseq<=frame;
                    out_frame_start<=0;
                    out_frame_end<=sc==215 && last_data;
                    fill_count<=0; group<=0;
                    if (sc==215) begin
                        sc<=0; sym<=sym+1; dsym<=dsym+1;
                        bfull[rbk]<=0; rbk<=!rbk; ic<=0; ir<=0; ij<=0; bc<=0; brw<=6'd1; bj<=11'(gN);
                        if (last_data) estate<=E_FINISH;
                        else if (dmrs_after) estate<=E_DMRS;
                    end else sc<=sc+1;
                end else fill_count<=fill_count+2;
            end
            E_DMRS: if (advance) begin
                out_valid<=1; out_bits<=0; out_n<=0; out_sc<=0;
                out_sym_idx<=sym; out_stype<=ST_DMRS; out_fseq<=frame;
                out_frame_start<=0; out_frame_end<=0;
                sym<=sym+1; estate<=E_DATA;
            end
            E_FINISH: if (advance) begin
                dvalid[ebank]<=0; ebank<=!ebank; frame_done<=1; estate<=E_IDLE;
            end
            default: begin estate<=E_IDLE; st_error<=1; end
            endcase
        end
    end
endmodule
