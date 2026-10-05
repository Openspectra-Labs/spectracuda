// Frozen ad0a396 TX profile: host bytes -> block interleave -> conv_v27.
// clk_bit is 125 MHz. rst is synchronous. CRC/RS remain host-side.
// T1 bits[0] is FIRST. TRAIN is one n=0 token; HEADER/DATA have 216 groups.
//
// Two packet banks: the capture side (cfg + bytes + interleaver geometry)
// fills one bank while the emit side encodes the other, so the next frame
// is admitted while the current one is still going out. A frame is only
// emitted once it is completely captured (no RF-visible partial frame).
// The emit side prefetches the next interleaved byte while the current one
// is being encoded, so the encoder runs at one trellis step (2 coded bits)
// per clock and only waits on out_ready when a full group is to be issued.
`timescale 1ns/1ps
module tx_bit_domain #(
    parameter integer MAX_BYTES = 4095,
    parameter MASK_FILE = "rtl/generated/tx/header_mask.mem",
    parameter SELECT_FILE = "rtl/generated/tx/header_select.mem",
    parameter HEADER_FILL_FILE = "rtl/generated/tx/header_filler.mem",
    parameter PAYLOAD_FILL_FILE = "rtl/generated/tx/payload_filler.mem"
)(
    input wire clk_bit, rst,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [15:0] cfg_payload_bits,
    input wire [2:0] cfg_mod, // header codes: QPSK=1,16QAM=2,64QAM=3
    input wire [63:0] cfg_user,
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
    // capture side
    localparam C_IDLE=0, C_RECV=1, C_SQRT=2, C_COLS=3, C_COUNT=4;
    // emit side
    localparam E_IDLE=0, E_TRAIN=1, E_HEADER=2, E_FIRST=3, E_ENCODE=4, E_FINISH=5;

    (* ram_style="block" *) reg [7:0] packet [0:(2<<AW)-1];
    reg [0:0] mask [0:111], select_bit [0:215], header_fill [0:103];
    // Three modulation-specific 1296-bit sequences, zero-filled beyond bps*216.
    reg [0:0] payload_fill [0:3887];
    initial begin
        $readmemh(MASK_FILE, mask); $readmemh(SELECT_FILE, select_bit);
        $readmemh(HEADER_FILL_FILE, header_fill);
        $readmemh(PAYLOAD_FILL_FILE, payload_fill);
    end

    // ---- per-bank frame descriptors (written by capture, read by emit) ----
    reg        dvalid [0:1];
    reg [111:0] d_header [0:1];
    reg [2:0]  d_bps [0:1];
    reg [1:0]  d_frame [0:1];
    reg [12:0] d_units [0:1], d_rows [0:1], d_cols [0:1];
    reg [17:0] d_encoded [0:1];
    reg [7:0]  d_body [0:1];

    // ---- capture side ----
    reg [2:0]  cstate;
    reg        cbank;
    reg [12:0] c_units, received, c_rows, c_cols, col_acc;
    reg [25:0] square;
    reg [17:0] c_encoded, sym_acc;
    reg [2:0]  c_bps;
    reg [7:0]  c_body;
    wire [17:0] enc_request = ({2'b0,cfg_payload_bits}+18'd6)<<1;
    wire [2:0] request_bps = {cfg_mod[1:0],1'b0};
    wire [17:0] capacity = 18'd27648 * {15'd0,request_bps};
    wire legal = cfg_payload_bits != 0 && cfg_payload_bits[2:0]==0 &&
        32'(cfg_payload_bits[15:3])<=MAX_BYTES && cfg_mod>=1 && cfg_mod<=3 &&
        enc_request<=18'd65535 && enc_request<=capacity;
    assign cfg_ready = !rst && cstate==C_IDLE && !dvalid[cbank];
    assign in_ready = !rst && cstate==C_RECV;

    // ---- emit side ----
    reg [2:0]  estate;
    reg        ebank;
    reg [111:0] header;
    reg [2:0]  bps;
    reg [1:0]  frame;
    reg [12:0] units, rows, cols;
    reg [17:0] produced, encoded;
    reg [7:0]  body_syms, sc, sym, byte_q;
    reg [6:0]  hdr_content, hdr_filler;
    reg [2:0]  bit_idx, fill_count;
    reg [5:0]  conv_state, group;
    reg [10:0] filler_idx;
    reg [12:0] byte_count;
    // prefetch: (r, c, row_base) is the next interleaver cell to look at
    reg [12:0] r, c, row_base, fetched;
    reg        nxt_ok, fetching;
    reg [7:0]  byte_rd;
    wire [12:0] address = row_base+c;
    wire advance = !out_valid || out_ready;
    wire source_bit = byte_count < units ? byte_q[7-bit_idx] : 1'b0;
    wire [6:0] trellis = {conv_state,source_bit};
    wire first_code = ^(trellis & 7'o171);
    wire second_code = ^(trellis & 7'o133);
    wire [11:0] filler_base = (bps==2) ? 12'd0 : (bps==4 ? 12'd1296 : 12'd2592);
    wire code0 = produced<encoded ? first_code : payload_fill[filler_base+{1'b0,filler_idx}];
    wire code1 = produced<encoded ? second_code : payload_fill[filler_base+{1'b0,filler_idx}+12'd1];
    reg [5:0] next_group;
    always @* begin
        next_group=group;
        next_group[fill_count]=code0;
        next_group[fill_count+3'd1]=code1;
    end
    wire emit_group = fill_count+2==bps;
    // a byte boundary needs the prefetched byte unless the payload is done
    wire need_byte = produced<encoded && byte_count<units && bit_idx==7 &&
                     byte_count!=units-1;
    wire step = estate==E_ENCODE && (!emit_group || advance) && (!need_byte || nxt_ok);

    // capture-side memory write
    always @(posedge clk_bit)
        if (cstate==C_RECV && in_valid && !rst && in_last==(received==c_units-1))
            packet[{cbank,received[AW-1:0]}] <= in_byte;
    // emit-side memory read (registered: byte_rd valid together with nxt_ok)
    wire fetch = fetching && !nxt_ok && address<units;
    always @(posedge clk_bit)
        if (fetch) byte_rd <= packet[{ebank,address[AW-1:0]}];

    always @(posedge clk_bit) begin
        if (rst) begin
            cstate<=C_IDLE; cbank<=0; estate<=E_IDLE; ebank<=0;
            dvalid[0]<=0; dvalid[1]<=0;
            out_valid<=0; frame_done<=0; st_error<=0;
            out_bits<=0; out_n<=0; out_sc<=0; out_sym_idx<=0;
            out_stype<=0; out_fseq<=0; out_frame_start<=0; out_frame_end<=0;
            received<=0; c_units<=0; c_rows<=0; c_cols<=0; col_acc<=0; square<=0;
            c_encoded<=0; sym_acc<=0; c_bps<=0; c_body<=0;
            header<=0; bps<=0; frame<=0; units<=0; rows<=0; cols<=0;
            produced<=0; encoded<=0; body_syms<=0; sc<=0; sym<=0; byte_q<=0;
            hdr_content<=0; hdr_filler<=0; bit_idx<=0; fill_count<=0;
            conv_state<=0; group<=0; filler_idx<=0; byte_count<=0;
            r<=0; c<=0; row_base<=0; fetched<=0; nxt_ok<=0; fetching<=0;
        end else begin
            frame_done<=0;
            if (out_valid && out_ready) out_valid<=0;

            // ================= capture =================
            case (cstate)
            C_IDLE: if (cfg_valid && cfg_ready) begin
                if (!legal) st_error<=1;
                else begin
                    d_frame[cbank]<=cfg_fseq; c_bps<=request_bps; d_bps[cbank]<=request_bps;
                    c_units<=cfg_payload_bits[15:3]; received<=0;
                    c_encoded<=enc_request;
                    // Version1, big-endian raw length, mod, CRC none=1,
                    // fec0 none=0, fec1 conv_v27=1, eight user bytes.
                    d_header[cbank]<={8'd1,cfg_payload_bits,{5'd0,cfg_mod},8'h20,8'h01,cfg_user};
                    c_body<=0; sym_acc<=0;
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
                d_units[cbank]<=c_units; d_rows[cbank]<=c_rows; d_cols[cbank]<=c_cols;
                d_encoded[cbank]<=c_encoded; d_body[cbank]<=c_body;
                dvalid[cbank]<=1; cbank<=!cbank; cstate<=C_IDLE;
            end else begin sym_acc<=sym_acc+18'd216*{15'd0,c_bps}; c_body<=c_body+1; end
            default: begin cstate<=C_IDLE; st_error<=1; end
            endcase

            // ================= prefetch =================
            // Column-major walk of the rows x cols grid, skipping virtual
            // cells (address >= units); one cell per clock.
            if (fetching && !nxt_ok) begin
                if (address<units) begin
                    nxt_ok<=1; fetched<=fetched+1;
                    if (fetched==units-1) fetching<=0;
                end
                if (r==rows-1) begin r<=0; row_base<=0; c<=c+1; end
                else begin r<=r+1; row_base<=row_base+cols; end
            end

            // ================= emit =================
            case (estate)
            E_IDLE: if (dvalid[ebank]) begin
                header<=d_header[ebank]; bps<=d_bps[ebank]; frame<=d_frame[ebank];
                units<=d_units[ebank]; rows<=d_rows[ebank]; cols<=d_cols[ebank];
                encoded<=d_encoded[ebank]; body_syms<=d_body[ebank];
                r<=0; c<=0; row_base<=0; fetched<=0; nxt_ok<=0; fetching<=1;
                estate<=E_TRAIN;
            end
            E_TRAIN: if (advance) begin
                out_valid<=1; out_bits<=0; out_n<=0; out_sc<=0;
                out_sym_idx<=0; out_stype<=0; out_fseq<=frame;
                out_frame_start<=1; out_frame_end<=0;
                sc<=0; hdr_content<=0; hdr_filler<=0; estate<=E_HEADER;
            end
            E_HEADER: if (advance) begin
                out_valid<=1; out_n<=1; out_sc<=sc; out_sym_idx<=1;
                out_stype<=1; out_fseq<=frame; out_frame_start<=0; out_frame_end<=0;
                if (select_bit[sc]) begin
                    out_bits<={5'd0,header[111-hdr_content]^mask[hdr_content]};
                    hdr_content<=hdr_content+1;
                end else begin
                    out_bits<={5'd0,header_fill[hdr_filler]}; hdr_filler<=hdr_filler+1;
                end
                if (sc==215) begin
                    sc<=0; sym<=2; byte_count<=0; produced<=0; conv_state<=0;
                    fill_count<=0; group<=0; filler_idx<=0; bit_idx<=0;
                    estate<=E_FIRST;
                end else sc<=sc+1;
            end
            E_FIRST: if (nxt_ok) begin byte_q<=byte_rd; nxt_ok<=0; estate<=E_ENCODE; end
            E_ENCODE: if (step) begin
                group<=next_group; produced<=produced+2;
                if (produced<encoded) begin
                    conv_state<=trellis[5:0];
                    if (byte_count<units) begin
                        if (bit_idx==7) begin
                            byte_count<=byte_count+1; bit_idx<=0;
                            if (need_byte) begin byte_q<=byte_rd; nxt_ok<=0; end
                        end else bit_idx<=bit_idx+1;
                    end
                end else filler_idx<=filler_idx+2;
                if (emit_group) begin
                    out_valid<=1; out_bits<=next_group; out_n<=bps;
                    out_sc<=sc; out_sym_idx<=sym; out_stype<=3; out_fseq<=frame;
                    out_frame_start<=0;
                    out_frame_end<=sc==215 && sym==body_syms+1;
                    fill_count<=0; group<=0;
                    if (sc==215) begin
                        sc<=0; sym<=sym+1;
                        if (sym==body_syms+1) estate<=E_FINISH;
                    end else sc<=sc+1;
                end else fill_count<=fill_count+2;
            end
            E_FINISH: if (advance) begin
                dvalid[ebank]<=0; ebank<=!ebank; frame_done<=1; estate<=E_IDLE;
            end
            default: begin estate<=E_IDLE; st_error<=1; end
            endcase
        end
    end
endmodule
