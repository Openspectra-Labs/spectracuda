// ============================================================
// header_decode_v3.v -- protected frame header (PROTOCOL_VERSION 3)
//
// Port of spectracuda HeaderCodec.decode_bits at the v3 pin
// (hls/rtl/golden_ref_v3.py; plain-integer model hls/gen/hdr_v3_model.py):
//
//   432 header slots (2 BPSK symbols) --select--> 268 wire bits
//   wire ^ mask --> conv_v27 hard Viterbi (134 steps, zero tail) --> 128 bits
//   128 = 112 info bits || crc16 (big-endian), crc16 = liquid crc
//   (32-bit reflected register, poly 0xA001, init all-ones, complemented)
//
//   byte 0 version | 1-2 payload_len_bits | 3 mod_scheme | 4 crc(3)|fec0(5)
//   byte 5 rsvd(1)|dmrs_period(2)|fec1(5) | 6-7 c2_len_bytes | 8-13 user
//
// fields_valid = crc ok AND every code known (decode_bits RAISES on either:
// a false sync or a corrupted header must not configure the payload path).
// The Viterbi is the validated overlapped hard decoder (viterbi_dec_ovl.v,
// equivalent to viterbi_dec.v and timing-closed at 125 MHz), a private
// instance: the header is decoded before the frame's config exists, so it
// cannot share the payload decoder without stalling the payload path.
// ============================================================
`timescale 1ns / 1ps
`include "header_params.vh"

module header_decode_v3 (
    input  wire        clk,
    input  wire        rst,
    input  wire        in_bit,
    input  wire        in_valid,
    input  wire        in_sof,      // first slot of a header
    output reg         done,
    output reg         fields_valid,
    output reg         crc_ok,
    output reg  [7:0]  protocol_version,
    output reg  [15:0] payload_len_bits,
    output reg  [7:0]  mod_scheme,
    output reg  [3:0]  bits_per_symbol,
    output reg  [2:0]  crc_code,
    output reg  [4:0]  fec0_code,
    output reg  [4:0]  fec1_code,
    output reg  [1:0]  dmrs_code,
    output reg  [15:0] c2_len_bytes,
    output reg  [47:0] user_data
);
    localparam integer N_WIRE  = `HDR_WIRE_BITS;     // 268
    localparam integer N_SLOTS = `HDR_TOTAL_SLOTS;   // 432
    localparam integer N_STEP  = N_WIRE / 2;         // 134
    localparam integer N_DEC   = 128;                // info || crc
    localparam integer SLOT_W  = $clog2(N_SLOTS);

    reg [0:0] mask [0:N_WIRE-1];
    reg [0:0] sel  [0:N_SLOTS-1];
    reg [7:0] bps_rom [0:`HDR_MOD_CODE_MAX];
    initial begin
        $readmemh(`HDR_MASK_MEM, mask);
        $readmemh(`HDR_SEL_MEM, sel);
        $readmemh(`HDR_BPS_MEM, bps_rom);
    end

    localparam S_COLLECT = 0, S_FEED = 1, S_WAIT = 2, S_CRC = 3, S_OUT = 4;
    reg [2:0]        state;
    reg [N_WIRE-1:0] wbits;        // descrambled wire bit k at [k]
    reg [8:0]        cnt;          // wire bits taken
    reg [SLOT_W-1:0] slot;
    reg [7:0]        step;         // pairs fed to the Viterbi
    reg [N_DEC-1:0]  dec;          // decoded bits, first at [N_DEC-1]
    reg [7:0]        nout;
    reg [31:0]       crc;
    reg [3:0]        cbyte;

    // ---- Viterbi ----
    wire       v_ready, v_bit, v_valid, v_done;
    reg        v_start;
    wire       v_push = (state == S_FEED) && v_ready;
    viterbi_dec_ovl u_vit (
        .clk(clk), .rst(rst), .start(v_start),
        .sym(wbits[1:0]), .in_valid(state == S_FEED), .in_ready(v_ready),
        .last(step == N_STEP - 1),
        .out_bit(v_bit), .out_valid(v_valid), .frame_done(v_done));

    function [31:0] crc_byte(input [31:0] c_in, input [7:0] d);
        integer i; reg [31:0] c;
        begin
            c = c_in ^ {24'd0, d};
            for (i = 0; i < 8; i = i + 1) c = c[0] ? ((c >> 1) ^ 32'h0000a001) : (c >> 1);
            crc_byte = c;
        end
    endfunction
    function [7:0] dbyte(input integer k);   // k-th decoded byte, MSB-first
        dbyte = dec[N_DEC-1-8*k -: 8];
    endfunction

    wire [7:0]  b3 = dbyte(3), b4 = dbyte(4), b5 = dbyte(5);
    wire [31:0] MOD_VALID = `HDR_MOD_VALID;
    wire [7:0]  CRC_VALID = `HDR_CRC_VALID;
    wire [31:0] FEC_VALID = `HDR_FEC_VALID;
    wire mod_ok  = (b3 <= `HDR_MOD_CODE_MAX) && MOD_VALID[b3[4:0]];
    wire codes_ok = mod_ok && CRC_VALID[b4[7:5]] && FEC_VALID[b4[4:0]] && FEC_VALID[b5[4:0]];
    wire [15:0] key = {dbyte(14), dbyte(15)};

    always @(posedge clk) begin
        if (rst) begin
            state <= S_COLLECT; cnt <= 0; slot <= 0; step <= 0; nout <= 0;
            v_start <= 1'b0; done <= 1'b0; fields_valid <= 1'b0; crc_ok <= 1'b0;
        end else begin
            done <= 1'b0; v_start <= 1'b0;
            // a new header always restarts collection (a stuck decode of a
            // previous, truncated header must not swallow it)
            if (in_valid && in_sof) begin
                state <= S_COLLECT; slot <= SLOT_W'(1);
                if (sel[0]) begin wbits[0] <= in_bit ^ mask[0]; cnt <= 9'd1; end
                else cnt <= 9'd0;
            end else case (state)
            S_COLLECT: if (in_valid) begin
                slot <= slot + 1'b1;
                if (sel[slot] && cnt < N_WIRE) begin
                    wbits[cnt] <= in_bit ^ mask[cnt];
                    cnt <= cnt + 1'b1;
                    if (cnt == N_WIRE - 1) begin
                        state <= S_FEED; step <= 0; nout <= 0; v_start <= 1'b1;
                    end
                end
            end
            // the pair is always wbits[1:0]: shift instead of a 268:1 mux
            S_FEED: if (v_push) begin
                wbits <= wbits >> 2;
                if (step == N_STEP - 1) state <= S_WAIT;
                step <= step + 1'b1;
            end
            S_WAIT: if (v_done) begin
                crc <= 32'hffffffff; cbyte <= 0; state <= S_CRC;
            end
            S_CRC: if (cbyte == 14) state <= S_OUT;
                else begin crc <= crc_byte(crc, dbyte(cbyte)); cbyte <= cbyte + 1'b1; end
            S_OUT: begin
                done             <= 1'b1;
                crc_ok           <= (~crc[15:0]) == key;
                fields_valid     <= ((~crc[15:0]) == key) && codes_ok;
                protocol_version <= dbyte(0);
                payload_len_bits <= {dbyte(1), dbyte(2)};
                mod_scheme       <= b3;
                bits_per_symbol  <= mod_ok ? bps_rom[b3[2:0]][3:0] : 4'd0;
                crc_code         <= b4[7:5];
                fec0_code        <= b4[4:0];
                fec1_code        <= b5[4:0];
                dmrs_code        <= b5[6:5];
                c2_len_bytes     <= {dbyte(6), dbyte(7)};
                user_data        <= {dbyte(8), dbyte(9), dbyte(10), dbyte(11), dbyte(12), dbyte(13)};
                state            <= S_COLLECT; cnt <= 0;
            end
            default: state <= S_COLLECT;
            endcase
            // decoded bits: keep the first 128 (the last 6 are the tail)
            if (v_valid && nout < N_DEC) begin
                dec  <= {dec[N_DEC-2:0], v_bit};
                nout <= nout + 1'b1;
            end
        end
    end
endmodule
