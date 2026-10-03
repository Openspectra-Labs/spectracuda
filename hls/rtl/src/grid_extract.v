// ============================================================
// grid_extract.v -- split an OFDM symbol's bins into data and pilots
//
// Port of spectracuda's ResourceGrid.extract_data/extract_pilots
// (spectracuda/ofdm/resource_grid.py:117-123), which are plain index
// selections: grid[:, data_indices] and grid[:, pilot_indices].
//
// cp_fft streams all N_FFT bins out in natural order, one per valid
// cycle. This block classifies each bin as it arrives and routes it to
// the data stream, the pilot stream, or nowhere. There is no buffering:
// selection is a function of the bin index alone, so the two output
// streams come out in bin order, which is the order every downstream
// block (mmse_eq, demapper) already expects.
//
// THE TYPE MAP IS A ROM, NOT AN EXPRESSION. The pilot pattern is not a
// regular stride -- for fft=256 it is 1,33,65,97,159,191,223,255, with
// the run broken where DC and the guard band sit. hls/gen/emit_rtl.py
// derives grid_type.mem from the live ResourceGrid, so the RTL and the
// golden model cannot disagree about which bin is what.
//
// sof marks the first bin of a symbol. The bin counter is reset from it
// rather than free-running, so a dropped or extra sample resynchronises
// at the next symbol instead of corrupting every symbol after it.
// ============================================================
`timescale 1ns / 1ps
`include "grid_params.vh"

module grid_extract #(
    parameter integer DATA_W = 32   // per component, matching cp_fft's output
)(
    input  wire                     clk,
    input  wire                     rst,

    input  wire signed [DATA_W-1:0] in_re,
    input  wire signed [DATA_W-1:0] in_im,
    input  wire                     in_valid,
    input  wire                     sof,        // first bin of a symbol
    input  wire [1:0]               in_stype,   // symbol tag, see cp_fft

    output reg  signed [DATA_W-1:0] data_re,
    output reg  signed [DATA_W-1:0] data_im,
    output reg                      data_valid,

    output reg  signed [DATA_W-1:0] pilot_re,
    output reg  signed [DATA_W-1:0] pilot_im,
    output reg                      pilot_valid,

    output reg                      sym_done,   // pulses after the last bin

    // Tag of the bin now on data_* / pilot_*. Registered alongside them,
    // so it is valid whenever data_valid or pilot_valid is.
    output reg  [1:0]               out_stype
);
    localparam integer N_FFT  = `GRID_N_FFT;
    localparam integer IDX_W  = $clog2(N_FFT);
    // Sized, so the compare does not silently widen to 32 bits.
    localparam [IDX_W-1:0] LAST_BIN = IDX_W'(N_FFT - 1);

    // 2-bit type per bin: 0 null, 1 data, 2 pilot.
    reg [1:0] sctype [0:N_FFT-1];
    initial $readmemh(`GRID_TYPE_MEM, sctype);

    reg [IDX_W-1:0] bin;
    wire [1:0] t = sctype[bin];

    always @(posedge clk) begin
        if (rst) begin
            bin         <= {IDX_W{1'b0}};
            data_valid  <= 1'b0;
            pilot_valid <= 1'b0;
            sym_done    <= 1'b0;
            out_stype   <= 2'd0;
            data_re <= 0; data_im <= 0;
            pilot_re <= 0; pilot_im <= 0;
        end else begin
            data_valid  <= 1'b0;
            pilot_valid <= 1'b0;
            sym_done    <= 1'b0;

            if (in_valid) begin
                out_stype <= in_stype;
                // sof overrides the counter: resynchronise per symbol.
                if (sof)
                    bin <= 1;
                else
                    bin <= (bin == LAST_BIN) ? {IDX_W{1'b0}} : bin + 1'b1;

                case (sof ? sctype[0] : t)
                    `GRID_TYPE_DATA: begin
                        data_re    <= in_re;
                        data_im    <= in_im;
                        data_valid <= 1'b1;
                    end
                    `GRID_TYPE_PILOT: begin
                        pilot_re    <= in_re;
                        pilot_im    <= in_im;
                        pilot_valid <= 1'b1;
                    end
                    default: ;   // null / guard / DC -- dropped
                endcase

                if (!sof && bin == LAST_BIN)
                    sym_done <= 1'b1;
            end
        end
    end
endmodule
