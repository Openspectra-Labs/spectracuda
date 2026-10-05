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
// the run broken where DC and the guard band sit. fpga/gen/emit_rtl.py
// derives grid_type.mem from the live ResourceGrid, so the RTL and the
// golden model cannot disagree about which bin is what.
//
// The bin index comes IN with each sample (in_bin) -- it is metadata the
// time domain already knows -- so this block is stateless: type and
// ordinal are two ROM lookups on in_bin. It used to count bins itself
// and resynchronise on a `sof` pulse, i.e. it rebuilt position that its
// producer already had (fpga/docs/rx_modular_architecture.md section 6).
//
// out_sc is the bin's ORDINAL among data bins (0..N_DATA-1) or among
// pilot bins (0..N_PILOT-1), whichever stream it is emitted on, from the
// generated grid_ord.mem. Downstream addresses H by it instead of
// counting items.
// ============================================================
`timescale 1ns / 1ps
`include "grid_params.vh"

module grid_extract #(
    parameter integer DATA_W = 32,  // per component, matching cp_fft's output
    // Opaque sideband registered alongside the data; >= 1 bit.
    parameter integer META_W = 1
)(
    input  wire                     clk,
    input  wire                     rst,

    input  wire signed [DATA_W-1:0] in_re,
    input  wire signed [DATA_W-1:0] in_im,
    input  wire                     in_valid,
    input  wire [7:0]               in_bin,     // natural-order bin of this sample
    input  wire [META_W-1:0]        in_meta,

    output reg  signed [DATA_W-1:0] data_re,
    output reg  signed [DATA_W-1:0] data_im,
    output reg                      data_valid,

    output reg  signed [DATA_W-1:0] pilot_re,
    output reg  signed [DATA_W-1:0] pilot_im,
    output reg                      pilot_valid,

    output reg  [7:0]               out_sc,     // ordinal, see header
    output reg  [META_W-1:0]        out_meta,   // in_meta of this item
    output reg                      sym_done    // with the symbol's last bin
);
    localparam integer N_FFT = `GRID_N_FFT;
    localparam [7:0]   LAST_BIN = 8'(N_FFT - 1);

    // 2-bit type per bin: 0 null, 1 data, 2 pilot.
    //
    // DISTRIBUTED ROM, on purpose. Left to itself Vivado put scord + the
    // out_sc register into a block RAM; BRAM clock-to-out then drove the
    // H-store read and the equalizer's DSP inputs in one cycle and failed
    // timing (-0.452 ns, 153 endpoints, step 3c). These are 256 x 2 and
    // 256 x 8 -- a few LUTs -- and out_sc must leave a plain flip-flop,
    // like the old d_rd counter did.
    (* rom_style = "distributed" *) reg [1:0] sctype [0:N_FFT-1];
    (* rom_style = "distributed" *) reg [7:0] scord  [0:N_FFT-1];
    initial $readmemh(`GRID_TYPE_MEM, sctype);
    initial $readmemh(`GRID_ORD_MEM,  scord);

    wire [1:0] t = sctype[in_bin];

    always @(posedge clk) begin
        if (rst) begin
            data_valid  <= 1'b0;
            pilot_valid <= 1'b0;
            sym_done    <= 1'b0;
            out_sc      <= 8'd0;
            out_meta    <= {META_W{1'b0}};
            data_re <= 0; data_im <= 0;
            pilot_re <= 0; pilot_im <= 0;
        end else begin
            data_valid  <= 1'b0;
            pilot_valid <= 1'b0;
            sym_done    <= 1'b0;

            if (in_valid) begin
                out_sc   <= scord[in_bin];
                out_meta <= in_meta;
                case (t)
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
                sym_done <= (in_bin == LAST_BIN);
            end
        end
    end
endmodule
