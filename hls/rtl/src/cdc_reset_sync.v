// ============================================================
// cdc_reset_sync.v -- reset for a second clock domain
//
// Asserts immediately (asynchronously) and releases synchronously to
// `clk`, after STAGES flops, so every flop in the domain leaves reset on
// the same edge. Used to give rx_bit_domain (clk_bd) its own reset from
// rx_top's rst.
// ============================================================
`timescale 1ns / 1ps

module cdc_reset_sync #(
    parameter integer STAGES = 3
)(
    input  wire clk,
    input  wire rst_in,
    output wire rst_out
);
    (* ASYNC_REG = "TRUE" *) reg [STAGES-1:0] sync;
    always @(posedge clk or posedge rst_in) begin
        if (rst_in) sync <= {STAGES{1'b1}};
        else        sync <= {sync[STAGES-2:0], 1'b0};
    end
    assign rst_out = sync[STAGES-1];
endmodule
