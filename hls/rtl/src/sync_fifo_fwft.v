// ============================================================
// sync_fifo_fwft.v -- single-clock FIFO, first-word-fall-through
//
// The head item sits in a REGISTER (head_data) and is presented as soon as
// it exists; it changes only when it is popped. That gives two properties
// the receiver relies on:
//   * a valid/ready output built on it holds every field stable while
//     ready is low (the I2 rule, docs/rx_modular_architecture.md section 7);
//   * the memory's read data always lands in a register before any
//     consumer logic (rundown section 6, "a memory output that feeds a
//     consumer gets its own register stage").
//
// Push and pop may happen in the same cycle, every cycle: a full-rate
// stream passes straight through with two cycles of latency.
//
// DEPTH must be a power of two (pointer wrap). `level` counts the head
// too. Pushing while full is a caller bug: the item is dropped and
// `overflow` latches (sticky) so it can never pass silently.
// ============================================================
`timescale 1ns / 1ps

module sync_fifo_fwft #(
    parameter integer WIDTH = 8,
    parameter integer DEPTH = 16
)(
    input  wire             clk,
    input  wire             rst,

    input  wire             push,
    input  wire [WIDTH-1:0] push_data,

    output reg              head_valid,
    output reg  [WIDTH-1:0] head_data,
    input  wire             pop,          // ignored unless head_valid

    output wire [$clog2(DEPTH):0] level,  // stored + head
    output wire             full,
    output reg              overflow,     // sticky
    output reg              underflow     // sticky: pop with nothing at the head
);
    localparam integer AW = $clog2(DEPTH);

    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [AW:0] wr, rd;

    wire [AW:0] stored    = wr - rd;               // in memory, not yet at head
    wire        mem_empty = (stored == {(AW+1){1'b0}});
    assign      full      = (stored == (AW+1)'(DEPTH));
    assign      level     = stored + {{AW{1'b0}}, head_valid};

    wire do_pop  = pop && head_valid;
    wire do_load = !mem_empty && (!head_valid || do_pop);

    always @(posedge clk) begin
        if (push && !full) mem[wr[AW-1:0]] <= push_data;
        if (do_load)       head_data <= mem[rd[AW-1:0]];
    end

    always @(posedge clk) begin
        if (rst) begin
            wr <= {(AW+1){1'b0}};
            rd <= {(AW+1){1'b0}};
            head_valid <= 1'b0;
            overflow   <= 1'b0;
            underflow  <= 1'b0;
        end else begin
            if (push) begin
                if (full) overflow <= 1'b1;
                else      wr <= wr + 1'b1;
            end
            if (pop && !head_valid) underflow <= 1'b1;
            if (do_load) begin
                rd <= rd + 1'b1;
                head_valid <= 1'b1;
            end else if (do_pop) begin
                head_valid <= 1'b0;
            end
        end
    end
endmodule
