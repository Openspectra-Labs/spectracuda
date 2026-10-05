// ============================================================
// cdc_bundle.v -- move a multi-bit, slowly changing register bundle
//                 across clock domains, atomically
//
// Used for the C1 config bundle (rx_bit_domain on clk_bd -> rx_freq_domain
// and rx_time_domain on clk). The bundle changes at most once per frame, so
// a request/acknowledge handshake is cheap and exact:
//
//   source: when src_data differs from what was last SENT and no transfer
//           is in flight, latch it into `hold` and toggle req.
//   dest:   synchronize req (two flops); on a toggle, copy `hold` (stable
//           since the source latched it) into dst_data, toggle ack.
//   source: synchronize ack; when it matches req, the next change may go.
//
// `hold` is only ever sampled by the destination while it is stable, so
// the multi-bit data path needs no synchronizer -- only req and ack do.
// A change made while a transfer is in flight is sent next, so the
// destination always converges to the latest value, never a torn mix.
//
// Constraints: hold -> dst_data is a quasi-static path (set_max_delay
// -datapath_only); req/ack are the synchronized signals.
// ============================================================
`timescale 1ns / 1ps

module cdc_bundle #(
    parameter integer WIDTH = 8
)(
    input  wire             src_clk,
    input  wire             src_rst,
    input  wire [WIDTH-1:0] src_data,

    input  wire             dst_clk,
    input  wire             dst_rst,
    output reg  [WIDTH-1:0] dst_data
);
    reg              ack;               // dest -> source
    (* ASYNC_REG = "TRUE" *) reg req_s1, req_s2;

    // ---- source side ----
    reg  [WIDTH-1:0] hold;
    reg              req;
    (* ASYNC_REG = "TRUE" *) reg ack_s1, ack_s2;
    wire             idle = (ack_s2 == req);

    always @(posedge src_clk) begin
        if (src_rst) begin
            hold <= {WIDTH{1'b0}}; req <= 1'b0; ack_s1 <= 1'b0; ack_s2 <= 1'b0;
        end else begin
            ack_s1 <= ack; ack_s2 <= ack_s1;
            if (idle && src_data != hold) begin
                hold <= src_data;
                req  <= ~req;
            end
        end
    end

    // ---- destination side ----

    always @(posedge dst_clk) begin
        if (dst_rst) begin
            dst_data <= {WIDTH{1'b0}}; ack <= 1'b0; req_s1 <= 1'b0; req_s2 <= 1'b0;
        end else begin
            req_s1 <= req; req_s2 <= req_s1;
            if (req_s2 != ack) begin
                dst_data <= hold;
                ack      <= req_s2;
            end
        end
    end
endmodule
