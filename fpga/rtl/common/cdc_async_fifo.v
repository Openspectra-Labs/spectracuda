// ============================================================
// cdc_async_fifo.v -- dual-clock FIFO for a valid/ready stream
//
// Clock-crossing infrastructure between two IP blocks on different clocks
// (the role of an AXI4-Stream clock converter in a Vivado block design):
// used between rx_freq_domain (clk) and rx_bit_domain (clk_bd). Neither
// block knows the other's clock.
//
// Classic Gray-code design: each side keeps a binary pointer for
// addressing and a Gray copy that is the ONLY thing synchronized into the
// other domain (two flops). A Gray pointer changes one bit per step, so a
// synchronizer catching it mid-change sees either the old or the new
// value, never a mix. full/empty are therefore conservative, never wrong.
//
// Constraints (impl TCL / XDC): the clocks are asynchronous groups, and the
// Gray pointer paths get set_max_delay -datapath_only of one source period.
// The memory read is asynchronous (LUT-RAM), so r_data is valid with
// r_valid and holds while r_ready is low (the stream rule).
//
// DEPTH must be a power of two (Gray wrap). wrst / rrst are each already
// synchronous to their own clock (cdc_reset_sync).
// ============================================================
`timescale 1ns / 1ps

module cdc_async_fifo #(
    parameter integer WIDTH = 8,
    parameter integer DEPTH = 16
)(
    input  wire             wclk,
    input  wire             wrst,
    input  wire             w_valid,
    output wire             w_ready,
    input  wire [WIDTH-1:0] w_data,

    input  wire             rclk,
    input  wire             rrst,
    output wire             r_valid,
    input  wire             r_ready,
    output wire [WIDTH-1:0] r_data
);
    localparam integer AW = $clog2(DEPTH);

    (* ram_style = "distributed" *) reg [WIDTH-1:0] mem [0:DEPTH-1];

    function [AW:0] bin2gray(input [AW:0] b);
        bin2gray = b ^ (b >> 1);
    endfunction

    // ---- write side ----
    reg  [AW:0] wbin, wgray;
    (* ASYNC_REG = "TRUE" *) reg [AW:0] rgray_w1, rgray_w2;   // read ptr, synced
    wire [AW:0] wbin_n  = wbin + 1'b1;
    // full: next write would land on the read pointer one lap behind
    // (Gray form: top two bits inverted, rest equal)
    wire        wfull   = (wgray == {~rgray_w2[AW:AW-1], rgray_w2[AW-2:0]});
    assign      w_ready = !wfull;
    wire        wpush   = w_valid && !wfull;

    always @(posedge wclk) begin
        if (wpush) mem[wbin[AW-1:0]] <= w_data;
    end
    always @(posedge wclk) begin
        if (wrst) begin
            wbin <= 0; wgray <= 0; rgray_w1 <= 0; rgray_w2 <= 0;
        end else begin
            rgray_w1 <= rgray;  rgray_w2 <= rgray_w1;
            if (wpush) begin
                wbin  <= wbin_n;
                wgray <= bin2gray(wbin_n);
            end
        end
    end

    // ---- read side ----
    reg  [AW:0] rbin, rgray;
    (* ASYNC_REG = "TRUE" *) reg [AW:0] wgray_r1, wgray_r2;   // write ptr, synced
    wire [AW:0] rbin_n  = rbin + 1'b1;
    wire        rempty  = (rgray == wgray_r2);
    assign      r_valid = !rempty;
    assign      r_data  = mem[rbin[AW-1:0]];
    wire        rpop    = r_valid && r_ready;

    always @(posedge rclk) begin
        if (rrst) begin
            rbin <= 0; rgray <= 0; wgray_r1 <= 0; wgray_r2 <= 0;
        end else begin
            wgray_r1 <= wgray;  wgray_r2 <= wgray_r1;
            if (rpop) begin
                rbin  <= rbin_n;
                rgray <= bin2gray(rbin_n);
            end
        end
    end
endmodule
