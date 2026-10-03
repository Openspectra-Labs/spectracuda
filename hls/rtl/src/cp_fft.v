// ============================================================
// cp_fft.v -- cyclic-prefix strip + forward FFT
//
// Wraps the Xilinx FFT LogiCORE (xfft v9.1). The transform itself is a
// vendor core and is not re-verified here; what this block adds, and what
// the testbench actually checks, is:
//   * stripping the CP at the right offset,
//   * driving the AXI-Stream framing (tlast on the last sample),
//   * the output scaling convention.
// Those are the parts that can be wrong.
//
// CONFIGURATION FOLLOWS openofdm (reference/openofdm/verilog/Xilinx/zynq/
// xfft/xfft_v9.xci), which is proven in openwifi, with transform_length
// raised from their 64 (802.11) to spectracuda's 256:
//
//     unscaled, natural_order, 16-bit input, truncation rounding
//
// UNSCALED matters twice over. It cannot overflow -- the core widens the
// datapath each stage instead of discarding bits, so unlike a scaled core
// there is no schedule that can be got wrong. (~/work/ofdm-hls uses
// scaled with scale_sch=0xAA and argues at length about why 0x55 would
// overflow; unscaled makes that whole question disappear.) And numpy's
// forward FFT is ALSO unscaled -- spectracuda/ofdm/fft.py uses numpy's
// standard convention -- so the output correlates with the golden model
// directly, with no scale factor to reconcile.
//
// The cost is width: output is 16 + log2(256) + 1 = 25 bits per
// component, which the core pads to 32. openofdm narrows back to 16 bits
// downstream by selecting a window (sync_long.v takes fft_out_re[22:7]);
// that choice belongs to the equalizer, not here, so this block passes
// the full width on.
//
// The core's s_axis_data_tready is monitored, not assumed. A pipelined
// streaming FFT holds tready once running, but "generally holds" is not a
// guarantee, and a dropped sample would show up downstream as a subtly
// wrong symbol rather than as an obvious break. `overflow` latches if a
// sample is ever presented while the core is not ready.
// ============================================================
`timescale 1ns / 1ps

module cp_fft #(
    parameter integer SAMPLE_W = 16,
    parameter integer FFT_SIZE = 256,
    parameter integer CP_LEN   = 32,
    parameter integer OUT_W    = 32   // per component, as the core pads it
)(
    input  wire                       clk,
    input  wire                       rst,

    // Sample stream. `sof` marks the first sample of a slot -- the first
    // CP sample, not the first FFT sample.
    input  wire signed [SAMPLE_W-1:0] in_i,
    input  wire signed [SAMPLE_W-1:0] in_q,
    input  wire                       in_valid,
    input  wire                       sof,
    // Symbol-type tag (train / header / payload / ...), sampled at sof
    // and handed back on out_stype with the same symbol's bins. Opaque
    // to this block -- it only carries it across the core.
    input  wire [1:0]                 in_stype,

    // Frequency-domain output, natural order.
    output wire signed [OUT_W-1:0]    out_re,
    output wire signed [OUT_W-1:0]    out_im,
    output wire                       out_valid,
    output wire                       out_last,
    output wire                       out_sof,     // first bin of a symbol
    output wire [1:0]                 out_stype,   // that symbol's tag

    output reg                        overflow
);

    localparam integer SLOT = FFT_SIZE + CP_LEN;
    localparam integer CNT_W = $clog2(SLOT + 1);

    // ---- slot counter ----
    // Free-running once started, so a single sof at the head of a burst
    // aligns every following symbol.
    reg [CNT_W-1:0] cnt;
    reg             started;

    /* verilator lint_off WIDTHTRUNC */
    localparam [CNT_W-1:0] SLOT_M1 = SLOT - 1;
    localparam [CNT_W-1:0] CP_END  = CP_LEN;
    /* verilator lint_on WIDTHTRUNC */

    // `idx` is the index of the sample being presented RIGHT NOW, not a
    // lagging count. The first version reset `cnt` on the same edge that
    // consumed sample 0, so sample i was judged against cnt = i-1: the
    // last sample of the slot never matched SLOT_M1, tlast never fired,
    // only 255 of 256 samples reached the core, and the transform never
    // completed. Symptom was 288 samples in, ZERO bins out, with no
    // overflow flagged -- a silent stall rather than a visible error.
    wire [CNT_W-1:0] idx = (sof && in_valid) ? {CNT_W{1'b0}} : cnt;
    wire             active = in_valid && (started || sof);

    always @(posedge clk) begin
        if (rst) begin
            cnt     <= {CNT_W{1'b0}};
            started <= 1'b0;
        end else if (active) begin
            cnt     <= (idx == SLOT_M1) ? {CNT_W{1'b0}} : idx + 1'b1;
            started <= 1'b1;
        end
    end

    // Samples 0..CP_LEN-1 are the cyclic prefix and are dropped; the
    // remaining FFT_SIZE go to the transform, tlast on the last.
    wire in_body   = active && (idx >= CP_END);
    wire in_islast = in_body && (idx == SLOT_M1);

    // ---- FFT core ----
    wire [31:0] s_tdata = {in_q, in_i};   // {imag, real}, core convention
    wire        s_tready;
    wire [63:0] m_tdata;
    wire        m_tvalid, m_tlast;

    // Config: unscaled, so only the forward/inverse bit carries meaning --
    // the scaling-schedule field does not exist in this configuration.
    // openofdm drives the same thing ({7'b0, 1'b1}, sync_long.v:227).
    reg  config_sent;
    wire config_tvalid = !config_sent;
    wire config_tready;

    always @(posedge clk) begin
        if (rst) config_sent <= 1'b0;
        else if (config_tvalid && config_tready) config_sent <= 1'b1;
    end

    always @(posedge clk) begin
        if (rst) overflow <= 1'b0;
        else if (in_body && !s_tready) overflow <= 1'b1;
    end

    xfft_256 fft_inst (
        .aclk                        (clk),
        .s_axis_config_tdata         (8'h01),      // forward transform
        // 8 bits, not 16: with scaling_options=unscaled the core has no
        // scaling-schedule field, so only the FWD/INV bit exists. A scaled
        // core would widen this port to carry the schedule.
        .s_axis_config_tvalid        (config_tvalid),
        .s_axis_config_tready        (config_tready),
        .s_axis_data_tdata           (s_tdata),
        .s_axis_data_tvalid          (in_body),
        .s_axis_data_tready          (s_tready),
        .s_axis_data_tlast           (in_islast),
        .m_axis_data_tdata           (m_tdata),
        .m_axis_data_tvalid          (m_tvalid),
        .m_axis_data_tready          (1'b1),
        .m_axis_data_tlast           (m_tlast),
        .event_frame_started         (),
        .event_tlast_unexpected      (),
        .event_tlast_missing         (),
        .event_status_channel_halt   (),
        .event_data_in_channel_halt  (),
        .event_data_out_channel_halt ()
    );

    // ---- symbol tag, carried across the core ----
    // A FIFO of one tag per symbol, NOT a delay line. The core's latency
    // is not something this design knows: the simulation stub in
    // tb/stubs/ and the real xfft_256 differ, and a delay line matched to
    // either would silently mis-tag on the other. Pushing at the input
    // sof and popping at the first output bin is correct for any latency,
    // as long as the core keeps symbols in order -- which a streaming FFT
    // does. Depth 4 covers the symbols a pipelined-streaming core can
    // hold in flight (about 2) with margin.
    //
    // The push is at `sof`, i.e. the first CP sample. A slot that starts
    // but never reaches tlast would leave a stale tag; the frame-level
    // reset (rst) clears it.
    reg [1:0] tag_mem [0:3];
    reg [1:0] tag_wr, tag_rd;
    reg       out_first;          // next output beat is a symbol's bin 0
    reg [1:0] cur_stype;          // tag of the symbol now streaming out

    always @(posedge clk) begin
        if (rst) begin
            tag_wr    <= 2'd0;
            tag_rd    <= 2'd0;
            out_first <= 1'b1;
            cur_stype <= 2'd0;
        end else begin
            if (sof && in_valid) begin
                tag_mem[tag_wr] <= in_stype;
                tag_wr          <= tag_wr + 1'b1;
            end
            if (m_tvalid) begin
                if (out_first) begin
                    tag_rd    <= tag_rd + 1'b1;
                    cur_stype <= tag_mem[tag_rd];
                end
                out_first <= m_tlast;
            end
        end
    end

    assign out_sof   = m_tvalid && out_first;
    // Valid on EVERY bin of the symbol, not just bin 0, so no consumer
    // has to latch it.
    assign out_stype = out_first ? tag_mem[tag_rd] : cur_stype;

    assign out_re    = m_tdata[OUT_W-1:0];
    assign out_im    = m_tdata[32 + OUT_W-1:32];
    assign out_valid = m_tvalid;
    assign out_last  = m_tlast;
endmodule
