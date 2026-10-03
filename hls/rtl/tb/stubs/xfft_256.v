// ============================================================
// xfft_256.v -- BEHAVIOURAL STUB of the Xilinx FFT LogiCORE
//
// SIMULATION ONLY. Never add this to a synthesis file list.
//
// Exists because Verilator cannot elaborate the real xfft_256 (it is a
// Xilinx primitive netlist needing unisims), which would otherwise make
// rx_top.v unsimulatable on this machine and leave the whole top level
// untested until someone opened Vivado.
//
// It computes a plain unnormalised DFT in `real`, matching the core's
// scaling_options=unscaled setting (no 1/N, no per-stage scaling), and
// streams 256 bins out in natural order with tlast on the last.
//
// WHAT THIS DOES NOT PROVE: the real core's latency, its AXI
// handshaking under backpressure, its rounding, or its output ordering
// option. Those need xsim against the actual IP. A pass here means the
// surrounding chain is wired correctly, not that the FFT integration is.
// ============================================================
`timescale 1ns / 1ps

module xfft_256 (
    input  wire        aclk,
    input  wire [7:0]  s_axis_config_tdata,
    input  wire        s_axis_config_tvalid,
    output wire        s_axis_config_tready,
    input  wire [31:0] s_axis_data_tdata,
    input  wire        s_axis_data_tvalid,
    output wire        s_axis_data_tready,
    input  wire        s_axis_data_tlast,
    output reg  [63:0] m_axis_data_tdata,
    output reg         m_axis_data_tvalid,
    input  wire        m_axis_data_tready,
    output reg         m_axis_data_tlast,
    output wire        event_frame_started,
    output wire        event_tlast_unexpected,
    output wire        event_tlast_missing,
    output wire        event_status_channel_halt,
    output wire        event_data_in_channel_halt,
    output wire        event_data_out_channel_halt
);
    localparam integer N = 256;
    localparam real PI = 3.14159265358979323846;

    assign s_axis_config_tready = 1'b1;
    assign s_axis_data_tready   = 1'b1;
    assign event_frame_started        = 1'b0;
    assign event_tlast_unexpected     = 1'b0;
    assign event_tlast_missing        = 1'b0;
    assign event_status_channel_halt  = 1'b0;
    assign event_data_in_channel_halt = 1'b0;
    assign event_data_out_channel_halt= 1'b0;

    // DOUBLE BUFFERED. A single buffer made the stub drop symbols: it
    // cannot accept input while streaming a result, so 4 of 6 symbols
    // vanished and the receiver starved. The real core pipelines; this
    // at least accepts continuously, which is what the chain assumes.
    // Twiddles precomputed once. The naive form called $cos/$sin inside
    // the O(N^2) inner loop -- 131072 trig calls per symbol, tolerable
    // for a 4-symbol frame and hopeless for the 78-symbol frames a
    // 2048-byte QPSK packet produces.
    real tw_c [0:N-1];
    real tw_s [0:N-1];

    real xr [0:N-1];
    real xi [0:N-1];
    real cr [0:N-1];
    real ci [0:N-1];
    integer wr;
    integer k, n, outp;
    real sr, si, ang;
    reg busy;

    initial begin
        for (k = 0; k < N; k = k + 1) begin
            tw_c[k] =  $cos(-2.0 * PI * $itor(k) / $itor(N));
            tw_s[k] =  $sin(-2.0 * PI * $itor(k) / $itor(N));
        end
        wr = 0; outp = -1; busy = 0;
        m_axis_data_tvalid = 1'b0;
        m_axis_data_tlast  = 1'b0;
        m_axis_data_tdata  = 64'd0;
    end

    always @(posedge aclk) begin
        m_axis_data_tvalid <= 1'b0;
        m_axis_data_tlast  <= 1'b0;

        // Input side: always accepting.
        if (s_axis_data_tvalid) begin
            xr[wr] = $itor($signed(s_axis_data_tdata[15:0]));
            xi[wr] = $itor($signed(s_axis_data_tdata[31:16]));
            wr = wr + 1;
            if (wr == N) begin
                wr = 0;
                for (k = 0; k < N; k = k + 1) begin
                    cr[k] = xr[k]; ci[k] = xi[k];
                end
                busy = 1; outp = 0;
            end
        end

        // Output side: one bin per clock off the snapshot.
        if (busy) begin
            sr = 0.0; si = 0.0;
            for (n = 0; n < N; n = n + 1) begin
                k = (outp * n) % N;
                sr = sr + cr[n]*tw_c[k] - ci[n]*tw_s[k];
                si = si + cr[n]*tw_s[k] + ci[n]*tw_c[k];
            end
            m_axis_data_tdata  <= {$rtoi(si), $rtoi(sr)};
            m_axis_data_tvalid <= 1'b1;
            m_axis_data_tlast  <= (outp == N-1);
            if (outp == N-1) begin busy = 0; outp = -1; end
            else outp = outp + 1;
        end
    end
endmodule
