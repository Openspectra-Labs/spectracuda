// ============================================================
// capture_taps.vh -- reference dumps at the frozen stage boundaries
//
// Included INSIDE rx_top_tb when RXT_CAPTURE_DIR is defined (run_frame.py
// --capture DIR). Writes, in the formats frozen in
// docs/rx_modular_architecture.md section 7:
//
//   i1.txt  TD -> FD   one FFT bin per line:
//           fseq sym_idx bin stype re im
//   i2.txt  FD -> BD   one LLR group per line (LLR_W = 1):
//           fseq sym_idx sc stype n llr0 llr1 llr2 llr3 llr4 llr5
//           llr is 0 (bit 0) or -1 (bit 1); llr0 = first bit sent;
//           llr[i >= n] = 0
//   c1.txt  config the header produced (one "key value" per line)
//
// THESE TAPS READ PRE-REFACTOR INTERNALS by hierarchical path (u_td,
// u_fd, u_hd, u_bd). They are the bridge from the old RTL to the new
// boundaries and only compile against that hierarchy -- the dumps are
// what the refactored stages are checked against, so they are captured
// once (capture_golden.sh) and committed. Do not expect this file to
// build after the refactor.
//
// sym_idx / fseq / bin / sc are DERIVED here from the old RTL's markers;
// the old RTL does not carry them. sym_idx 0 = the first training symbol.
// ============================================================

    // stype codes: rx_if.vh (frozen). The old 2-bit tag uses the same
    // values for TRAIN/HEADER/BODY, so I1's stype is the tag itself.
    localparam [2:0] CAP_ST_HEADER = 3'd1, CAP_ST_DATA = 3'd3;

    integer cap_i1, cap_i2, cap_c1;
    integer cap_frames = 0;
    integer cap_bin = 0, cap_sym = -1;
    integer cap_hsc = 0, cap_psc = 0, cap_psym = 0;
    reg     cap_c1_done = 1'b0;

    initial begin
        cap_i1 = $fopen({`RXT_CAPTURE_DIR, "/i1.txt"}, "w");
        cap_i2 = $fopen({`RXT_CAPTURE_DIR, "/i2.txt"}, "w");
        cap_c1 = $fopen({`RXT_CAPTURE_DIR, "/c1.txt"}, "w");
    end

    wire [1:0] cap_fseq = 2'((cap_frames - 1) & 3);

    always @(posedge clk) if (!rst) begin
        // ---- frame boundary (TD's frame_start) ----
        if (dut.u_td.fsq_start) begin
            cap_frames = cap_frames + 1;
            cap_sym  = -1;
            cap_hsc  = 0;
            cap_psc  = 0;
            cap_psym = 0;
        end

        // ---- I1: FFT output ----
        if (dut.fft_valid) begin
            if (dut.fft_sof) begin
                cap_bin = 0;
                cap_sym = cap_sym + 1;
            end
            $fwrite(cap_i1, "%0d %0d %0d %0d %0d %0d\n", cap_fseq, cap_sym,
                    cap_bin, dut.fft_stype, $signed(dut.fft_re), $signed(dut.fft_im));
            cap_bin = cap_bin + 1;
        end

        // ---- I2, header: one BPSK bit per data subcarrier ----
        if (dut.u_fd.hdr_valid_bit) begin
            $fwrite(cap_i2, "%0d %0d %0d %0d 1 %0d 0 0 0 0 0\n", cap_fseq,
                    `N_TRAINING, cap_hsc, CAP_ST_HEADER,
                    dut.u_fd.hdr_bit ? -1 : 0);
            cap_hsc = cap_hsc + 1;
        end

        // ---- I2, payload: demapper output, bits[5] is the first bit ----
        if (dut.u_bd.dm_valid) begin
            $fwrite(cap_i2, "%0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d\n",
                    cap_fseq, `N_TRAINING + 1 + cap_psym, cap_psc, CAP_ST_DATA,
                    dut.u_bd.dm_nbits,
                    (dut.u_bd.dm_nbits > 0 && dut.u_bd.dm_bits[5]) ? -1 : 0,
                    (dut.u_bd.dm_nbits > 1 && dut.u_bd.dm_bits[4]) ? -1 : 0,
                    (dut.u_bd.dm_nbits > 2 && dut.u_bd.dm_bits[3]) ? -1 : 0,
                    (dut.u_bd.dm_nbits > 3 && dut.u_bd.dm_bits[2]) ? -1 : 0,
                    (dut.u_bd.dm_nbits > 4 && dut.u_bd.dm_bits[1]) ? -1 : 0,
                    (dut.u_bd.dm_nbits > 5 && dut.u_bd.dm_bits[0]) ? -1 : 0);
            if (cap_psc == `N_DATA - 1) begin
                cap_psc  = 0;
                cap_psym = cap_psym + 1;
            end else
                cap_psc = cap_psc + 1;
        end
    end

    // ---- C1: what the header produced, once per run ----
    task cap_write_c1;
        begin
            if (!cap_c1_done) begin
                cap_c1_done = 1'b1;
                $fwrite(cap_c1, "fseq %0d\n", cap_fseq);
                $fwrite(cap_c1, "cfg_body_syms %0d\n", dut.u_hd.n_pay_sym);
                $fwrite(cap_c1, "cfg_mod %0d\n", dut.mod_scheme);
                $fwrite(cap_c1, "cfg_payload_len_bits %0d\n", dut.payload_len_bits);
                $fwrite(cap_c1, "cfg_crc %0d\n", dut.crc_code);
                $fwrite(cap_c1, "cfg_fec0 %0d\n", dut.fec0_code);
                $fwrite(cap_c1, "cfg_fec1 %0d\n", dut.fec1_code);
                $fwrite(cap_c1, "cfg_c2_syms 0\ncfg_dmrs_period 0\n");
                $fclose(cap_i1); $fclose(cap_i2); $fclose(cap_c1);
            end
        end
    endtask
