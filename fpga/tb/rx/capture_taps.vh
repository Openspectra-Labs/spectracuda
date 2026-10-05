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
// SINCE STEP 3c the taps read the stage BUSES in rx_top, not internals:
//   I1 = what rx_freq_domain receives (in_* port connections: the old
//        TD's FFT output plus adapter A's sym_idx / fseq);
//   I2 = every transfer ACCEPTED on rx_freq_domain's out_* port.
// So check_golden.sh compares the new FD's real interfaces against the
// pre-refactor reference. Since step 4b c1 reads the C1 config bus that
// rx_bit_domain publishes.
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

        // ---- I1: the TD -> FD bus ----
        if (dut.fft_valid)
            $fwrite(cap_i1, "%0d %0d %0d %0d %0d %0d\n", dut.i1_fseq, dut.i1_sym,
                    dut.fft_bin, dut.fft_stype, $signed(dut.fft_re), $signed(dut.fft_im));

        // ---- I2: accepted FD -> BIT transfers, 4-bit signed LLRs
        //      (rx_top LLR_W = 4; > 0 means bit 0, header items are +/-7) ----
        if (dut.fd_out_valid && dut.fd_out_ready)
            $fwrite(cap_i2, "%0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d\n",
                    dut.fd_out_fseq, dut.fd_out_sym_idx, dut.fd_out_sc,
                    dut.fd_out_stype, dut.fd_out_n,
                    $signed(dut.fd_out_llr[3:0]),   $signed(dut.fd_out_llr[7:4]),
                    $signed(dut.fd_out_llr[11:8]),  $signed(dut.fd_out_llr[15:12]),
                    $signed(dut.fd_out_llr[19:16]), $signed(dut.fd_out_llr[23:20]));
    end

    // ---- C1: what the header produced, once per run ----
    task cap_write_c1;
        begin
            if (!cap_c1_done) begin
                cap_c1_done = 1'b1;
                $fwrite(cap_c1, "fseq %0d\n", dut.cfg_fseq);
                $fwrite(cap_c1, "cfg_body_syms %0d\n", dut.cfg_body_syms);
                $fwrite(cap_c1, "cfg_mod %0d\n", dut.mod_scheme);
                $fwrite(cap_c1, "cfg_payload_len_bits %0d\n", dut.payload_len_bits);
                $fwrite(cap_c1, "cfg_crc %0d\n", dut.crc_code);
                $fwrite(cap_c1, "cfg_fec0 %0d\n", dut.fec0_code);
                $fwrite(cap_c1, "cfg_fec1 %0d\n", dut.fec1_code);
                $fwrite(cap_c1, "cfg_c2_syms 0\ncfg_dmrs_period %0d\n", dut.cfg_dmrs_period);
                $fclose(cap_i1); $fclose(cap_i2); $fclose(cap_c1);
            end
        end
    endtask
