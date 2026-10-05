// ============================================================
// rx_if.vh -- FROZEN stage-interface definitions (2026-10-03)
//
// Single source for the symbol-type enum and the widths of the three
// stage interfaces defined in fpga/docs/rx_modular_architecture.md section 7:
//
//   I1  rx_time_domain -> rx_freq_domain   FFT bins, no backpressure
//   I2  rx_freq_domain -> rx_bit_domain    LLR groups, valid/ready
//   C1  rx_bit_domain  -> TD / FD / host   per-frame config bundle
//
// Included INSIDE a module body (localparams, not macros). Changing any
// value here is an interface change: update the design doc first.
//
// Not yet used by the RTL -- the refactor (doc section 11, steps 3-5)
// moves each stage onto it. The current 2-bit tag in rx_stype.vh is the
// pre-freeze version and goes away with that refactor.
// ============================================================

// ---- symbol type, 3 bits, one enum for the whole receiver ----------
//   TD assigns TRAIN / HEADER / BODY.  FD refines BODY into DATA / DMRS /
//   C2.  I2 only ever carries HEADER, DATA, C2.  DMRS never leaves FD.
localparam integer STYPE_W   = 3;
localparam [2:0]   ST_TRAIN  = 3'd0;
localparam [2:0]   ST_HEADER = 3'd1;
localparam [2:0]   ST_BODY   = 3'd2;   // TD only: not yet classified
localparam [2:0]   ST_DATA   = 3'd3;
localparam [2:0]   ST_DMRS   = 3'd4;
localparam [2:0]   ST_C2     = 3'd5;   // future

// ---- common metadata ------------------------------------------------
// fseq wrap rule: a value may be reused only after the previous frame
// carrying it has retired from EVERY stage and buffer (asserted in sim).
localparam integer FSEQ_W    = 2;
localparam integer SYMIDX_W  = 8;      // OFDM symbol index within frame, 0 = first TRAIN

// ---- I1: FFT bin stream ---------------------------------------------
localparam integer BIN_W     = 8;      // natural-order bin 0..255; start = 0, end = 255

// ---- I2: LLR group stream -------------------------------------------
// llr[0] is the FIRST bit in transmission order; llr[i >= n] driven 0.
// Hard decision == sign bit (MSB) of the LLR at every width.
localparam integer LLR_MAX   = 6;      // 64-QAM
localparam integer LLR_N_W   = 3;      // ll_n: 1 BPSK, 2 QPSK, 4 16-QAM, 6 64-QAM
localparam integer SC_W      = 8;      // data-subcarrier ordinal 0..215

// ---- C1: config bundle ----------------------------------------------
// cfg_body_syms = OFDM symbols AFTER the last header symbol
//               = C2 + main data + DMRS symbols.
localparam integer BODYSYM_W = 8;
localparam integer MOD_W     = 3;
localparam integer DMRSP_W   = 2;      // header dmrs_period code, 0 = off
