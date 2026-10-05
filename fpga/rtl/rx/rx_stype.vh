// rx_stype.vh -- symbol-type sideband codes, carried with the bins from
// rx_time_domain's phase FSM through cp_fft and grid_extract.
// Included INSIDE a module body (localparams, not macros).
// Code 3 is free -- reserved for DMRS.
localparam [1:0] ST_TRAIN = 2'd0, ST_HDR = 2'd1, ST_PAY = 2'd2;
