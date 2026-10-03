# Fast C-sim without launching vitis_hls. `vitis_hls -f tcl/synth_sc_sync.tcl`
# runs the same sources for synthesis; this is purely to iterate quickly on
# correctness, which is where nearly all the time goes.
XILINX_HLS_INC ?= /home/abhi/work/xilinx/2025.2/Vitis/include
CXXFLAGS       ?= -O2 -std=c++14 -I$(XILINX_HLS_INC) -Wall -Wno-unknown-pragmas

BUILD := build

.PHONY: headers csim csim-fixed golden clean

headers:
	cd .. && .venv/bin/python -m hls.gen.emit

golden:
	cd .. && .venv/bin/python -m hls.gen.golden

# Stage 1: double-precision, proves the ALGORITHM matches Python.
csim: $(BUILD)/sc_sync_tb
	cd golden && ../$(BUILD)/sc_sync_tb

# Stage 2: ap_fixed types, measured against the stage-1 build as reference.
csim-fixed: $(BUILD)/sc_sync_tb_fixed
	cd golden && ../$(BUILD)/sc_sync_tb_fixed

$(BUILD)/sc_sync_tb: src/sc_sync.cpp tb/sc_sync_tb.cpp src/sc_sync.h
	@mkdir -p $(BUILD)
	g++ $(CXXFLAGS) -o $@ src/sc_sync.cpp tb/sc_sync_tb.cpp

$(BUILD)/sc_sync_tb_fixed: src/sc_sync.cpp tb/sc_sync_tb.cpp src/sc_sync.h
	@mkdir -p $(BUILD)
	g++ $(CXXFLAGS) -DSC_SYNC_USE_FIXED -DSC_METRIC_TOL=1e-3 -o $@ src/sc_sync.cpp tb/sc_sync_tb.cpp

clean:
	rm -rf $(BUILD)

# Word length sweep across the whole golden case grid (4 SNRs x 3 seeds,
# straddling spectracuda's own 6-9 dB detection cliff). One frame at one
# comfortable SNR cannot answer this: quantization noise only competes
# with channel noise near the cliff, so a high-SNR-only sweep reports
# that 4 bits is fine regardless of whether it is.
#
# The pass bar is agreement with PYTHON, not with the true frame start.
# Python itself lands a few samples early at 6 dB on some seeds; the HLS
# block is required to reproduce that, because matching the golden model
# is the contract.
.PHONY: sweep
sweep:
	@mkdir -p $(BUILD)
	@echo "width  cases_matching_python / total   (worst |offset| vs python)"
	@for w in 4 6 8 10 12 14 16 18; do \
	  g++ $(CXXFLAGS) -DSC_SYNC_USE_FIXED -DSC_SAMPLE_W=$$w -DSC_METRIC_TOL=1e-3 \
	      -o $(BUILD)/tb_w$$w src/sc_sync.cpp tb/sc_sync_tb.cpp 2>/dev/null; \
	  ok=0; tot=0; worst=0; \
	  for d in golden/snr*; do \
	    tot=$$((tot+1)); \
	    line=$$(HLS_GOLDEN_DIR=$$d $(BUILD)/tb_w$$w 2>/dev/null | grep "start_index"); \
	    h=$$(echo "$$line" | sed -n 's/.*hls=\([0-9-]*\).*/\1/p'); \
	    p=$$(echo "$$line" | sed -n 's/.*python=\([0-9-]*\).*/\1/p'); \
	    if [ "$$h" = "$$p" ]; then ok=$$((ok+1)); else \
	      dd=$$((h-p)); [ $$dd -lt 0 ] && dd=$$((-dd)); \
	      [ $$dd -gt $$worst ] && worst=$$dd; fi; \
	  done; \
	  printf "  W=%-3s %2s / %-2s                        %s\n" $$w $$ok $$tot $$worst; \
	done

# How many bits does the pre-squaring narrow type actually need? Same
# discipline as `sweep`: measured across the full case grid, judged by
# agreement with Python, not by eyeballing one frame.
.PHONY: narrow-sweep
narrow-sweep:
	@mkdir -p $(BUILD)
	@echo "narrow width  cases_matching_python / total"
	@for w in 16 18 20 22 24 26 28; do \
	  g++ $(CXXFLAGS) -DSC_SYNC_USE_FIXED -DSC_NARROW_W=$$w -DSC_METRIC_TOL=1e-3 \
	      -o $(BUILD)/tb_n$$w src/sc_sync.cpp tb/sc_sync_tb.cpp 2>/dev/null; \
	  ok=0; tot=0; \
	  for d in golden/snr*; do \
	    tot=$$((tot+1)); \
	    line=$$(HLS_GOLDEN_DIR=$$d $(BUILD)/tb_n$$w 2>/dev/null | grep "start_index"); \
	    h=$$(echo "$$line" | sed -n 's/.*hls=\([0-9-]*\).*/\1/p'); \
	    p=$$(echo "$$line" | sed -n 's/.*python=\([0-9-]*\).*/\1/p'); \
	    [ "$$h" = "$$p" ] && ok=$$((ok+1)); \
	  done; \
	  printf "  W=%-3s %2s / %-2s\n" $$w $$ok $$tot; \
	done
