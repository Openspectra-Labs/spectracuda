#!/bin/bash
# Capture the stage-boundary reference dumps (docs/rx_modular_architecture.md
# section 10) from the CURRENT, pre-refactor RTL into fixtures/frames/.
#
# Each case runs at C=1 and C=10; every dump (i1, i2, c1, o1, stimulus)
# must be byte-identical between the two (Rule 0) and the run must be
# bit-exact against the pinned Python reference. Only then is the C=1 set
# stored, with SHA256SUMS.
#
# Needs the pre-refactor hierarchy (tb/rx/capture_taps.vh). Run once, commit
# fixtures/frames/; the refactored stages are checked against it.
#
#   fpga/sim/setup_ref.sh && fpga/sim/capture_golden.sh
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"  # fpga/
PY="${PY:-$(cd .. && pwd)/.venv/bin/python}"
TMP=build/fixtures/frames_tmp; rm -rf "$TMP"; mkdir -p "$TMP"
DST=fixtures/frames
CASES=("f2000_qam64           2000 qam64 0   0"
       "f16384_qpsk           16384 qpsk 0   0"
       "f512_qam16            512 qam16  0   0"
       "f64_qpsk              64 qpsk    0   0"
       "f2000_qam64_cfo0p3    2000 qam64 0.3 0"
       "f16384_qam64_evm0p12  16384 qam64 0  0.12"
       "f16384_qam16_cfo0p2_evm0p08 16384 qam16 0.2 0.08")
fail=0
for c in "${CASES[@]}"; do
  set -- $c; name=$1
  for cps in 1 10; do
    "$PY" sim/run_frame.py --bits $2 --modem $3 --cfo $4 --evm $5 --cps $cps \
        --capture "$TMP/$name/c$cps" > "$TMP/$name.c$cps.log" 2>&1
    if ! grep -q "VERDICT: PASS -- bit-exact" "$TMP/$name.c$cps.log"; then
      echo "$name C=$cps: NOT bit-exact vs Python -- see $TMP/$name.c$cps.log"; fail=1; continue 2
    fi
  done
  for f in stim.hex i1.txt i2.txt c1.txt o1.txt; do
    if ! cmp -s "$TMP/$name/c1/$f" "$TMP/$name/c10/$f"; then
      echo "$name: $f differs between C=1 and C=10"; fail=1; continue 2
    fi
  done
  rm -rf "$DST/$name"; mkdir -p "$DST/$name"
  cp "$TMP/$name/c1/"* "$DST/$name/"
  (cd "$DST/$name" && sha256sum stim.hex i1.txt i2.txt c1.txt o1.txt > SHA256SUMS)
  echo "$name: OK (bit-exact vs Python; identical at C=1 and C=10; $(wc -l < $DST/$name/i1.txt) bins, $(wc -l < $DST/$name/i2.txt) LLR groups)"
done
exit $fail
