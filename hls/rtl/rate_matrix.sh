#!/bin/bash
# Rate-invariance matrix (docs/rx_modular_architecture.md, Rule 0 / H11).
#
# Every frame config at every clocks-per-sample ratio C. Each run must be
# bit-exact against the pinned Python reference (golden_ref.py), AND its
# FFT output (I1) must be bit-identical to the same config at C=1.
#
#   hls/rtl/setup_ref.sh          # once
#   hls/rtl/rate_matrix.sh        # ~25 min, results in build/rate_matrix/
#
# Exit status 0 only if every run passes both checks.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
PY="${PY:-$(cd ../.. && pwd)/.venv/bin/python}"
OUT=build/rate_matrix; rm -rf "$OUT"; mkdir -p "$OUT"
RES="$OUT/results.txt"; : > "$RES"
RATES="${RATES:-1 2.5 5 10 20}"
CFGS=("2000 qam64 0 0" "16384 qpsk 0 0" "512 qam16 0 0" "64 qpsk 0 0"
      "2000 qam64 0.05 0" "16384 qam64 0 0" "16384 qam64 0 0.12"
      "2000 qam64 0.3 0" "16384 qam16 0.2 0.08")
fail=0; n=0
for cfg in "${CFGS[@]}"; do
  n=$((n+1)); set -- $cfg
  for c in $RATES; do
    log="$OUT/cfg${n}_c${c}.log"; i1="$OUT/cfg${n}_c${c}.i1"
    "$PY" run_frame.py --bits $1 --modem $2 --cfo $3 --evm $4 --cps $c --dump-i1 "$i1" > "$log" 2>&1
    v=$(grep -oE "VERDICT: .*" "$log" || echo "VERDICT: ERROR (see $log)")
    if [ "$c" = "$(echo $RATES | cut -d' ' -f1)" ]; then same="I1 ref"
    elif cmp -s "$OUT/cfg${n}_c$(echo $RATES | cut -d' ' -f1).i1" "$i1"; then same="I1 identical"
    else same="I1 DIFFERS"; fi
    case "$v$same" in *"PASS -- bit-exact"*"DIFFERS"*|*"DIVERGED"*|*"ERROR"*) fail=$((fail+1));; esac
    echo "cfg$n bits=$1 $2 cfo=$3 evm=$4 C=$c :: $v :: $same" | tee -a "$RES"
  done
done
echo "FAILURES: $fail" | tee -a "$RES"
exit $([ $fail -eq 0 ] && echo 0 || echo 1)
