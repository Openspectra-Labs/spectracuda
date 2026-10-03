#!/bin/bash
# Re-capture every golden_if/ case from the CURRENT RTL and require every
# dump (i1, i2, c1, o1, stimulus) to be byte-identical to the committed
# reference -- the bit-exact gate for each refactor step.
#
#   hls/rtl/check_golden.sh [C ...]     (default: 1 10)
#
# Uses tb/capture_taps.vh, i.e. the pre-refactor hierarchy; update the taps
# when a stage is replaced (the I1/I2 buses then exist at top level).
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
PY="${PY:-$(cd ../.. && pwd)/.venv/bin/python}"
RATES="${*:-1 10}"
TMP=build/check_golden; rm -rf "$TMP"; mkdir -p "$TMP"
fail=0
for d in golden_if/f*/; do
  name=$(basename "$d")
  args=$("$PY" -c "import json;m=json.load(open('$d/meta.json'));print(m['bits'],m['modem'],m['cfo'],m['evm'])")
  set -- $args
  for c in $RATES; do
    out="$TMP/$name.c$c"
    "$PY" run_frame.py --bits $1 --modem $2 --cfo $3 --evm $4 --cps $c --capture "$out" > "$out.log" 2>&1
    bad=""
    for f in stim.hex i1.txt i2.txt c1.txt o1.txt; do
      cmp -s "$d/$f" "$out/$f" || bad="$bad $f"
    done
    v=$(grep -oE "VERDICT: .*" "$out.log" || echo "VERDICT: ERROR")
    if [ -z "$bad" ] && echo "$v" | grep -q "PASS"; then
      echo "$name C=$c: identical to golden ($v)"
    else
      echo "$name C=$c: MISMATCH:${bad:- none} ($v) -- see $out"; fail=$((fail+1))
    fi
  done
done
echo "FAILURES: $fail"
exit $([ $fail -eq 0 ] && echo 0 || echo 1)
