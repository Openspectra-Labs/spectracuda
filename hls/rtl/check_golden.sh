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
    for f in stim.hex i2.txt c1.txt o1.txt; do
      cmp -s "$d/$f" "$out/$f" || bad="$bad $f"
    done
    # I1: the reference must be an exact PREFIX. Since step 3c the time
    # domain keeps capturing BODY symbols until the config says how many
    # there are, so it may emit a few extra trailing ones -- the frequency
    # domain drops them (frozen I1 contract: no frame_end). Every extra
    # item must be BODY, same frame, sym_idx past the declared length.
    extra=$("$PY" - "$d" "$out" <<'PYEOF'
import sys
gold = open(sys.argv[1] + "/i1.txt").read().splitlines()
new  = open(sys.argv[2] + "/i1.txt").read().splitlines()
body = int(dict(l.split() for l in open(sys.argv[1] + "/c1.txt"))["cfg_body_syms"])
if new[:len(gold)] != gold:
    print("MISMATCH"); sys.exit()
bad = [l for l in new[len(gold):]
       if not (l.split()[3] == "2" and int(l.split()[1]) >= 2 + body)]
print("BADEXTRA" if bad else f"OK+{(len(new) - len(gold)) // 256}")
PYEOF
)
    case "$extra" in OK*) ;; *) bad="$bad i1.txt($extra)";; esac
    v=$(grep -oE "VERDICT: .*" "$out.log" || echo "VERDICT: ERROR")
    if [ -z "$bad" ] && echo "$v" | grep -q "PASS"; then
      echo "$name C=$c: identical to golden, I1 $extra extra trailing BODY symbols dropped by FD ($v)"
    else
      echo "$name C=$c: MISMATCH:${bad:- none} ($v) -- see $out"; fail=$((fail+1))
    fi
  done
done
echo "FAILURES: $fail"
exit $([ $fail -eq 0 ] && echo 0 || echo 1)
