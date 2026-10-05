"""header_decode_v3 vs pinned HeaderCodec: random headers, correctable bit
errors (must decode identically), heavy corruption (must fail crc)."""
import sys, os, subprocess, uuid
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_ref_v3 as g
g.use()
import numpy as np
o = g.ofdm("qam64", 16)
codec = o.header_codec
pos = np.asarray(o._header_positions_flat).astype(int)
fpos = np.asarray(o._header_filler_positions).astype(int)
rng = np.random.default_rng(11)
lines, want = [], []
from spectracuda.framing.header import MOD_SCHEME_CODES
for t in range(60):
    L = int(rng.integers(8, 65535)); mod = ["qpsk", "qam16", "qam64"][t % 3]
    dm = [0, 16, 32, 64][t % 4]; user = bytes(rng.integers(0, 256, 6).tolist())
    wire = np.asarray(codec.encode_bits(L, mod, "none", user, "none", "conv_v27", dm, 0)).astype(int)
    flat = np.zeros(432, int); flat[pos] = wire; flat[fpos] = o._header_filler_bits
    kind = t % 5
    if kind == 3:                      # 3 scattered wire-bit errors: correctable
        for k in rng.choice(268, 3, replace=False): flat[pos[k]] ^= 1
    if kind == 4:                      # 60 errors: must not validate
        for k in rng.choice(268, 60, replace=False): flat[pos[k]] ^= 1
    lines.append("".join(map(str, flat)))
    want.append((kind, L, MOD_SCHEME_CODES[mod], {0: 0, 16: 1, 32: 2, 64: 3}[dm], user.hex()))
build = os.path.join(HERE, "build", f"hdr3_{uuid.uuid4().hex[:8]}"); os.makedirs(build)
open(os.path.join(build, "in.txt"), "w").write("\n".join(lines) + "\n")
p = subprocess.run(["verilator", "--binary", "--timing", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC",
                    "-Isrc/generated", "-Isrc", "--top-module", "header_decode_v3_tb", "--Mdir",
                    os.path.join(build, "obj"), "-o", "tb", "tb/header_decode_v3_tb.v",
                    "src/header_decode_v3.v", "src/viterbi_dec_ovl.v"], cwd=HERE, capture_output=True, text=True)
if p.returncode: print(p.stdout[-3000:], p.stderr[-3000:]); raise SystemExit("compile failed")
r = subprocess.run([os.path.join(build, "obj", "tb"), f"+in={build}/in.txt"], cwd=HERE, capture_output=True, text=True)
got = [l.split()[1:] for l in r.stdout.splitlines() if l.startswith("HDR")]
bad = 0
for (kind, L, m, dm, u), f in zip(want, got):
    fv, cok, ver, ln, mod, crc, f0, f1, d, c2, user = f
    if kind == 4:
        ok = fv == "0"
    else:
        ok = (fv, cok, ver, int(ln), int(mod), crc, f0, f1, int(d), c2, user) == \
             ("1", "1", "3", L, m, "1", "0", "1", dm, "0", u)
    bad += not ok
    if not ok: print("MISMATCH", kind, (L, m, dm, u), f)
print(f"{len(got)}/{len(want)} decoded, {bad} wrong  ({sum(w[0]==3 for w in want)} with 3 bit errors, "
      f"{sum(w[0]==4 for w in want)} corrupted -> rejected)")
raise SystemExit(0 if bad == 0 and len(got) == len(want) else 1)
