"""demapper_soft.v vs spectracuda Modem.demodulate_soft_tableq (the
"thresh_wq" table metric), module level: random equalized points over and around
each constellation, random weight exponents k = -3..1, Q12 inputs fed to
both. Python sees the SAME quantized value (y_q / 4096), so the only
table and arithmetic are the same: soft values must be IDENTICAL."""
import os, sys, subprocess, uuid
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from spectracuda.modem import Modem
rng = np.random.default_rng(5)
lines, want = [], []
for code, mod in ((0, "qpsk"), (1, "qam16"), (2, "qam64")):
    m = Modem(mod)
    pts, _ = m._point_table()
    n = 3000
    base = pts[rng.integers(0, len(pts), n)]
    y = base + (rng.normal(size=n) + 1j * rng.normal(size=n)) * rng.choice([0.02, 0.1, 0.3], n)
    yq = np.clip(np.rint(y.real * 4096), -131072, 131071) + 1j * np.clip(np.rint(y.imag * 4096), -131072, 131071)
    k = rng.integers(-3, 2, n)
    soft = (np.asarray(m.demodulate_soft_tableq((yq / 4096)[None, :], k[None, :])).astype(int) - 128).reshape(n, -1)
    q = np.rint(soft * 7 / 127).astype(int)            # bytes -> -7..7, > 0 means bit 1
    for i in range(n):
        lines.append(f"{code} {int(yq[i].real)} {int(yq[i].imag)} {int(k[i]) + 3}")
        want.append(q[i])
d = os.path.join(HERE, "build", f"demap_soft_{uuid.uuid4().hex[:6]}"); os.makedirs(d)
open(os.path.join(d, "in.txt"), "w").write("\n".join(lines) + "\n")
r = subprocess.run(["verilator", "--binary", "--timing", "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Isrc/generated",
                    "-Isrc", "--top-module", "demapper_soft_tb", "--Mdir", os.path.join(d, "obj"), "-o", "tb",
                    "tb/demapper_soft_tb.v", "src/demapper_soft.v"], cwd=HERE, capture_output=True, text=True)
if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit("compile failed")
out = subprocess.run([os.path.join(d, "obj", "tb"), f"+in={d}/in.txt"], cwd=HERE, capture_output=True, text=True)
got = [list(map(int, l.split()[1:])) for l in out.stdout.splitlines() if l.startswith("O ")]
same = near = tot = 0
bad_shown = 0
for ii, (g, w) in enumerate(zip(got, want)):
    nb = g[0]
    rtl = -np.array(g[1:1 + nb])                      # RTL > 0 means bit 0
    diff = np.abs(rtl - w[:nb])
    same += int((diff == 0).sum()); near += int((diff <= 1).sum()); tot += nb
    if (diff != 0).any() and bad_shown < 8:
        bad_shown += 1; print("MISMATCH", lines[ii], "rtl", rtl.tolist(), "py", w[:nb].tolist())
print(f"{len(got)}/{len(want)} items, {tot} soft values: identical {100*same/tot:.3f}%, within one level {100*near/tot:.3f}%")
raise SystemExit(0 if len(got) == len(want) and same == tot else 1)
