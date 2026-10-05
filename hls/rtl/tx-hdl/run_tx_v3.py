"""v3 TX full chain (protected header, interleaver2, DMRS) vs pinned Python.

Every ROM the TX reads is regenerated here from the pinned reference
(golden_ref_v3). Expected samples come from Python's OWN frequency grids:
Ofdm.mod.process (the IFFT) is wrapped to record its inputs while
generate_frame() runs, then the symbols are laid out in transmission order
(train, 2 header symbols, data with DMRS = training grid at the slot-map
positions). Each grid is quantized to Q14 and passed through the same
exact-integer IFFT/rounding oracle the TX TB uses (1 LSB tolerance for the
floating IFFT model). A second check runs Python's own receiver on the RTL
samples and requires the payload back exactly.

    python3 run_tx_v3.py
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import subprocess, json, uuid, hashlib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "gen"))
import golden_ref_v3 as g
commit = g.use()
from spectracuda.modem import Modem
import hdr_v3_model

ref = HERE / "reference"
ref.mkdir(exist_ok=True)


def iq(x):
    re = int(np.rint(float(np.real(x)) * 16384)); im = int(np.rint(float(np.imag(x)) * 16384))
    return ((re & 65535) << 16) | (im & 65535)


def mem(name, values, width=1):
    (ref / name).write_text("".join(f"{int(x):0{width}x}\n" for x in values))


# ---- ROMs from the pinned reference ----
base = g.ofdm("qam64", 16)
codec = base.header_codec
pos = np.asarray(base._header_positions_flat).astype(int).ravel()
sel = np.zeros(432, int); sel[pos] = 1
mem("hdr3_mask.mem", codec._scramble_mask)
mem("hdr3_sel.mem", sel)
mem("hdr3_fill.mem", base._header_filler_bits)
fill = []
for bps in (2, 4, 6):
    o = g.ofdm({2: "qpsk", 4: "qam16", 6: "qam64"}[bps])
    fill.extend(int(x) for x in o._payload_filler_bits[:216 * bps])
    fill.extend([0] * (1296 - 216 * bps))
mem("payload_filler.mem", fill)
mapping = []
for mod, n in (("bpsk", 1), ("qpsk", 2), ("qam16", 4), ("qam64", 6)):
    labels = np.array([[(k >> j) & 1 for j in range(n)] for k in range(64)], dtype=np.uint8)
    mapping.extend(iq(x) for x in Modem(mod).modulate(labels.reshape(1, -1))[0])
mem("mapper.mem", mapping, 8)
mem("grid_type.mem", [0 if x == 0 else (2 if x == 1 else 1) for x in base.grid.sctype])
mem("data_bins.mem", base.grid.data_indices, 2)
mem("training.mem", [iq(x) for x in base._train_grid_freq], 8)


def quant(x, scale):
    return np.clip(np.rint(np.asarray(x) * scale), -32768, 32767).astype(np.int64)


pre = quant(np.real(base._preamble_time), 32768) + 1j * quant(np.imag(base._preamble_time), 32768)
(ref / "preamble.mem").write_text("".join(f"{((int(x.real)&65535)<<16)|(int(x.imag)&65535):08x}\n" for x in pre))


def round_away(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def frame_grids(o, raw, user):
    """Python's IFFT inputs for one frame, in transmission order."""
    seen = []
    real = o.mod.process
    def spy(freq):
        seen.append(np.asarray(freq).reshape(-1, 256).copy())
        return real(freq)
    o.mod.process = spy
    try:
        o.generate_frame(raw, user_data=user)
    finally:
        o.mod.process = real
    train = seen[0][0]
    hdr = [seen[1][0], seen[2][0]]
    data = seen[3]
    from spectracuda.framing import dmrs as D
    order = [train] + hdr
    k = 0
    for s in D.dmrs_slot_map(len(data), o.dmrs_interval):
        if s == D.DMRS_SLOT:
            order.append(train)
        else:
            order.append(data[k]); k += 1
    assert len(seen) == 4
    return order


cases = [(64, "qpsk", 0), (2000, "qam16", 16), (16384, "qam64", 16), (512, "qpsk", 32),
         (24000, "qpsk", 16), (24000, "qam16", 64), (8, "qam64", 0), (30000, "qam64", 32)]
build = HERE / "build" / f"v3_{uuid.uuid4().hex[:12]}"
build.mkdir(parents=True)
inputs, expected, payloads = [], [], []
for frame, (length, mod, dm) in enumerate(cases):
    o = g.ofdm(mod, dm)
    user = bytes(np.random.default_rng(frame).integers(0, 256, 6).tolist())
    raw = np.random.default_rng(length + frame).integers(0, 2, size=(1, length), dtype=np.uint8)
    payload = np.packbits(raw[0]); payloads.append((o, raw))
    inputs.append(f"{length} {g.MOD_CODE[mod]} {user.hex()} {frame % 4} {len(payload)} {g.DMRS_CODE[dm]}\n")
    inputs.extend(f"{int(x)}\n" for x in payload)
    wave = [pre]
    for grid in frame_grids(o, raw, user):
        fixed = quant(grid.real, 16384) + 1j * quant(grid.imag, 16384)
        z = np.fft.ifft(fixed) * 256
        re = np.clip(round_away(round_away(z.real) / 128), -32768, 32767)
        im = np.clip(round_away(round_away(z.imag) / 128), -32768, 32767)
        t = re + 1j * im
        wave.append(np.concatenate([t[-32:], t]))
    wave = np.concatenate(wave)
    for i, z in enumerate(wave):
        expected.append([int(z.real), int(z.imag), frame % 4, int(i == 0), int(i == len(wave) - 1)])
(build / "input.txt").write_text("".join(inputs))
(build / "expected.txt").write_text("".join(" ".join(map(str, r)) + "\n" for r in expected))

sources = [str(p.relative_to(HERE)) for p in sorted((HERE / "src").glob("*.v"))] + \
          ["tb/tx_xfft_256_model.v", "tb/tx_top_tb.v"]
p = subprocess.run(["verilator", "--binary", "--timing", "--top-module", "tx_top_tb",
                    "--Mdir", str(build / "obj"), "-o", "tx_top_tb"] + sources,
                   cwd=HERE, capture_output=True, text=True)
(build / "compile.log").write_text(p.stdout + p.stderr)
if p.returncode:
    print(p.stdout[-3000:] + p.stderr[-3000:])
    raise SystemExit(f"compile failure {build}")
runs = {}
for name, extra in (("normal", [f"+dump={build/'rtl_iq.txt'}"]), ("underrun_frame2", ["+abort_frame=2"]),
                    ("bad_symbol_frame3", ["+bad_frame=3"])):
    p = subprocess.run([str(build / "obj/tx_top_tb"), f"+input={build/'input.txt'}",
                        f"+expected={build/'expected.txt'}"] + extra,
                       cwd=HERE, capture_output=True, text=True, timeout=600)
    (build / f"simulation_{name}.log").write_text(p.stdout + p.stderr)
    runs[name] = p.returncode == 0 and "PASS frames=" in p.stdout
    print(name, "PASS" if runs[name] else "FAIL", (p.stdout.strip().splitlines() or [""])[-1][:200])

# ---- Python's own receiver on the RTL samples ----
py_rx = {}
if runs["normal"]:
    iqs = np.loadtxt(build / "rtl_iq.txt", dtype=np.int64).reshape(-1, 2)
    starts = [i for i, r in enumerate(expected) if r[3]] + [len(expected)]
    for frame, (o, raw) in enumerate(payloads):
        seg = iqs[starts[frame]:starts[frame + 1]]
        x = (seg[:, 0] + 1j * seg[:, 1]) / 32768.0
        rx = np.concatenate([np.zeros(200), x, np.zeros(200)])[None, :].astype(np.complex64)
        r = o.rx_process(rx)
        ok = bool(r.get("frame_found")) and np.array_equal(np.asarray(r["bits"]).ravel()[:raw.shape[1]], raw.ravel())
        py_rx[f"{cases[frame]}"] = ok
        print(f"python RX on RTL frame {frame} {cases[frame]}: {'payload exact' if ok else 'FAIL'}")
ok = all(runs.values()) and py_rx and all(py_rx.values())
(build / "results.json").write_text(json.dumps(dict(python_ref=commit, passed=bool(ok), runs=runs, python_rx=py_rx,
    cases=cases, rtl_sha256={s: hashlib.sha256((HERE / s).read_bytes()).hexdigest() for s in sources}), indent=2) + "\n")
print(f"{'PASS' if ok else 'FAIL'} v3 TX; {build.relative_to(HERE)}")
raise SystemExit(0 if ok else 1)
