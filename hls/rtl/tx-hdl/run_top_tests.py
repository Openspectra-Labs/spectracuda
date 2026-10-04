"""Full chain with simulation-only inverse DFT; not vendor IFFT signoff."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import subprocess,json,uuid,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;REF=HERE.parent/"build/spectracuda_ref"
commit=subprocess.check_output(["git","-C",str(REF),"rev-parse","HEAD"],text=True).strip()
if not commit.startswith("ad0a396"):raise SystemExit("wrong reference")
sys.path.insert(0,str(REF))
from spectracuda.pipeline import Ofdm
from spectracuda.framing.header import HeaderCodec
reference=HERE/"reference"
if not (reference/"mapper.mem").exists():raise SystemExit("run_bit_tests.py and run_freq_tests.py first")
base=Ofdm(fft_size=256,cp_len=32,n_data=216,n_pilot=8)
def quant(x,scale):
    return np.clip(np.rint(np.asarray(x)*scale),-32768,32767).astype(np.int64)
pre=quant(np.real(base._preamble_time),32768)+1j*quant(np.imag(base._preamble_time),32768)
(reference/"preamble.mem").write_text("".join(f"{((int(x.real)&65535)<<16)|(int(x.imag)&65535):08x}\n" for x in pre))
build=HERE/"build"/uuid.uuid4().hex[:12];build.mkdir(parents=True)
inputs=[];expected=[];user=bytes.fromhex("0123456789abcdef")
cases=[(8,"qam64"),(64,"qpsk"),(2000,"qam16"),(16384,"qam64"),
       (512,"qpsk"),(16,"qam64")]
codec=HeaderCodec();positions=np.unique(np.linspace(0,215,112).round().astype(int))
fill_positions=np.setdiff1d(np.arange(216),positions)
for frame,(length,mod) in enumerate(cases):
    ofdm=Ofdm(fft_size=256,cp_len=32,n_data=216,n_pilot=8,modem=mod,
              fec1="conv_v27",interleaver="block",interleaver_kwargs={"unit_bits":8})
    code={"qpsk":1,"qam16":2,"qam64":3}[mod];bps=ofdm.modem.bits_per_symbol
    raw=np.random.default_rng(length+frame).integers(0,2,size=(1,length),dtype=np.uint8)
    payload=np.packbits(raw[0]);inputs.append(f"{length} {code} {user.hex()} {frame%4} {len(payload)}\n")
    inputs.extend(f"{int(x)}\n" for x in payload)
    encoded=np.asarray(ofdm.packetizer.encode(raw))[0]
    padded=np.concatenate([encoded,ofdm._payload_filler_bits[:(-len(encoded))%(216*bps)]])
    header=np.empty(216,dtype=np.uint8)
    header[positions]=codec.encode_bits(length,mod,"none",user,"none","conv_v27")
    header[fill_positions]=ofdm._header_filler_bits
    pilots=np.ones((1,8),dtype=np.complex64)
    from spectracuda.modem import Modem
    grids=[ofdm._train_grid_freq,ofdm.grid.scatter(np,pilots,Modem("bpsk").modulate(header[None,:]))[0]]
    for chunk in padded.reshape(-1,216*bps):
        grids.append(ofdm.grid.scatter(np,pilots,ofdm.modem.modulate(chunk[None,:]))[0])
    wave=[pre]
    for grid in grids:
        fixed=quant(grid.real,16384)+1j*quant(grid.imag,16384)
        # Floating model rounds the unscaled inverse sum, then TD rounds /128.
        z=np.fft.ifft(fixed)*256
        def round_away(x):return np.sign(x)*np.floor(np.abs(x)+0.5)
        re=np.clip(round_away(round_away(z.real)/128),-32768,32767)
        im=np.clip(round_away(round_away(z.imag)/128),-32768,32767)
        time=re+1j*im;wave.append(np.concatenate([time[-32:],time]))
    wave=np.concatenate(wave)
    for i,z in enumerate(wave):expected.append([int(z.real),int(z.imag),frame%4,int(i==0),int(i==len(wave)-1)])
(build/"input.txt").write_text("".join(inputs))
(build/"expected.txt").write_text("".join(" ".join(map(str,r))+"\n" for r in expected))
sources=[str(p.relative_to(HERE)) for p in sorted((HERE/"src").glob("*.v"))]+["tb/tx_xfft_256_model.v","tb/tx_top_tb.v"]
p=subprocess.run(["verilator","--binary","--timing","--top-module","tx_top_tb",
                  "--Mdir",str(build/"obj"),"-o","tx_top_tb"]+sources,cwd=HERE,capture_output=True,text=True)
(build/"compile.log").write_text(p.stdout+p.stderr)
if p.returncode:raise SystemExit(f"compile failure {build}")
runs={}
for name,extra in (("normal",[]),("underrun_frame2",["+abort_frame=2"]),("bad_symbol_frame3",["+bad_frame=3"])):
    p=subprocess.run([str(build/"obj/tx_top_tb"),f"+input={build/'input.txt'}",f"+expected={build/'expected.txt'}"]+extra,
                     cwd=HERE,capture_output=True,text=True,timeout=120)
    (build/f"simulation_{name}.log").write_text(p.stdout+p.stderr)
    runs[name]=p.returncode==0 and "PASS frames=" in p.stdout
    print(name,"PASS" if runs[name] else "FAIL",(p.stdout.strip().splitlines() or [""])[-1])
ok=all(runs.values())
(build/"results.json").write_text(json.dumps(dict(python_ref=commit,passed=ok,runs=runs,
    oracle="floating inverse DFT, quantized input; NOT vendor bit-accurate model",
    cases=cases,rtl_sha256={s:hashlib.sha256((HERE/s).read_bytes()).hexdigest() for s in sources}),indent=2)+"\n")
print(f"results: {build.relative_to(HERE)}/results.json")
raise SystemExit(0 if ok else 1)
