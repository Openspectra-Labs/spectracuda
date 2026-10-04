"""Exact mapper/grid and 125->100MHz stream checks; TX-local artifacts only."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import subprocess,json,uuid,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
REF=HERE.parent/"build/spectracuda_ref"
commit=subprocess.check_output(["git","-C",str(REF),"rev-parse","HEAD"],text=True).strip()
if not commit.startswith("ad0a396"): raise SystemExit("Reference must be ad0a396")
sys.path.insert(0,str(REF))
from spectracuda.pipeline import Ofdm
from spectracuda.modem import Modem
reference=HERE/"reference";reference.mkdir(exist_ok=True)
def iq(x):
    re=int(np.rint(float(np.real(x))*16384));im=int(np.rint(float(np.imag(x))*16384))
    return ((re&65535)<<16)|(im&65535)
def mem(name,values,width):
    (reference/name).write_text("".join(f"{int(x):0{width}x}\n" for x in values))
mapping=[]
for mod,n in (("bpsk",1),("qpsk",2),("qam16",4),("qam64",6)):
    labels=np.array([[(k>>j)&1 for j in range(n)] for k in range(64)],dtype=np.uint8)
    mapping.extend(iq(x) for x in Modem(mod).modulate(labels.reshape(1,-1))[0])
ofdm=Ofdm(fft_size=256,cp_len=32,n_data=216,n_pilot=8,modem="qam64",
          fec1="conv_v27",interleaver="block",interleaver_kwargs={"unit_bits":8})
mem("mapper.mem",mapping,8)
mem("grid_type.mem",[0 if x==0 else (2 if x==1 else 1) for x in ofdm.grid.sctype],1)
mem("data_bins.mem",ofdm.grid.data_indices,2)
mem("training.mem",[iq(x) for x in ofdm._train_grid_freq],8)
build=HERE/"build"/uuid.uuid4().hex[:12];build.mkdir(parents=True)
inputs=[];outputs=[]
# TRAIN+HEADER+three DATA symbols per frame; all labels and wrapped fseq.
for frame in range(6):
    fq=frame%4
    inputs.append([fq,0,0,0,0,0,1,0])
    grids=[np.asarray(ofdm._train_grid_freq)]
    for sym,(mod,n) in enumerate((("bpsk",1),("qpsk",2),("qam16",4),("qam64",6)),1):
        bits=np.array([[(sc+frame>>j)&1 for j in range(n)] for sc in range(216)],dtype=np.uint8)
        for sc in range(216):
            value=sum(int(x)<<j for j,x in enumerate(bits[sc]))
            inputs.append([fq,1 if sym==1 else 3,sym,sc,n,value,0,int(sym==4 and sc==215)])
        points=Modem(mod).modulate(bits.reshape(1,-1))
        grids.append(ofdm.grid.scatter(np,np.ones((1,8),dtype=np.complex64),points)[0])
    for sym,grid in enumerate(grids):
        for bin,x in enumerate(grid):
            q=iq(x);re=(q>>16);im=q&65535
            re=re-65536 if re>=32768 else re;im=im-65536 if im>=32768 else im
            outputs.append([fq,0 if sym==0 else (1 if sym==1 else 3),sym,bin,re,im,
                            int(sym==0 and bin==0),int(sym==4 and bin==255)])
for name,records in (("input.txt",inputs),("output.txt",outputs)):
    (build/name).write_text("".join(" ".join(map(str,r))+"\n" for r in records))
sources=["src/tx_freq_domain.v","src/tx_stream_cdc.v","tb/tx_freq_domain_tb.v"]
p=subprocess.run(["verilator","--binary","--timing","--top-module","tx_freq_domain_tb",
                  "--Mdir",str(build/"obj"),"-o","tx_freq_tb"]+sources,cwd=HERE,capture_output=True,text=True)
(build/"compile.log").write_text(p.stdout+p.stderr)
if p.returncode:raise SystemExit(f"compile failed {build}")
results=[]
for stall,corrupt in ((0,-1),(1,-1),(0,500),(1,2000)):
    p=subprocess.run([str(build/"obj/tx_freq_tb"),f"+input={build/'input.txt'}",
                      f"+output={build/'output.txt'}",f"+stall={stall}",f"+corrupt={corrupt}"],cwd=HERE,capture_output=True,text=True,timeout=60)
    (build/f"stall{stall}_c{corrupt}.log").write_text(p.stdout+p.stderr)
    ok=p.returncode==0 and "PASS bins=" in p.stdout
    results.append(dict(stall=stall,corrupt=corrupt,passed=ok));print(p.stdout.strip() if ok else f"FAIL stall={stall} corrupt={corrupt}: {p.stdout}")
(build/"results.json").write_text(json.dumps(dict(python_ref=commit,results=results,
    rtl_sha256={s:hashlib.sha256((HERE/s).read_bytes()).hexdigest() for s in sources}),indent=2)+"\n")
print(f"results: {build.relative_to(HERE)}/results.json")
raise SystemExit(0 if all(r['passed'] for r in results) else 1)
