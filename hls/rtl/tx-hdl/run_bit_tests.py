"""Pinned Python -> standalone TX BIT exact checks; writes only below tx-hdl."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import subprocess
import uuid
import numpy as np

HERE = Path(__file__).resolve().parent
RTL = HERE.parent
REF = RTL / "build/spectracuda_ref"
head = subprocess.check_output(["git", "-C", str(REF), "rev-parse", "HEAD"], text=True).strip()
if not head.startswith("ad0a396"):
    raise SystemExit("Expected existing ad0a396 reference worktree; no automatic checkout")
sys.path.insert(0, str(REF))
from spectracuda.pipeline import Ofdm
from spectracuda.framing.header import HeaderCodec

reference = HERE / "reference"
reference.mkdir(exist_ok=True)
def mem(name, values):
    (reference/name).write_text("".join(f"{int(x):x}\n" for x in values))
codec=HeaderCodec(scramble_seed=42)
positions=np.unique(np.linspace(0,215,112).round().astype(int))
select=np.zeros(216,dtype=np.uint8);select[positions]=1
mem("header_mask.mem",codec._scramble_mask)
mem("header_select.mem",select)
mem("header_filler.mem",np.random.default_rng(2024).integers(0,2,size=104))
fill=[]
for bps in (2,4,6):
    fill.extend(np.random.default_rng(7777).integers(0,2,size=216*bps))
    fill.extend([0]*(1296-216*bps))
mem("payload_filler.mem",fill)
build=HERE/"build"/uuid.uuid4().hex[:12]
build.mkdir(parents=True)
cmd=["verilator","--binary","--timing","--top-module","tx_bit_domain_tb",
     "--Mdir",str(build/"obj"),"-o","tx_bit_tb", "tb/tx_bit_domain_tb.v","src/tx_bit_domain.v"]
p=subprocess.run(cmd,cwd=HERE,capture_output=True,text=True)
(build/"compile.log").write_text(p.stdout+p.stderr)
if p.returncode: raise SystemExit(f"Build failed: {build/'compile.log'}")
results=[]
for length in (8,16,64,512,2000,16384):
    for mod,code,bps in (("qpsk",1,2),("qam16",2,4),("qam64",3,6)):
        ofdm=Ofdm(fft_size=256,cp_len=32,n_data=216,n_pilot=8,modem=mod,
                  fec1="conv_v27",interleaver="block",interleaver_kwargs={"unit_bits":8})
        raw=np.random.default_rng(length+code).integers(0,2,size=(1,length),dtype=np.uint8)
        payload=np.packbits(raw[0])
        encoded=np.asarray(ofdm.packetizer.encode(raw))[0]
        padding=(-len(encoded))%(216*bps)
        padded=np.concatenate([encoded,ofdm._payload_filler_bits[:padding]])
        user=bytes.fromhex("0123456789abcdef");fq=code%4
        content=codec.encode_bits(length,mod,"none",user,"none","conv_v27")
        hdr=np.empty(216,dtype=np.uint8);hdr[positions]=content
        hdr[select==0]=ofdm._header_filler_bits
        records=[[fq,0,0,0,0,0,1,0]]
        for sc,bit in enumerate(hdr): records.append([fq,1,1,sc,1,int(bit),0,0])
        for k in range(len(padded)//bps):
            bits=sum(int(x)<<j for j,x in enumerate(padded[k*bps:(k+1)*bps]))
            records.append([fq,3,2+k//216,k%216,bps,bits,0,int(k==len(padded)//bps-1)])
        fixture=build/f"{length}_{mod}.txt"
        fixture.write_text(f"{length} {code} {user.hex()} {fq} {len(payload)}\n"+
                           "".join(f"{int(x)}\n" for x in payload)+
                           "".join(" ".join(map(str,r))+"\n" for r in records))
        for stall in (0,1):
            p=subprocess.run([str(build/"obj/tx_bit_tb"),f"+case={fixture}",f"+stall={stall}"],
                             cwd=HERE,capture_output=True,text=True,timeout=60)
            log=build/f"{length}_{mod}_stall{stall}.log";log.write_text(p.stdout+p.stderr)
            ok=p.returncode==0 and "PASS items=" in p.stdout
            results.append(dict(bits=length,mod=mod,stall=stall,passed=ok,log=str(log.relative_to(HERE))))
            print(f"{'PASS' if ok else 'FAIL'} {length} {mod} stall={stall}")
manifest=dict(python_ref=head,rtl_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [HERE/"src/tx_bit_domain.v",HERE/"tb/tx_bit_domain_tb.v"]},results=results)
(build/"results.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(f"{sum(r['passed'] for r in results)}/{len(results)} passed; {build.relative_to(HERE)}/results.json")
raise SystemExit(0 if all(r['passed'] for r in results) else 1)
