#!/usr/bin/env python3
"""E75: the definitive layer sweep — SP(ESM layer 1..33) per protein
(2026-08-03, settles the layer question: 4 proteins x 33 layers x 3 splits x
2 inits x 250ep, single-stream SP probes, best-epoch).

Usage: python e75_layer_sweep.py <protein>  (casp3|rbd|tim|ci)
Input: {protein}/faesm_all33/{mid}.pt [33,L,1280] fp16 (esm_extract_all33.py).
BS: 128 (casp3/rbd/tim), 32 (ci convention). Output:
{protein}/e75_layer_sweep/results.json (incremental) + log.txt.
"""
import os,sys
PROT=sys.argv[1]
GPU={'casp3':'1','rbd':'1','tim':'0','ci':'0'}[PROT]  # 0=5090D, 1=6000D (WSL CUDA order, NOT nvidia-smi order)
os.environ['CUDA_VISIBLE_DEVICES']=GPU
import numpy as np,json,math,warnings,csv,re
import torch
import torch.nn as nn
from scipy import stats
from pathlib import Path
warnings.filterwarnings('ignore')
torch.backends.cuda.enable_flash_sdp(True)
DEV=torch.device('cuda:0')
sys.path.insert(0,'/mnt/j/conda_envs/foundry/DMS_Project')

ROOT=Path("/mnt/k/output_heads");BASE=ROOT/PROT
OUT=BASE/"e75_layer_sweep";OUT.mkdir(exist_ok=True)
LOG=open(OUT/"log.txt","w",1)
def log(m):print(m,flush=True);LOG.write(m+"\n")

DM,DO,FD=256,0.1,128;EPOCHS,LR,WD=250,1e-4,0.05;NH=8
BS=32 if PROT=='ci' else 128

class PE(nn.Module):
    def __init__(self,d,m=600):super().__init__();pe=torch.zeros(m,d);pos=torch.arange(m).float().unsqueeze(1);div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000)/d));pe[:,0::2]=torch.sin(pos*div);pe[:,1::2]=torch.cos(pos*div);self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x):return x+self.pe[:,:x.size(1)]
class Enc(nn.Module):
    def __init__(self,di):super().__init__();self.p=nn.Linear(di,DM);self.pe=PE(DM);L_=nn.TransformerEncoderLayer(DM,NH,DM*4,DO,batch_first=True);self.e=nn.TransformerEncoder(L_,3);self.o=nn.Sequential(nn.Linear(DM,DM*2),nn.GELU(),nn.Linear(DM*2,FD))
    def forward(self,x):return self.o(self.e(self.pe(self.p(x))))
class AP(nn.Module):
    def __init__(self,d):super().__init__();self.w=nn.Linear(d,1)
    def forward(self,x):a=torch.softmax(self.w(x).squeeze(-1),1);return(x*a.unsqueeze(-1)).sum(1)
class MLP(nn.Module):
    def __init__(self,d):super().__init__();self.net=nn.Sequential(nn.Linear(d,d//2),nn.GELU(),nn.Dropout(DO),nn.Linear(d//2,1))
    def forward(self,x):return self.net(x).squeeze(-1)
class SP(nn.Module):
    def __init__(self,di):super().__init__();self.e=Enc(di);self.p=AP(FD);self.h=MLP(FD)
    def forward(self,x):return self.h(self.p(self.e(x)))

def load_meta():
    if PROT=='casp3':
        M=[json.loads(l) for l in open(BASE/"manifest.jsonl") if l.strip()]
        M=[m for m in M if m['position']<=244]
        return [(m['mutant_id'],m['label'],m['position']-1) for m in M]
    if PROT=='rbd':
        M=[json.loads(l) for l in open(BASE/"metadata.jsonl") if l.strip()]
        return [(m['mutant_id'],m['bind_avg'],m['site_rbd']-1) for m in M]
    if PROT=='tim':
        M=[json.loads(l) for l in open(BASE/"metadata.jsonl") if l.strip()]
        seen=set();rows=[]
        for m in M:
            if m['mutant_id'] in seen:continue
            seen.add(m['mutant_id']);rows.append((m['mutant_id'],m['score'],m['site']-1))
        return rows
    if PROT=='ci':
        rows=list(csv.DictReader(open('/mnt/j/conda_envs/foundry/DMS_Project/data/DMS_ProteinGym_substitutions/RPC1_LAMBD_Li_2019_high-expression.csv')))
        out=[]
        for i,r in enumerate(rows):
            m=re.match(r'[A-Z](\d+)[A-Z]',r['mutant'])
            out.append((f"ci_{i}",float(r['DMS_score']),int(m.group(1))-1))
        return out

META=load_meta();N=len(META)
L={'casp3':244,'rbd':201,'tim':254,'ci':237}[PROT]
log("Loading %d %s mutants (L=%d)..."%(N,PROT,L))
Xall=np.zeros((33,N,L,1280),dtype=np.float16)
yb=np.zeros(N);sites=np.zeros(N,dtype=int)
for i,(mid,lab,pos) in enumerate(META):
    Xall[:,i]=torch.load(BASE/"faesm_all33"/f"{mid}.pt",map_location="cpu",weights_only=True).numpy()
    yb[i]=lab;sites[i]=pos
y=(yb-yb.mean())/max(yb.std(),1e-8);yt=torch.tensor(y).to(DEV)

def make_split(seed):
    r=np.random.RandomState(seed);ap=sorted(np.unique(sites));r.shuffle(ap)
    vp=set(ap[:max(1,int(len(ap)*0.3))])
    return (np.array([i for i,p in enumerate(sites) if p not in vp]),
            np.array([i for i,p in enumerate(sites) if p in vp]))

def run_layer(k,split_seed):
    X=np.array(Xall[k-1],dtype=np.float32)
    for i in range(L):s=X[:,i].std(axis=0);X[:,i]=(X[:,i]-X[:,i].mean(axis=0))/np.maximum(s,1e-8)
    Xc=torch.tensor(X).to(DEV);del X
    tr,va=make_split(split_seed)
    def eval_preds(m):
        ep=[]
        for i in range(0,len(va),BS):
            ids=va[i:min(i+BS,len(va))]
            ep.append(m(Xc[ids]).detach())
        return torch.cat(ep)
    def rho_of(pv):r,_=stats.spearmanr(pv.cpu().numpy(),yt[va].cpu().numpy());return float(r) if not np.isnan(r) else -1.0
    vals=[]
    for init in range(2):
        torch.manual_seed(init*100+7)
        m=SP(1280).to(DEV)
        opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
        best=-1
        for ep in range(EPOCHS):
            m.train();idx=torch.randperm(len(tr))
            for bi in range(0,len(tr),BS):
                b=idx[bi:bi+BS]
                pred=m(Xc[tr][b])
                l=nn.MSELoss()(pred,yt[tr][b]);opt.zero_grad();l.backward();opt.step()
            sched.step();m.eval()
            with torch.no_grad():best=max(best,rho_of(eval_preds(m)))
        vals.append(float(best))
    del Xc;torch.cuda.empty_cache()
    return dict(split=split_seed,mean=float(np.mean(vals)),rho_vals=vals)

RES={}
if (OUT/"results.json").exists():
    RES=json.load(open(OUT/"results.json"))
for k in range(1,34):
    name=f"L{k}"
    if name in RES:
        log("  %s cached, skip"%name);continue
    runs=[run_layer(k,sd) for sd in [42,43,44]]
    allrho=[v for r in runs for v in r['rho_vals']]
    RES[name]=dict(runs=runs,mean=float(np.mean(allrho)),std=float(np.std(allrho)),
                   split_means=[r['mean'] for r in runs])
    log("  >> %-4s mu=%.4f sigma=%.4f splits=%s"%(name,RES[name]['mean'],RES[name]['std'],json.dumps([round(v,3) for v in RES[name]['split_means']])))
    json.dump(RES,open(OUT/"results.json","w"),indent=2)

log("\n"+"="*60)
means=[(k,RES[f"L{k}"]['mean']) for k in range(1,34)]
for k,v in means:log("  L%-3d %.4f"%(k,v))
bk,bv=max(means,key=lambda t:t[1])
log("BEST: L%d = %.4f (L33 = %.4f)"%(bk,bv,RES['L33']['mean']))
log("DONE");LOG.close()
