#!/usr/bin/env python3
"""E75b: layer sweep on the FUSION cell — XAttn(ESM layer 1..33, Z_II nomsa)
(2026-08-03, companion of E75; the user's point: SP-only doesn't answer our
mechanism question — we need the layer effect inside the fusion, where the
loop lives).

Usage: python e75b_layer_sweep_xa.py <protein>  (casp3|rbd|tim|ci)
Same splits (42/43/44) x inits (7/107) as E75 -> paired SP-vs-XA per layer.
z-stream = canonical zii-grid (casp3 row slices tile-488; rbd/tim [0] row;
ci artifacted ci/zii). 250ep, BS=128 (ci 32), best-epoch, per-position
per-channel std. Output {protein}/e75b_layer_sweep_xa/results.json.
"""
import os,sys
PROT=sys.argv[1]
GPU=os.environ.get('E75B_GPU',{'casp3':'1','rbd':'1','tim':'0','ci':'0'}[PROT])  # 0=5090D, 1=6000D; env override for 5090D deaths
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
from fusion_v3 import CrossAttnOrthoConcatFusion
from throttle import Throttle
THR=int(os.environ.get('THROTTLE_RATIO','1'))  # 5090D duty-cycle protection (0=off)

ROOT=Path("/mnt/k/output_heads");BASE=ROOT/PROT
OUT=BASE/"e75b_layer_sweep_xa";OUT.mkdir(exist_ok=True)
LOG=open(OUT/"log.txt","w",1)
def log(m):print(m,flush=True);LOG.write(m+"\n")

BS=32 if PROT=='ci' else 128
EPOCHS,LR,WD=250,1e-4,0.05

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
LZ=2*L if PROT=='casp3' else L
log("Loading %d %s mutants (L=%d)..."%(N,PROT,L))
Xz=np.zeros((N,LZ,128),dtype=np.float32)
yb=np.zeros(N);sites=np.zeros(N,dtype=int)
for i,(mid,lab,pos) in enumerate(META):
    if PROT=='casp3':
        Xz[i]=torch.load(BASE/"zii"/f"{mid}_zii.pt",map_location="cpu")[pos,:LZ,:].float().numpy()
    elif PROT=='ci':
        Xz[i]=torch.load(BASE/"zii"/f"{mid}.pt",map_location="cpu").float().numpy()
    else:
        Xz[i]=torch.load(BASE/"zii"/f"{mid}_zii.pt",map_location="cpu")[0,:LZ,:].float().numpy()
    yb[i]=lab;sites[i]=pos
for i in range(LZ):s=Xz[:,i].std(axis=0);Xz[:,i]=(Xz[:,i]-Xz[:,i].mean(axis=0))/np.maximum(s,1e-8)
Xzc=torch.from_numpy(Xz).float().to(DEV);del Xz
y=(yb-yb.mean())/max(yb.std(),1e-8);yt=torch.tensor(y).to(DEV)

def make_split(seed):
    r=np.random.RandomState(seed);ap=sorted(np.unique(sites));r.shuffle(ap)
    vp=set(ap[:max(1,int(len(ap)*0.3))])
    return (np.array([i for i,p in enumerate(sites) if p not in vp]),
            np.array([i for i,p in enumerate(sites) if p in vp]))

def prep_layer(k):
    X=np.zeros((N,L,1280),dtype=np.float32)
    for i,(mid,_,_) in enumerate(META):
        X[i]=torch.load(BASE/"faesm_all33"/f"{mid}.pt",map_location="cpu",weights_only=True)[k-1].float().numpy()
    for i in range(L):s=X[:,i].std(axis=0);X[:,i]=(X[:,i]-X[:,i].mean(axis=0))/np.maximum(s,1e-8)
    if PROT=='casp3':X=np.tile(X,(1,2,1))
    Xfc=torch.tensor(X).to(DEV);del X
    return Xfc

def run_layer(Xfc,k,split_seed):
    tr,va=make_split(split_seed)
    def eval_preds(m):
        ep=[]
        for i in range(0,len(va),BS):
            ids=va[i:min(i+BS,len(va))]
            ep.append(m(Xzc[ids],Xfc[ids])[0].squeeze(-1).detach())
        return torch.cat(ep)
    def rho_of(pv):r,_=stats.spearmanr(pv.cpu().numpy(),yt[va].cpu().numpy());return float(r) if not np.isnan(r) else -1.0
    vals=[]
    for init in range(2):
        torch.manual_seed(init*100+7)
        m=CrossAttnOrthoConcatFusion().to(DEV)
        opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
        TH=Throttle(m,Xzc,Xfc,yt,va,DEV,ratio=THR,log=log) if THR>0 else None
        best=-1;gstep=0
        for ep in range(EPOCHS):
            m.train();idx=torch.randperm(len(tr))
            for bi in range(0,len(tr),BS):
                b=idx[bi:bi+BS]
                pred=m(Xzc[tr][b],Xfc[tr][b])[0]
                l=nn.MSELoss()(pred.squeeze(-1),yt[tr][b]);opt.zero_grad();l.backward()
                if TH:TH.capture_grads()
                opt.step()
                if TH:TH.on_step(gstep)
                gstep+=1
            sched.step();m.eval()
            with torch.no_grad():best=max(best,rho_of(eval_preds(m)))
        vals.append(float(best))
    return dict(split=split_seed,mean=float(np.mean(vals)),rho_vals=vals)

RES={}
if (OUT/"results.json").exists():
    RES=json.load(open(OUT/"results.json"))
for k in range(1,34):
    name=f"L{k}"
    if name in RES:
        log("  %s cached, skip"%name);continue
    Xfc=prep_layer(k)
    runs=[run_layer(Xfc,k,sd) for sd in [42,43,44]]
    del Xfc;torch.cuda.empty_cache()
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
