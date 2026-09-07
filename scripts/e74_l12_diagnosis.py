#!/usr/bin/env python3
"""E74: WHY does L12 make the loop milder? H1/H2/H3 discrimination (2026-08-03).

Three candidate explanations for L12's milder competition (E69/E71):
  H1 granularity: L12 carries lower-rank/insufficient information.
  H2 complementarity: L12 is less redundant with Z_II than L33 is.
  H3 SNR: L12 is simply worse (≈ built-in noise, E53 det05 analogue).
Measurements (RBD primary; CASP3 rank/CKA for contrast):
  (A) SP(l33) vs SP(l12) vs SP(zii): single-stream probes, 3 splits x 2 inits,
      300ep, BS=128, best-epoch. H3 predicts SP(l12) << SP(l33).
  (B) effective rank (eff90 + participation ratio) per stream — H1 predicts
      much lower rank for L12.
  (C) linear CKA(l33,zii) vs CKA(l12,zii) + ridge R2(f->z): H2 predicts
      LOWER redundancy for L12. Also CKA(l33,l12).
Device: CVD=0 (6000D).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'   # post-reboot: 0=6000D
import numpy as np,json,math,warnings,sys
import torch
import torch.nn as nn
from scipy import stats
from pathlib import Path
warnings.filterwarnings('ignore')
torch.backends.cuda.enable_flash_sdp(True)
DEV=torch.device('cuda:0')
sys.path.insert(0,'/mnt/j/conda_envs/foundry/DMS_Project')

RBD=Path("/mnt/k/output_heads/rbd");CASP3=Path("/mnt/k/output_heads/casp3")
OUT=RBD/"e74_l12_diagnosis";OUT.mkdir(exist_ok=True)
LOG=open(OUT/"log.txt","w",1)
def log(m):print(m,flush=True);LOG.write(m+"\n")

DM,BS,DO,FD=256,128,0.1,128;EPOCHS,LR,WD=300,1e-4,0.05;NH=8

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

def load_rbd():
    META=[json.loads(l) for l in open(RBD/"metadata.jsonl") if l.strip()];N=len(META);L=201
    X={k:np.zeros((N,L,d),dtype=np.float32) for k,d in [('zii',128),('l33',1280),('l12',1280)]}
    yb=np.zeros(N);sites=np.zeros(N,dtype=int)
    for i,m in enumerate(META):
        mid=m["mutant_id"]
        X['zii'][i]=torch.load(RBD/"zii"/f"{mid}_zii.pt",map_location="cpu")[0,:L,:].float().numpy()
        X['l33'][i]=torch.load(RBD/"faesm"/f"{mid}.pt",map_location="cpu")[:L].float().numpy()
        X['l12'][i]=torch.load(RBD/"faesm_l12"/f"{mid}.pt",map_location="cpu")[:L].float().numpy()
        yb[i]=m["bind_avg"];sites[i]=m["site_rbd"]-1
    return X,yb,sites

def stdz(X):
    for i in range(X.shape[1]):
        s=X[:,i].std(axis=0);X[:,i]=(X[:,i]-X[:,i].mean(axis=0))/np.maximum(s,1e-8)
    return X

def make_split(sites,seed):
    r=np.random.RandomState(seed);ap=sorted(np.unique(sites));r.shuffle(ap)
    vp=set(ap[:max(1,int(len(ap)*0.3))])
    return (np.array([i for i,p in enumerate(sites) if p not in vp]),
            np.array([i for i,p in enumerate(sites) if p in vp]))

# ---------- Part B: ranks, CKA, ridge redundancy (CPU, per protein) ----------
def effrank(X):
    """X [N,L,D] -> subsample positions, covariance spectrum, eff90 + PR."""
    V=X.reshape(-1,X.shape[-1])
    V=V[np.random.RandomState(0).choice(len(V),min(60000,len(V)),replace=False)]
    V=V-V.mean(0)
    sv=np.linalg.svd(V,compute_uv=False)**2
    sv=sv/sv.sum()
    eff90=int(np.searchsorted(np.cumsum(sv),0.90)+1)
    pr=float(1.0/np.sum(sv**2))
    return eff90,pr

def cka(X,Y,n=4000):
    """linear CKA between per-position vectors of X [N,L,dx], Y [N,L,dy]."""
    rng=np.random.RandomState(0)
    Xf=X.reshape(-1,X.shape[-1]);Yf=Y.reshape(-1,Y.shape[-1])
    idx=rng.choice(len(Xf),min(n,len(Xf)),replace=False)
    Xf=Xf[idx]-Xf[idx].mean(0);Yf=Yf[idx]-Yf[idx].mean(0)
    return float((np.linalg.norm(Yf.T@Xf,'fro')**2)/(np.linalg.norm(Xf.T@Xf,'fro')*np.linalg.norm(Yf.T@Yf,'fro')+1e-12))

def ridge_r2(X,Y,n=20000,lam=1e2):
    """R2 of reconstructing Y (zii) from X (f stream) by ridge, per-position."""
    rng=np.random.RandomState(0)
    Xf=X.reshape(-1,X.shape[-1]);Yf=Y.reshape(-1,Y.shape[-1])
    idx=rng.choice(len(Xf),min(n,len(Xf)),replace=False)
    Xf=Xf[idx];Yf=Yf[idx]
    ntr=int(0.8*len(Xf))
    Xtr,Xte=Xf[:ntr],Xf[ntr:];Ytr,Yte=Yf[:ntr],Yf[ntr:]
    mx,my=Xtr.mean(0),Ytr.mean(0)
    Xtr,Xte,Ytr,Yte=Xtr-mx,Xte-mx,Ytr-my,Yte-my
    W=np.linalg.solve(Xtr.T@Xtr+lam*np.eye(Xtr.shape[1]),Xtr.T@Ytr)
    P=Xte@W
    ss=1-np.sum((P-Yte)**2)/np.sum(Yte**2)
    return float(ss)

diag={}
for pname,loader in [("RBD",None)]:
    log("="*60);log("PART B (ranks/CKA/ridge): %s"%pname)
    X,yb,sites=load_rbd()
    Xs={k:stdz(v) for k,v in X.items()}
    for k,v in Xs.items():
        e90,pr=effrank(v)
        diag[f"effrank_{k}"]=dict(eff90=e90,pr=pr)
        log("  %-4s eff90=%d PR=%.1f"%(k,e90,pr))
    for a,b in [('l33','zii'),('l12','zii'),('l33','l12')]:
        c=cka(Xs[a],Xs[b])
        diag[f"cka_{a}_{b}"]=c
        log("  CKA(%s,%s)=%.4f"%(a,b,c))
    for a in ['l33','l12']:
        r2=ridge_r2(Xs[a],Xs['zii'])
        diag[f"ridge_{a}_to_zii"]=r2
        log("  ridge R2 %s->zii = %.4f"%(a,r2))
    json.dump(diag,open(OUT/"diag.json","w"),indent=2)

# ---------- Part A: SP probes ----------
log("="*60);log("PART A: SP probes (RBD)")
y=(yb-yb.mean())/max(yb.std(),1e-8);yt=torch.tensor(y).to(DEV)

def run_sp(Xnp,split_seed):
    Xc=torch.tensor(Xnp).to(DEV)
    tr,va=make_split(sites,split_seed)
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
        m=SP(Xnp.shape[-1]).to(DEV)
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
        log("  SP s%d init%d: rho=%.4f"%(split_seed,init+1,best))
    del Xc;torch.cuda.empty_cache()
    return dict(split=split_seed,mean=float(np.mean(vals)),rho_vals=vals)

sp_res={}
for k in ['zii','l33','l12']:
    runs=[run_sp(Xs[k],sd) for sd in [42,43,44]]
    allrho=[v for r in runs for v in r['rho_vals']]
    sp_res[k]=dict(runs=runs,mean=float(np.mean(allrho)),std=float(np.std(allrho)),
                   split_means=[r['mean'] for r in runs])
    log("  >> SP(%s) mu=%.4f sigma=%.4f splits=%s"%(k,sp_res[k]['mean'],sp_res[k]['std'],json.dumps([round(v,3) for v in sp_res[k]['split_means']])))
    json.dump(sp_res,open(OUT/"sp_results.json","w"),indent=2)

log("\n"+"="*60)
log("SUMMARY:")
for k in ['zii','l33','l12']:
    log("  SP(%s) %.4f | eff90 %d PR %.1f"%(k,sp_res[k]['mean'],diag[f"effrank_{k}"]['eff90'],diag[f"effrank_{k}"]['pr']))
log("  CKA l33-zii %.4f / l12-zii %.4f / l33-l12 %.4f"%(diag['cka_l33_zii'],diag['cka_l12_zii'],diag['cka_l33_l12']))
log("  ridge R2 l33->zii %.4f / l12->zii %.4f"%(diag['ridge_l33_to_zii'],diag['ridge_l12_to_zii']))
log("DONE");LOG.close()
