#!/usr/bin/env python3
"""E77: per-layer ESM<->Z_II redundancy curves (2026-08-04, CPU-only).

User hypothesis from the TIM marginal curve: for fitness tasks, mid-layer ESM
representations overlap heavily with structural information (marginal ~0 =
redundancy), while the last layer deviates from pure structure (MLM
specialization) so Z_II becomes valuable again. Testable prediction: the
ESM<->Z_II redundancy curve peaks at mid layers (TIM: ~L13-15 where the
XA-SP marginal hits zero) and drops at L33.
Measures per layer k=1..33, per protein:
  - linear CKA(ESM_k, Z_II) on subsampled (mutant,position) vectors
  - ridge R2(ESM_k -> Z_II) (directional reconstructability)
E74's RBD single-point precedent: CKA l12-zii 0.077 > l33-zii 0.026.
"""
import numpy as np,json,math,warnings,csv,re
import torch
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT=Path("/mnt/k/output_heads")
OUT=ROOT/"e77_layer_redundancy";OUT.mkdir(exist_ok=True)
LOG=open(OUT/"log.txt","w",1)
def log(m):print(m,flush=True);LOG.write(m+"\n")

def cka_lin(X,Y):
    X=X-X.mean(0);Y=Y-Y.mean(0)
    return float((np.linalg.norm(Y.T@X,'fro')**2)/
                 (np.linalg.norm(X.T@X,'fro')*np.linalg.norm(Y.T@Y,'fro')+1e-12))

def ridge_r2(X,Y,lam=1e2):
    ntr=int(0.8*len(X))
    Xtr,Xte=X[:ntr],X[ntr:];Ytr,Yte=Y[:ntr],Y[ntr:]
    mx,my=Xtr.mean(0),Ytr.mean(0)
    Xtr,Xte=Xtr-mx,Xte-mx;Ytr,Yte=Ytr-my,Yte-my
    W=np.linalg.solve(Xtr.T@Xtr+lam*np.eye(Xtr.shape[1]),Xtr.T@Ytr)
    P=Xte@W
    return float(1-np.sum((P-Yte)**2)/np.sum(Yte**2))

def load_prot(prot):
    """returns list of (esm_all33_path, z_vector) per mutant + L."""
    if prot=='rbd':
        META=[json.loads(l) for l in open(ROOT/"rbd/metadata.jsonl")];L=201
        rows=[(ROOT/"rbd/faesm_all33"/f"{m['mutant_id']}.pt",
               torch.load(ROOT/"rbd/zii"/f"{m['mutant_id']}_zii.pt",map_location="cpu",weights_only=True)[0,:L,:].float().numpy())
              for m in META]
    elif prot=='tim':
        META=[json.loads(l) for l in open(ROOT/"tim/metadata.jsonl")]
        seen=set();rows=[]
        for m in META:
            if m['mutant_id'] in seen:continue
            seen.add(m['mutant_id'])
            rows.append((ROOT/"tim/faesm_all33"/f"{m['mutant_id']}.pt",
                         torch.load(ROOT/"tim/zii"/f"{m['mutant_id']}_zii.pt",map_location="cpu",weights_only=True)[0,:254,:].float().numpy()))
        L=254
    elif prot=='ci':
        rowscsv=list(csv.DictReader(open('/mnt/j/conda_envs/foundry/DMS_Project/data/DMS_ProteinGym_substitutions/RPC1_LAMBD_Li_2019_high-expression.csv')))
        rows=[(ROOT/"ci/faesm_all33"/f"ci_{i}.pt",
               torch.load(ROOT/"ci/zii"/f"ci_{i}.pt",map_location="cpu",weights_only=True).float().numpy())
              for i in range(len(rowscsv))]
        L=237
    elif prot=='casp3':
        META=[json.loads(l) for l in open(ROOT/"casp3/manifest.jsonl") if l.strip()]
        META=[m for m in META if m['position']<=244];L=244
        rows=[]
        for m in META:
            z=torch.load(ROOT/"casp3/zii"/f"{m['mutant_id']}_zii.pt",map_location="cpu")[m['position']-1,:2*L,:].float().numpy()
            rows.append((ROOT/"casp3/faesm_all33"/f"{m['mutant_id']}.pt",z[:L,:]))  # first monomer
    return rows,L

RES={}
if (OUT/"results.json").exists():
    RES=json.load(open(OUT/"results.json"))
for prot in ['tim','ci','rbd','casp3']:
    if prot in RES:
        log("%s cached, skip"%prot);continue
    log("="*60);log("PROTEIN %s"%prot)
    rows,L=load_prot(prot)
    rng=np.random.RandomState(0)
    mus=rng.choice(len(rows),min(300,len(rows)),replace=False)
    # build aligned vector stacks: subsample (mutant, position) pairs
    Es={k:[] for k in range(1,34)};Zs=[]
    for i in mus:
        p,t=rows[i]
        e=torch.load(p,map_location="cpu",weights_only=True).float().numpy()  # [33,L,1280]
        npos=min(L,e.shape[1],t.shape[0])
        pos=rng.choice(npos,20,replace=False)
        for k in range(1,34):Es[k].append(e[k-1,pos].reshape(-1,1280))
        Zs.append(t[pos].reshape(-1,128))
    Z=np.vstack(Zs)
    rec={}
    for k in range(1,34):
        E=np.vstack(Es[k])
        c=cka_lin(E,Z);r2=ridge_r2(E,Z)
        rec[f"L{k}"]=dict(cka=c,r2=r2)
        if k%4==0 or k in (1,12,13,14,15,33):
            log("  L%-3d CKA=%.4f R2=%.4f"%(k,c,r2))
    RES[prot]=rec
    json.dump(RES,open(OUT/"results.json","w"),indent=2)

log("\n"+"="*60)
for prot,rec in RES.items():
    ks=range(1,34)
    cka=[rec[f"L{k}"]['cka'] for k in ks]
    pk=ks[int(np.argmax(cka))]
    log("%-6s CKA peak: L%d (%.4f) | L33=%.4f | L1=%.4f"%(prot,pk,max(cka),cka[-1],cka[0]))
log("DONE");LOG.close()
