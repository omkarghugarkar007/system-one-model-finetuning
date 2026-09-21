"""The real engineering tradeoff.

Laya's shipped temperature map has choice:11+ -> T=0.1006 (a 10x sharpening),
which only fits if the raw logits are near-uniform past ~10 options. So treat
M=10 option slots as a hard cap. Anchors occupy slots that candidates could
have used:  real candidates per slate = M - A,  slates per query = ceil(N/(M-A)).

Question: what (M, A) maximises nDCG@10 per forward pass?
"""
import numpy as np, math
RNG = np.random.default_rng(20260921)

def ndcg_at_k(rs, ideal, k=10):
    d=lambda r: np.sum((2.0**np.asarray(r[:k],float)-1)/np.log2(np.arange(2,len(r[:k])+2)))
    i=d(np.sort(ideal)[::-1]); return d(rs)/i if i>0 else 0.0
def ndcg(s, rel, rng, k=10):
    return ndcg_at_k(np.asarray(rel)[np.argsort(-(np.asarray(s,float)+rng.random(len(s))*1e-9))], rel, k)

def run(n=100, M=10, A=4, noise=0.35, drift=0.15, lam=1.0, rng=RNG):
    u=rng.normal(0,1.5,n); rel=np.digitize(u,np.quantile(u,[.70,.90,.97]))
    per = max(M-A,1); anchors=np.linspace(-2.2,2.2,A) if A>1 else np.array([0.0])
    idx=np.argsort(-u)                      # blocked: slated by first-stage rank
    naive=np.full(n,-np.inf); ridge=np.full(n,-np.inf)
    passes=0
    for s in range(0,n,per):
        S=idx[s:s+per]; passes+=1
        l=np.concatenate([u[S],anchors])*(1.0+rng.normal(0,drift))
        z=l+rng.normal(0,noise,len(l)); lp=z-(z.max()+np.log(np.exp(z-z.max()).sum()))
        lr,la=lp[:len(S)],lp[len(S):]
        naive[S]=lr
        if A>=2:
            num=np.sum((la-la.mean())*(anchors-anchors.mean())); den=np.sum((la-la.mean())**2)
            a=(num+lam)/(den+lam); ridge[S]=a*lr+np.mean(anchors-a*la)
        else:
            ridge[S]=lr+np.mean(anchors-la)
    return ndcg(naive,rel,rng), ndcg(ridge,rel,rng), passes

def sweep(T=5000,**kw):
    nn=[];rr=[];pp=0
    for _ in range(T):
        a,b,p=run(rng=RNG,**kw); nn.append(a);rr.append(b);pp=p
    return float(np.mean(nn)),float(np.mean(rr)),pp

print("N=100 candidates, blocked slates, 5000 queries each\n")
print(f"{'M':>3} {'A':>3} {'cands/slate':>12} {'passes/query':>13} {'naive':>9} {'anchored':>10} {'gain':>8} {'nDCG/pass':>11}")
print("-"*76)
rows=[]
for M in (10,):
    for A in (0,1,2,3,4,5,6):
        nv,an,p = sweep(T=5000, M=M, A=max(A,1))
        if A==0: an=nv  # A=0 means no anchors at all
        rows.append((M,A,M-max(A,1) if A else M,p if A else math.ceil(100/M),nv,an))
        print(f"{M:>3} {A:>3} {M-A if A else M:>12} {(math.ceil(100/(M-A)) if A else math.ceil(100/M)):>13} {nv:>9.4f} {an:>10.4f} {an-nv:>+8.4f} {an/(math.ceil(100/(M-A)) if A else math.ceil(100/M)):>11.4f}")

print("\n\nWhat a big-slate model (Jev, M=255) gets for free — one slate, no cross-slate problem:\n")
for M in (30,50,100):
    nv,an,p=sweep(T=5000,M=M,A=1)
    print(f"  M={M:>3}: one slate covers {min(M-1,100)} candidates -> passes/query={math.ceil(100/(M-1))}, nDCG@10={an:.4f}")

print("\n\nAnchor slots are the cheapest quality you can buy at M=10:")
nv4,an4,_=sweep(T=5000,M=10,A=4)
nv0,an0,_=sweep(T=5000,M=10,A=1)
print(f"  A=1 (shift only): {an0:.4f} at {math.ceil(100/9)} passes")
print(f"  A=4 (ridge):      {an4:.4f} at {math.ceil(100/6)} passes  -> +{an4-an0:.4f} nDCG for +{math.ceil(100/6)-math.ceil(100/9)} passes")
