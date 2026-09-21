"""Anchored slate calibration, done properly.

Estimators compared, per slate S:
  naive   : rank by log p_i           (offset c_S left in -> broken)
  shift   : u_hat = log p_i + beta_S  (beta_S = mean over anchors)
  affine  : u_hat = a_S*log p_i + b_S (OLS over anchors; 2 params)
  ridge   : affine with slope shrunk toward 1 (lambda), b_S from residual mean
"""
import numpy as np

RNG = np.random.default_rng(20260921)

def ndcg_at_k(rel_sorted, ideal, k=10):
    d = lambda r: np.sum((2.0**np.asarray(r[:k],float)-1)/np.log2(np.arange(2,len(r[:k])+2)))
    i = d(np.sort(ideal)[::-1]);  return d(rel_sorted)/i if i>0 else 0.0

def ndcg(scores, rel, rng, k=10):
    o = np.argsort(-(np.asarray(scores,float)+rng.random(len(scores))*1e-9))
    return ndcg_at_k(np.asarray(rel)[o], rel, k)

def run(n=100, slate=8, A=4, noise=0.35, blocked=True, drift=0.15, lam=1.0, rng=RNG, k=10):
    u = rng.normal(0,1.5,n)
    rel = np.digitize(u, np.quantile(u,[.70,.90,.97]))
    anchors = np.linspace(-2.2, 2.2, A) if A>1 else np.array([0.0])   # always well spread

    idx = np.argsort(-u) if blocked else rng.permutation(n)
    out = {kk: np.full(n,-np.inf) for kk in ("naive","shift","affine","ridge")}

    for s in range(0,n,slate):
        S = idx[s:s+slate]
        l = np.concatenate([u[S], anchors]) * (1.0 + rng.normal(0,drift))
        z = l + rng.normal(0,noise,len(l))
        lp = z - (z.max()+np.log(np.exp(z-z.max()).sum()))
        lr, la = lp[:len(S)], lp[len(S):]

        out["naive"][S] = lr
        out["shift"][S] = lr + np.mean(anchors - la)
        if A >= 2:
            X = np.vstack([la, np.ones(A)]).T
            a, b = np.linalg.lstsq(X, anchors, rcond=None)[0]
            out["affine"][S] = a*lr + b
            # ridge: shrink slope toward 1, then re-centre on anchors
            num = np.sum((la-la.mean())*(anchors-anchors.mean()))
            den = np.sum((la-la.mean())**2)
            a_r = (num + lam) / (den + lam)          # -> 1 as lam dominates
            out["ridge"][S] = a_r*lr + np.mean(anchors - a_r*la)
        else:
            out["affine"][S] = out["ridge"][S] = out["shift"][S]

    rub = np.digitize(u+rng.normal(0,noise,n), np.quantile(u,[.70,.90,.97])).astype(float)
    res = {kk: ndcg(v, rel, rng, k) for kk,v in out.items()}
    res["rubric4"]   = ndcg(rub, rel, rng, k)
    res["pointwise"] = ndcg(u+rng.normal(0,noise,n), rel, rng, k)
    return res

def sweep(T=6000, **kw):
    acc={}
    for _ in range(T):
        for kk,v in run(rng=RNG,**kw).items(): acc.setdefault(kk,[]).append(v)
    return {kk:(float(np.mean(v)), float(np.std(v)/np.sqrt(T))) for kk,v in acc.items()}

print("nDCG@10 | 100 cands/query, slate=8, 6000 queries | blocked slates (= slated by first-stage rank)\n")
print(f"{'anchors A':>10} | {'naive':>14} {'shift-only':>14} {'affine(OLS)':>14} {'ridge':>14}")
print("-"*76)
for A in (1,2,3,4,6,8):
    r = sweep(T=6000, A=A, blocked=True)
    print(f"{A:>10} | {r['naive'][0]:>8.4f}      {r['shift'][0]:>8.4f}      {r['affine'][0]:>8.4f}      {r['ridge'][0]:>8.4f}")
print(f"\n  pointwise 4-level rubric (globally comparable, coarse) : {r['rubric4'][0]:.4f}")
print(f"  pointwise continuous, same noise (upper bound)         : {r['pointwise'][0]:.4f}")
print(f"  oracle                                                 : 1.0000")

print("\n\nSlate assignment matters (A=4, ridge):\n")
for blocked,tag in ((True,"blocked (by first-stage rank)"),(False,"random (shuffled)")):
    r = sweep(T=6000, A=4, blocked=blocked)
    print(f"  {tag:<32} naive {r['naive'][0]:.4f}   ridge {r['ridge'][0]:.4f}   delta +{r['ridge'][0]-r['naive'][0]:.4f}")

print("\n\nSensitivity to slate-temperature drift (A=4, blocked):\n")
for d in (0.0,0.10,0.25,0.40):
    r = sweep(T=4000, A=4, blocked=True, drift=d)
    print(f"  drift sigma={d:.2f}   naive {r['naive'][0]:.4f}   shift {r['shift'][0]:.4f}   ridge {r['ridge'][0]:.4f}")
