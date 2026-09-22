"""End-to-end smoke test on a synthetic Luce scorer. Seconds, no weights, no key.

It also encodes the corrected finding, which is NOT the one the first draft of
this project assumed:

  * Slate BLOCKED by first-stage rank  -> anchoring is essential (+0.25 nDCG@10).
  * Slate STRATIFIED (round-robin)     -> stratification alone recovers the
                                          within-query ranking for free, and
                                          anchoring is roughly neutral on nDCG.
  * Either way, ONLY anchoring produces an ABSOLUTE scale that is comparable
    across queries and across rounds -- which is what the frontier, the
    thresholds and the re-scoring after read_more actually need.

So: stratify for ranking, anchor for scale. They fix different problems.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from systemone.eval import ndcg_at_k, paired_bootstrap
from systemone.reranking import (
    Anchor,
    AnchoredScorer,
    Controller,
    compute_frontier,
    expected_regret,
)

ANCHOR_UTILS = np.linspace(-2.2, 2.2, 4)


class LuceScorer:
    """Stand-in for Laya: slate softmax with marker noise and temperature drift."""
    def __init__(self, utility, anchors, noise=0.35, drift=0.15, seed=0):
        self.u = dict(utility)
        self.u.update({a.text: a.utility for a in anchors})
        self.noise, self.drift = noise, drift
        self.rng = np.random.default_rng(seed)

    def choice_logprobs(self, instructions, options, state):
        l = np.array([self.u[o] for o in options], float)
        l = l * (1.0 + self.rng.normal(0, self.drift))
        z = l + self.rng.normal(0, self.noise, l.size)
        return z - (z.max() + np.log(np.exp(z - z.max()).sum()))


def one_query(seed, n_anchors=4, stratify=True, n=100, k=10):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1.5, n)
    rel = np.digitize(u, np.quantile(u, [.70, .90, .97]))
    sigs = [f"cand::{i}" for i in range(n)]
    anchors = [Anchor(f"anchor::{j}", v) for j, v in enumerate(ANCHOR_UTILS)]
    model = LuceScorer(dict(zip(sigs, u)), anchors, seed=seed)

    scorer = AnchoredScorer(model, anchors, max_options=10,
                            n_anchors=n_anchors, stratify=stratify)
    pool = scorer.score("the query", sigs, first_stage_order=np.argsort(-u))
    return dict(
        anchored=ndcg_at_k(rel[pool.order], rel, k),
        naive=ndcg_at_k(rel[np.argsort(-pool.raw_logp)], rel, k),
        utility=pool.utility, raw=pool.raw_logp, truth=u, rel=rel, pool=pool)


def sweep(stratify, n_anchors, trials=300):
    rows = [one_query(s, n_anchors, stratify) for s in range(trials)]
    a = np.array([r["anchored"] for r in rows])
    nv = np.array([r["naive"] for r in rows])
    util = np.concatenate([r["utility"] for r in rows])
    raw = np.concatenate([r["raw"] for r in rows])
    tru = np.concatenate([r["truth"] for r in rows])
    return dict(
        a=a, nv=nv,
        r_anchored=float(np.corrcoef(util, tru)[0, 1]),
        r_naive=float(np.corrcoef(raw, tru)[0, 1]),
        rmse=float(np.sqrt(np.mean((util - tru) ** 2))),
        rows=rows)


if __name__ == "__main__":
    print("FrontierRank smoke test -- 300 synthetic queries x 100 candidates, k=10\n")
    print("A. Within-query ranking (nDCG@10)\n")
    print(f"{'slating':>12} {'A':>2} {'naive':>8} {'anchored':>9} {'delta':>9}")
    print("-" * 44)
    res = {}
    for strat, tag in ((False, "blocked"), (True, "stratified")):
        for A in (2, 4):
            r = sweep(strat, A)
            res[(strat, A)] = r
            print(f"{tag:>12} {A:>2} {r['nv'].mean():>8.4f} "
                  f"{r['a'].mean():>9.4f} {r['a'].mean()-r['nv'].mean():>+9.4f}")

    print("\nB. Cross-query absolute scale -- correlation with true latent utility,")
    print("   pooled over all queries. This is what thresholds and the frontier need.\n")
    print(f"{'slating':>12} {'A':>2} {'naive r':>9} {'anchored r':>11} {'anchored RMSE':>14}")
    print("-" * 52)
    for (strat, A), r in res.items():
        tag = "stratified" if strat else "blocked"
        print(f"{tag:>12} {A:>2} {r['r_naive']:>9.4f} {r['r_anchored']:>11.4f} "
              f"{r['rmse']:>14.3f}")
    print("\n   (naive log-probs have no absolute scale at all -- RMSE is undefined)")

    b = res[(False, 2)]
    d, lo, hi, p = paired_bootstrap(b["a"], b["nv"], seed=1)
    print(f"\nC. Blocked slates, A=2, paired bootstrap: {d:+.4f} "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}")

    r0 = res[(True, 4)]["rows"][0]
    pool = r0["pool"]
    fr = compute_frontier(pool.utility, pool.sigma, k=10)
    reg = expected_regret(pool.utility, pool.sigma, r0["rel"], k=10)
    dec = Controller(lambda_cost=1000.0).decide(
        pool.utility, pool.sigma, r0["rel"], fr,
        pool_quality=float(np.max(pool.utility)))
    print(f"\nD. Pipeline: {pool.plan.n_passes} passes/query, "
          f"frontier {fr.size}/100, regret {reg:.4f}, action '{dec.action}'")
    print("   action values: " +
          ", ".join(f"{k}={v:+.5f}" for k, v in dec.detail.items()))

    # blocked slating: anchoring must rescue it
    assert res[(False, 2)]["a"].mean() - res[(False, 2)]["nv"].mean() > 0.15
    assert p < 0.01
    # either slating: anchoring must improve the cross-query scale
    for r in res.values():
        assert r["r_anchored"] > r["r_naive"], "anchoring must improve absolute scale"
    print("\n  PASS")
