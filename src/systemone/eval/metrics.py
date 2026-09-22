"""Metrics. nDCG with the TREC convention, ECE per bucket, paired bootstrap."""
from __future__ import annotations

import numpy as np

__all__ = ["ndcg_at_k", "recall_at_k", "mrr_at_k", "ece", "paired_bootstrap"]


def _dcg(rel, k):
    rel = np.asarray(rel, dtype=float)[:k]
    return float(np.sum((2.0**rel - 1.0) / np.log2(np.arange(2, rel.size + 2))))


def ndcg_at_k(ranked_rel, all_rel, k=10):
    idcg = _dcg(np.sort(np.asarray(all_rel, dtype=float))[::-1], k)
    return _dcg(ranked_rel, k) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_rel, all_rel, k=10, threshold=1):
    """threshold=2 matches `trec_eval -l 2`: DL grade 1 is NOT relevant."""
    tot = int(np.sum(np.asarray(all_rel) >= threshold))
    if tot == 0:
        return 0.0
    return float(np.sum(np.asarray(ranked_rel)[:k] >= threshold)) / tot


def mrr_at_k(ranked_rel, k=10, threshold=2):
    hit = np.flatnonzero(np.asarray(ranked_rel)[:k] >= threshold)
    return 1.0 / (hit[0] + 1) if hit.size else 0.0


def ece(probs, correct, n_bins=15):
    """Expected calibration error. Report it PER option-count bucket.

    Laya's own eval shows an aggregate ECE of 0.030 hiding a family at 0.438.
    An aggregate ECE is not evidence of calibration.
    """
    p = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(correct, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def paired_bootstrap(a, b, n_resamples=10000, seed=0):
    """Paired bootstrap over queries. Returns (mean diff, lo, hi, p_two_sided)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired bootstrap needs aligned per-query scores")
    d = a - b
    n = d.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))
