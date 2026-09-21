"""Sanity-check BM25 against published BEIR numbers before trusting any delta.

The first stage moves the metric more than the reranker does (plan Part II:
RankZephyr swings 10.7 nDCG@10 on DL20 purely by changing what it reranks). A
reranking gain measured on top of a broken first stage is not a gain, so the
floor gets checked against the literature before anything is built on it.

Published BEIR BM25 nDCG@10 (Thakur et al., Anserini defaults):
    trec-covid 0.656 | nfcorpus 0.325 | scifact 0.665
"""
import argparse

import numpy as np

from frontierrank.data import BM25Index, load_beir
from frontierrank.eval import ndcg_at_k

PUBLISHED = {"trec-covid": 0.656, "nfcorpus": 0.325, "scifact": 0.665}

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=["nfcorpus", "trec-covid", "scifact"])
ap.add_argument("--depth", type=int, default=100)
args = ap.parse_args()

print(f"{'dataset':>12} {'queries':>8} {'nDCG@10':>9} {'published':>10} {'delta':>8}")
print("-" * 52)
for name in args.datasets:
    ds = load_beir(name)
    ids = list(ds.docs)
    index = BM25Index.build(ids, (ds.docs[d].full for d in ids))
    scores = []
    for qid in ds.query_ids:
        ranked, _ = index.search(ds.queries[qid], args.depth)
        # nDCG@10 normalised by the ideal over ALL judged documents for the
        # query, not just the retrieved pool -- otherwise a retriever that
        # misses every relevant document still scores 1.0
        pool = np.array([ds.grade(qid, d) for d in ranked], dtype=float)
        all_rel = np.array(sorted(ds.qrels.get(qid, {}).values(), reverse=True),
                           dtype=float)
        scores.append(ndcg_at_k(pool, all_rel, 10))
    got = float(np.mean(scores))
    pub = PUBLISHED.get(name)
    d = f"{got - pub:+.3f}" if pub else "  --"
    print(f"{name:>12} {len(scores):>8} {got:>9.4f} "
          f"{pub if pub else float('nan'):>10.3f} {d:>8}")
print("\nWithin ~0.05 of published is a healthy first stage. A large shortfall")
print("means the tokenizer, the k1/b defaults or the title+text concatenation")
print("differ from Anserini, and every reranking delta measured on it is suspect.")
