"""The baseline row. Does Laya earn its place against what you could use today?

plan.md names the uncomfortable comparison itself: a 6-layer MiniLM
cross-encoder, available this afternoon, at 1,800 documents/second. This
measures it on an identical pool rather than citing it, alongside
Qwen3-Reranker-0.6B and Laya before and after fine-tuning.

Held constant per Part X: same corpus, same BM25 first stage, same top-100
pool, same queries, same truncation. Without that you are measuring the
retriever.

Two axes, and the second decides the project:

  QUALITY      nDCG@10 / Recall@10 / MRR@10, with paired bootstrap CIs.
  CALIBRATION  ECE and Brier on P(relevant), reported BOTH as shipped and
               after a one-parameter temperature fitted on a held-out half.
               The second is the comparison anyone would actually deploy; the
               first largely measures whether a model's training corpus had a
               similar positive rate.

Laya's P(relevant) comes from the CORAL ordinal head added in Phase 1. The base
checkpoint has none, so its calibration row is absent by construction -- and
that absence is itself the argument for adding one.

Run systems separately with --only. Loading Laya and Qwen together needs ~3 GB
of unified memory and drove this machine to 93% swap, which does not fail, it
just becomes unboundedly slow.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from systemone.eval import ece, mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k
from systemone.reranking.anchor_pool import AnchorPool
from systemone.reranking.data import BM25Index, load_beir, make_signature_builder
from systemone.reranking.packing import StatePacker
from systemone.reranking.scoring import AnchoredScorer
from systemone.utils.runner import Run


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def fit_binary_temperature(p, y, lo=0.05, hi=20.0, iters=60):
    """Temperature on the logit of a binary probability, by golden section.

    A reranker's P(relevant) is a squashed score, so the like-for-like
    calibration comparison is "after the one-parameter fix anyone would apply",
    not "as shipped".
    """
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    z = np.log(p / (1 - p))

    def nll(t):
        q = np.clip(1.0 / (1.0 + np.exp(-z / max(t, 1e-6))), 1e-9, 1 - 1e-9)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(d)
    return (a + b) / 2.0


def apply_temperature(p, t):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / max(t, 1e-6)))


def build_pools(ds, index, n_queries, depth, k, relevant_at, doc_chars):
    pools = []
    for qid in ds.query_ids[:n_queries]:
        query = ds.queries[qid]
        ranked, _ = index.search(query, depth)
        if len(ranked) < k:
            continue
        grades = np.array([ds.grade(qid, d) for d in ranked], dtype=float)
        if grades.max() < relevant_at:
            continue                      # nothing to find; not a ranking task
        pools.append({"qid": qid, "query": query, "ids": ranked, "grades": grades,
                      "docs": [ds.docs[d].full[:doc_chars] for d in ranked]})
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="trec-covid")
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--relevant-at", type=int, default=1)
    ap.add_argument("--only", default="minilm,laya",
                    help="comma-separated: minilm, laya, qwen. Run qwen alone.")
    ap.add_argument("--laya", default="data/cache/laya")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--minilm", default="data/cache/minilm-l6")
    ap.add_argument("--qwen", default="data/cache/qwen3-rerank-0.6b")
    ap.add_argument("--doc-chars", type=int, default=2000)
    ap.add_argument("--save-probs", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    want = {w.strip() for w in args.only.split(",") if w.strip()}

    with Run("baselines", vars(args), tag=f"{args.dataset}-{'-'.join(sorted(want))}") as run:
        ds = load_beir(args.dataset)
        doc_ids = list(ds.docs)
        index = BM25Index.build(doc_ids, (ds.docs[d].full for d in doc_ids))
        run.section("Setup")
        run.log(f"{ds}\n{index}")
        pools = build_pools(ds, index, args.queries, args.depth, args.k,
                            args.relevant_at, args.doc_chars)
        run.log(f"{len(pools)} queries, depth {args.depth}, "
                f"relevant = grade >= {args.relevant_at}, "
                f"{args.doc_chars} chars per doc, identical for every system")
        run.log(f"systems: {sorted(want)}")

        truth = np.concatenate(
            [(p["grades"] >= args.relevant_at).astype(float) for p in pools])
        # split by QUERY so a fitted temperature never sees its own evaluation
        nq = len(pools)
        first_half = np.concatenate(
            [np.full(len(p["grades"]), i < nq // 2) for i, p in enumerate(pools)])

        results, per_query, saved = {}, {}, {}

        def record(name, orders, probs=None, params_m=None, ms=None, note=""):
            nd, rc, mr = [], [], []
            for p, order in zip(pools, orders):
                g = p["grades"][order]
                nd.append(ndcg_at_k(g, p["grades"], args.k))
                rc.append(recall_at_k(g, p["grades"], args.k, args.relevant_at))
                mr.append(mrr_at_k(g, args.k, args.relevant_at))
            row = {"ndcg10": float(np.mean(nd)), "recall10": float(np.mean(rc)),
                   "mrr10": float(np.mean(mr)), "params_M": params_m,
                   "ms_per_query": ms, "note": note}
            if probs is not None:
                allp = np.concatenate(probs)
                t = fit_binary_temperature(allp[first_half], truth[first_half])
                cal = apply_temperature(allp[~first_half], t)
                row.update(
                    ece_raw=round(float(ece(allp[~first_half], truth[~first_half])), 4),
                    ece_cal=round(float(ece(cal, truth[~first_half])), 4),
                    brier_cal=round(brier(cal, truth[~first_half]), 4),
                    temperature=round(float(t), 3),
                    mean_p=round(float(allp.mean()), 4))
                saved[name] = allp
            results[name] = row
            per_query[name] = np.asarray(nd)
            run.log(f"  {name:<26} nDCG@10={row['ndcg10']:.4f}"
                    + (f"  ECE={row['ece_raw']:.4f}->{row['ece_cal']:.4f}"
                       if probs is not None else "  ECE=n/a")
                    + (f"  {ms:.0f} ms/q" if ms else ""))

        run.section("Systems")
        record("BM25 (first stage)", [np.arange(len(p["ids"])) for p in pools],
               params_m=0.0, ms=0.0, note="the floor")

        if "minilm" in want:
            from systemone.reranking.baselines import CrossEncoderReranker
            ce = CrossEncoderReranker(args.minilm, batch_size=64)
            t0 = time.time()
            orders, probs = [], []
            for p in pools:
                sc = ce.score(p["query"], p["docs"])
                orders.append(np.argsort(-sc))
                probs.append(1.0 / (1.0 + np.exp(-sc)))
            record("MiniLM-L6 cross-encoder", orders, probs, 22.7,
                   (time.time() - t0) * 1e3 / len(pools), "100 passes/query")
            run.metric("minilm_throughput", ce.throughput())
            del ce

        if "laya" in want:
            from systemone.model import LayaRuntime
            from systemone.reranking.scorer import LayaScorer
            rt = LayaRuntime(args.laya, apply_temperature=False)
            packer = StatePacker(token_counter=None)
            # built ONCE: inlining this as a lambda rebuilt a 207k-entry dict
            # on every term lookup and dominated the run
            idf = {t: float(index.idf[j]) for t, j in index.vocab.items()}
            sig = make_signature_builder("title+lexical", counter=None,
                                         idf_lookup=idf.get)
            anchor_pool = AnchorPool(strategy="templated")
            sig_cache = {}

            def run_laya(scorer, n_anchors, probs_fn=None):
                t0 = time.time()
                orders, pr = [], []
                for p in pools:
                    anchors = anchor_pool.for_query(p["query"], ds, p["qid"], n_anchors)
                    if p["qid"] not in sig_cache:
                        sig_cache[p["qid"]] = [
                            sig.build(p["query"], ds.docs[d].title, ds.docs[d].text, 37)
                            for d in p["ids"]]
                    sigs = sig_cache[p["qid"]]
                    pool = AnchoredScorer(scorer, anchors, packer=packer,
                                          max_options=10, n_anchors=n_anchors
                                          ).score(p["query"], sigs,
                                                  rng=np.random.default_rng(args.seed))
                    orders.append(pool.order)
                    if probs_fn is not None:
                        pr.append(probs_fn(p, sigs))
                return orders, (pr if probs_fn else None), \
                    (time.time() - t0) * 1e3 / len(pools)

            o, _, ms = run_laya(LayaScorer(rt, max_batch_slates=16), 4)
            record("Laya base (anchored)", o, None, 421.3, ms,
                   "no ordinal head -> no P(relevant) at all")

            if args.checkpoint:
                from systemone.model.heads import FrontierRankModel, coral_probs
                from systemone.model.sequence import QTYPES
                from systemone.train.trainer_ranking import TrainedScorer
                model = FrontierRankModel(rt.model, n_levels=4)
                model.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                                 weights_only=False)["model"],
                                      strict=True)
                model.to(rt.device).eval()

                @torch.no_grad()
                def laya_probs(p, sigs):
                    """P(grade >= relevant_at) from the CORAL head.

                    The ordinal head is pointwise -- it reads one marker's
                    hidden state and never sees the slate softmax -- so
                    candidates pack 9 to a slate instead of one at a time.
                    11x fewer forward passes, identical answer.
                    """
                    out = np.zeros(len(sigs))
                    groups = [list(range(i, min(i + 9, len(sigs))))
                              for i in range(0, len(sigs), 9)]
                    for gi in range(0, len(groups), 4):
                        batch = groups[gi:gi + 4]
                        built = [rt.build(x.state, "choice", x.instructions,
                                          list(x.options), rendered=True)
                                 for x in (packer.pack(p["query"],
                                                       [sigs[j] for j in g])
                                           for g in batch)]
                        L = max(len(b[0]) for b in built)
                        K = max(len(b[1]) for b in built)
                        ids = torch.full((len(built), L), rt.tok.pad_token_id,
                                         dtype=torch.long)
                        att = torch.zeros((len(built), L), dtype=torch.long)
                        mp = torch.zeros((len(built), K), dtype=torch.long)
                        mm = torch.zeros((len(built), K), dtype=torch.bool)
                        for r, (seq, mk, _) in enumerate(built):
                            ids[r, :len(seq)] = torch.tensor(seq)
                            att[r, :len(seq)] = 1
                            mp[r, :len(mk)] = torch.tensor(mk)
                            mm[r, :len(mk)] = True
                        out_t = model(ids.to(rt.device), att.to(rt.device),
                                      mp.to(rt.device), mm.to(rt.device),
                                      torch.full((len(built),),
                                                 QTYPES["choice"]).to(rt.device))
                        lp = coral_probs(out_t["ordinal_logits"]).float().cpu().numpy()
                        for r, g in enumerate(batch):
                            for c, j in enumerate(g):
                                out[j] = lp[r, c, args.relevant_at:].sum()
                    return out

                o, pr, ms = run_laya(TrainedScorer(model, rt, max_batch_slates=16),
                                     4, laya_probs)
                record("Laya tuned (anchored)", o, pr, 422.3, ms,
                       "CORAL head gives a real P(relevant)")

        if "qwen" in want:
            from systemone.reranking.baselines import QwenReranker
            qw = QwenReranker(args.qwen, batch_size=16)
            t0 = time.time()
            orders, probs = [], []
            for p in pools:
                pp = qw.probability(p["query"], p["docs"])
                orders.append(np.argsort(-pp))
                probs.append(pp)
            record("Qwen3-Reranker-0.6B", orders, probs, 596.0,
                   (time.time() - t0) * 1e3 / len(pools), "first-token yes/no, fp16")
            run.metric("qwen_throughput", qw.throughput())

        run.section("Quality and calibration, identical pool")
        run.log(f"{'system':<26} {'params':>8} {'nDCG@10':>9} {'R@10':>7} "
                f"{'MRR@10':>7} {'ECE raw':>8} {'ECE cal':>8} {'T':>6} "
                f"{'Brier':>7} {'ms/q':>7}")
        run.log("-" * 104)
        nan = float("nan")
        for name, r in sorted(results.items(), key=lambda kv: -kv[1]["ndcg10"]):
            run.log(f"{name:<26} {(r['params_M'] or 0):>8.1f} {r['ndcg10']:>9.4f} "
                    f"{r['recall10']:>7.4f} {r['mrr10']:>7.4f} "
                    f"{r.get('ece_raw', nan):>8.4f} {r.get('ece_cal', nan):>8.4f} "
                    f"{r.get('temperature', nan):>6.2f} "
                    f"{r.get('brier_cal', nan):>7.4f} {(r['ms_per_query'] or 0):>7.0f}")
        run.log("")
        run.log("ECE raw = as shipped, on a held-out half of the queries.")
        run.log("ECE cal = after a one-parameter temperature fitted on the other half.")
        run.log("The second is what you would deploy; the first largely measures")
        run.log("whether a model's training corpus had a similar positive rate.")
        run.metric("results", results)

        run.section("Paired bootstrap vs BM25, 10k resamples")
        base = per_query["BM25 (first stage)"]
        run.log(f"{'system':<26} {'delta nDCG@10':>14} {'95% CI':>22} {'p':>8}")
        run.log("-" * 74)
        stats = {}
        for name, arr in sorted(per_query.items(), key=lambda kv: -kv[1].mean()):
            if name == "BM25 (first stage)":
                continue
            d, lo, hi, p = paired_bootstrap(arr, base, seed=1)
            stats[name] = {"delta": d, "ci": [lo, hi], "p": p}
            run.log(f"{name:<26} {d:>+14.4f} [{lo:>+8.4f},{hi:>+8.4f}] {p:>8.4f}")
        run.metric("vs_bm25", stats)

        if args.save_probs and saved:
            np.savez_compressed(args.save_probs, truth=truth, **saved)
            run.log(f"\n  probabilities saved to {args.save_probs}")


if __name__ == "__main__":
    main()
