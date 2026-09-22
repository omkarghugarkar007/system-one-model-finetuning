"""Are the templated pivots actually pivots?

`training.anchors` assumes a direct-answer template is high-utility for any
query and a fixed off-topic passage is low-utility for any query. That is an
assumption, and it is cheap to falsify: put all four templates in one slate
with real candidates and check the model orders them as designed.

Three things have to hold for templated anchors to be usable:

  1. ORDER      the templates rank off_topic < topical < partial < direct.
  2. STABILITY  their log-probabilities move consistently across queries --
                a pivot whose score wanders per query is not a fixed reference.
  3. SPAN       real candidates fall *inside* the pivot range rather than
                piling up outside it, or the affine fit is extrapolating.

If (1) fails the templates are not pivots and `AnchorPool(strategy="judged")`
or teacher-graded pivots are the fallback.
"""
from __future__ import annotations

import argparse

import numpy as np

from systemone.model import LayaRuntime
from systemone.reranking.anchor_pool import DEFAULT_TEMPLATES, templated_anchors
from systemone.reranking.data import BM25Index, load_beir, make_signature_builder
from systemone.reranking.packing import OptionsPacker, StatePacker
from systemone.reranking.scorer import LayaScorer
from systemone.utils.runner import Run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="trec-covid")
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--candidates-per-slate", type=int, default=6)
    ap.add_argument("--slates-per-query", type=int, default=6)
    ap.add_argument("--layout", default="state", choices=["options", "state"])
    ap.add_argument("--signature", default="title+lexical")
    ap.add_argument("--model", default="data/cache/laya")
    ap.add_argument("--checkpoint", default="", help="a fine-tuned FrontierRankModel")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with Run("anchor-probe", vars(args), tag=args.dataset) as run:
        rng = np.random.default_rng(args.seed)
        ds = load_beir(args.dataset)
        doc_ids = list(ds.docs)
        index = BM25Index.build(doc_ids, (ds.docs[d].full for d in doc_ids))
        rt = LayaRuntime(args.model, apply_temperature=False)
        if args.checkpoint:
            import torch

            from systemone.model.heads import FrontierRankModel
            from systemone.train.trainer_ranking import TrainedScorer
            model = FrontierRankModel(rt.model, n_levels=4)
            sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(sd["model"], strict=True)
            model.to(rt.device).eval()
            scorer = TrainedScorer(model, rt, max_batch_slates=16)
            run.log(f"loaded fine-tuned checkpoint: {args.checkpoint}")
        else:
            scorer = LayaScorer(rt, max_batch_slates=16)
        counter = scorer.token_counter()
        idf = {t: index.idf[j] for t, j in index.vocab.items()}
        packer = (OptionsPacker(token_counter=counter) if args.layout == "options"
                  else StatePacker(token_counter=counter))
        n_opts = args.candidates_per_slate + len(DEFAULT_TEMPLATES)
        info = packer.pack("q", ["x"] * n_opts).info
        per_cand = info.get("option_tokens", info.get("per_candidate_tokens"))
        sig = make_signature_builder(args.signature, counter=counter,
                                     idf_lookup=lambda t: idf.get(t, 0.0))
        run.section("Setup")
        run.log(f"{ds}\nM={n_opts} ({args.candidates_per_slate} candidates + "
                f"{len(DEFAULT_TEMPLATES)} pivots), {per_cand} tokens/candidate, "
                f"layout={args.layout}")

        names = [t.name for t in DEFAULT_TEMPLATES]
        anchor_lp: list[np.ndarray] = []
        cand_lp: list[np.ndarray] = []
        inside = []

        for qid in ds.query_ids[: args.queries]:
            query = ds.queries[qid]
            ranked, _ = index.search(query, args.depth)
            anchors = templated_anchors(query)
            for _ in range(args.slates_per_query):
                picks = rng.choice(len(ranked), args.candidates_per_slate,
                                   replace=False)
                cands = [sig.build(query, ds.docs[ranked[i]].title,
                                   ds.docs[ranked[i]].text, per_cand) for i in picks]
                texts = cands + [a.text for a in anchors]
                perm = rng.permutation(len(texts))
                lp = np.asarray(scorer.score_many(
                    [packer.pack(query, [texts[i] for i in perm])])[0])
                inv = np.empty_like(perm)
                inv[perm] = np.arange(len(perm))
                lp = lp[inv]
                cand_lp.append(lp[: len(cands)])
                anchor_lp.append(lp[len(cands):])
                lo, hi = lp[len(cands):].min(), lp[len(cands):].max()
                inside.append(float(np.mean((lp[:len(cands)] >= lo)
                                            & (lp[:len(cands)] <= hi))))

        A = np.stack(anchor_lp)          # (n_slates, 4)
        run.section("1. ORDER -- do the pivots rank as designed?")
        run.log(f"{'pivot':>20} {'designed u':>11} {'mean log p':>11} {'std':>7} "
                f"{'mean rank':>10}")
        run.log("-" * 64)
        ranks = np.argsort(np.argsort(-A, axis=1), axis=1)
        for j, t in enumerate(DEFAULT_TEMPLATES):
            run.log(f"{t.name:>20} {t.utility:>+11.1f} {A[:, j].mean():>11.4f} "
                    f"{A[:, j].std():>7.4f} {ranks[:, j].mean() + 1:>10.2f}")
        designed = np.array([t.utility for t in DEFAULT_TEMPLATES])
        per_slate_rho = np.array([
            np.corrcoef(np.argsort(np.argsort(row)),
                        np.argsort(np.argsort(designed)))[0, 1] for row in A])
        exact = float(np.mean([list(np.argsort(row)) == list(np.argsort(designed))
                               for row in A]))
        run.log("")
        run.log(f"  rank correlation with the designed order : "
                f"{per_slate_rho.mean():.4f}")
        run.log(f"  slates with the exact designed order     : {exact:.1%}")

        run.section("2. STABILITY -- is a pivot a fixed reference?")
        run.log(f"  pivot log p std across slates : {A.std(axis=0).mean():.4f} nats")
        run.log(f"  pivot spread within a slate   : "
                f"{np.mean(A.max(1) - A.min(1)):.4f} nats")

        C = np.concatenate(cand_lp)
        run.section("3. SPAN -- do real candidates fall inside the pivot range?")
        run.log(f"  candidates inside the pivot range : {np.mean(inside):.1%}")
        run.log(f"  candidate log p range  : [{C.min():.3f}, {C.max():.3f}]")
        run.log(f"  pivot log p range      : [{A.min():.3f}, {A.max():.3f}]")

        for k, v in [("rank_corr", float(per_slate_rho.mean())),
                     ("exact_order_frac", exact),
                     ("pivot_std_across_slates", float(A.std(axis=0).mean())),
                     ("pivot_spread_within_slate", float(np.mean(A.max(1) - A.min(1)))),
                     ("candidates_inside_range", float(np.mean(inside))),
                     ("n_slates", int(A.shape[0]))]:
            run.metric(k, v)

        run.section("VERDICT")
        if per_slate_rho.mean() > 0.6:
            run.log("  USABLE: the model orders the templates broadly as designed,")
            run.log("  so they carry known utility and can serve as pivots.")
            v = "usable"
        elif per_slate_rho.mean() > 0.3:
            run.log("  WEAK: the order is right on average but noisy per slate.")
            run.log("  Usable with more pivots; re-check after fine-tuning.")
            v = "weak"
        else:
            run.log("  UNUSABLE: the model does not rank the templates as designed,")
            run.log("  so their utility is not known a priori. Fall back to")
            run.log("  teacher-graded pivots (continuous utilities) or judged ones.")
            v = "unusable"
        run.metric("verdict", v)
        run.log(f"\n  throughput: {rt.throughput()}")


if __name__ == "__main__":
    main()
