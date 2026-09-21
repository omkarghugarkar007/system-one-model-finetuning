"""Is there any relevance signal in the base checkpoint, and did we starve it?

T4 failing has two very different explanations, and the whole project turns on
which one it is:

  (a) the base checkpoint has no relevance signal to calibrate -- exactly what
      Laya's own card says ("a fast base to specialise, not a zero-shot
      decision engine"), in which case Phase 0's scale test simply cannot be
      run before Phase 1; or
  (b) the signal is there and the 16-token OPTIONS budget threw it away, in
      which case the fix is the slicer and the layout, not the weights.

They are distinguished by varying the evidence budget and watching whether the
signal moves. If a 40-token STATE slate recovers correlation that a 16-token
OPTIONS slate lost, it was (b). If nothing moves, it was (a).

Reported per configuration:
  corr(l_i, grade)  the free additive fit's latent logit against the label.
                    This is the cleanest available read on "does the model rank
                    at all", with the slate offset already removed.
  nDCG@10           against BM25 on the same pool, so there is a floor to beat.
  signature recall  how much of the query survived the slicer.
"""
from __future__ import annotations

import argparse

import numpy as np

from frontierrank.core.packing import OptionsPacker, StatePacker
from frontierrank.data import BM25Index, load_beir, make_signature_builder
from frontierrank.data.signatures import signature_recall
from frontierrank.eval import ndcg_at_k
from frontierrank.experiments.phase0_identifiability import (fit_additive,
                                                             make_block_design)
from frontierrank.experiments.runner import Run
from frontierrank.models.laya import LayaRuntime, LayaScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="trec-covid")
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--slate-size", type=int, default=10)
    ap.add_argument("--slates-per-query", type=int, default=30)
    ap.add_argument("--probes-per-grade", type=int, default=12)
    ap.add_argument("--model", default="data/cache/laya")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    CONFIGS = [("options", "title"), ("options", "lexical"),
               ("options", "title+lexical"), ("state", "title"),
               ("state", "lexical"), ("state", "title+lexical")]

    with Run("phase0-signal", vars(args), tag=args.dataset) as run:
        rng = np.random.default_rng(args.seed)
        ds = load_beir(args.dataset)
        doc_ids = list(ds.docs)
        index = BM25Index.build(doc_ids, (ds.docs[d].full for d in doc_ids))
        rt = LayaRuntime(args.model, apply_temperature=False)
        scorer = LayaScorer(rt, max_batch_slates=16)
        counter = scorer.token_counter()
        idf = {t: index.idf[j] for t, j in index.vocab.items()}
        run.section("Setup")
        run.log(f"{ds}\n{index}\nLaya on {rt.device}, base checkpoint, temperature OFF")

        # one shared probe set per query so configs are exactly comparable
        prepared = []
        for qid in ds.query_ids[: args.queries]:
            query = ds.queries[qid]
            ranked, bm_scores = index.search(query, args.depth)
            by_grade: dict[int, list] = {}
            for d, sc in zip(ranked, bm_scores):
                by_grade.setdefault(ds.grade(qid, d), []).append((d, sc))
            probes, grades, bm = [], [], []
            for g in sorted(by_grade):
                for d, sc in by_grade[g][: args.probes_per_grade]:
                    probes.append(d)
                    grades.append(g)
                    bm.append(sc)
            if len(set(grades)) < 3 or len(probes) < args.slate_size + 6:
                continue
            prepared.append((qid, query, probes, np.asarray(grades, float),
                             np.asarray(bm)))
        run.log(f"{len(prepared)} usable queries")

        run.section("Signal by evidence budget")
        run.log(f"{'layout':>8} {'signature':>14} {'tok/cand':>9} {'sig recall':>11} "
                f"{'corr(l,grade)':>14} {'nDCG@10':>9} {'BM25':>8}")
        run.log("-" * 80)

        results = {}
        for layout, signame in CONFIGS:
            packer = (OptionsPacker(token_counter=counter) if layout == "options"
                      else StatePacker(token_counter=counter))
            info = packer.pack("q", ["x"] * args.slate_size).info
            per_cand = info.get("option_tokens", info.get("per_candidate_tokens"))
            sigbuild = make_signature_builder(signame, counter=counter,
                                              idf_lookup=lambda t: idf.get(t, 0.0))

            corrs, ndcgs, bm_ndcgs, recalls = [], [], [], []
            for qid, query, probes, grades, bm in prepared:
                sigs = [sigbuild.build(query, ds.docs[d].title, ds.docs[d].text,
                                       per_cand) for d in probes]
                recalls += [signature_recall(query, s, ds.docs[d].full)
                            for s, d in zip(sigs, probes)]
                design = make_block_design(grades, args.slate_size,
                                           args.slates_per_query, rng)
                fit = fit_additive(design, scorer.score_many(
                    [packer.pack(query, [sigs[i] for i in s]) for s in design.slates]))
                if np.std(fit.l_hat) > 1e-9:
                    corrs.append(float(np.corrcoef(fit.l_hat, grades)[0, 1]))
                ndcgs.append(ndcg_at_k(grades[np.argsort(-fit.l_hat)], grades, 10))
                bm_ndcgs.append(ndcg_at_k(grades[np.argsort(-bm)], grades, 10))

            r = {"tokens_per_candidate": per_cand,
                 "signature_recall": float(np.mean(recalls)),
                 "corr_latent_grade": float(np.mean(corrs)),
                 "ndcg10": float(np.mean(ndcgs)),
                 "ndcg10_bm25": float(np.mean(bm_ndcgs))}
            results[f"{layout}/{signame}"] = r
            run.log(f"{layout:>8} {signame:>14} {per_cand:>9} "
                    f"{r['signature_recall']:>11.3f} {r['corr_latent_grade']:>14.4f} "
                    f"{r['ndcg10']:>9.4f} {r['ndcg10_bm25']:>8.4f}")

        run.metric("configs", results)
        best = max(results, key=lambda k: results[k]["corr_latent_grade"])
        worst = min(results, key=lambda k: results[k]["corr_latent_grade"])
        spread = (results[best]["corr_latent_grade"]
                  - results[worst]["corr_latent_grade"])

        run.section("Read")
        run.log(f"  best  {best:<22} corr={results[best]['corr_latent_grade']:.4f} "
                f"@ {results[best]['tokens_per_candidate']} tok/cand")
        run.log(f"  worst {worst:<22} corr={results[worst]['corr_latent_grade']:.4f} "
                f"@ {results[worst]['tokens_per_candidate']} tok/cand")
        run.log(f"  spread across evidence budgets: {spread:.4f}")
        run.log("")
        if results[best]["corr_latent_grade"] < 0.25:
            run.log("  (a) NO SIGNAL: even the most generous budget leaves the base")
            run.log("      checkpoint near chance. Matches Laya's own card -- below the")
            run.log("      majority-class baseline zero-shot. Falsification test 4")
            run.log("      cannot be evaluated until Phase 1 fine-tuning; T2's result")
            run.log("      (c_S is additive) still stands and is what carries forward.")
            verdict = "no_signal_base_checkpoint"
        elif spread > 0.10:
            run.log("  (b) STARVED: the budget moved the signal materially. The")
            run.log("      bottleneck is the slicer and the layout, not the weights.")
            verdict = "evidence_starved"
        else:
            run.log("  Signal present and budget-insensitive; anchoring is testable now.")
            verdict = "signal_present"
        run.metric("verdict", verdict)
        run.metric("budget_spread", spread)
        run.log(f"\n  throughput: {rt.throughput()}")


if __name__ == "__main__":
    main()
