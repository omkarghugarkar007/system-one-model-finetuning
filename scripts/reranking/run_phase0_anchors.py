"""Falsification tests 3-5: does anchoring buy the cross-slate scale, and at what A?

    python scripts/run_phase0_anchors.py --dataset trec-covid --queries 10

T2 (run_phase0.py) showed c_S exists and is additive. This asks the harder
question: can a handful of pivots recover it well enough to make scores
comparable across slates? That is an SNR question -- Laya's order-noise floor
is comparable to the c_S spread -- so the answer is a curve in the anchor count,
not a verdict.
"""
from __future__ import annotations

import argparse

import numpy as np

from systemone.reranking.packing import OptionsPacker, StatePacker
from systemone.reranking.data import BM25Index, load_beir, make_signature_builder
from systemone.reranking.experiments.phase0_identifiability import (
    anchored_slate_design, cross_slate_scale, estimate_slope_prior,
    fit_additive, make_block_design, spread_anchors)
from systemone.utils.runner import Run
from systemone.model import LayaRuntime
from systemone.reranking.scorer import LayaScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="trec-covid")
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--slate-size", type=int, default=10)
    ap.add_argument("--slates-per-query", type=int, default=40)
    ap.add_argument("--probes-per-grade", type=int, default=12)
    ap.add_argument("--anchor-counts", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--layout", default="options", choices=["options", "state"])
    ap.add_argument("--anchor-spread", default="grade",
                    choices=["grade", "latent", "teacher"])
    ap.add_argument("--signature", default="title+lexical")
    ap.add_argument("--model", default="data/cache/laya")
    ap.add_argument("--checkpoint", default="", help="a fine-tuned FrontierRankModel")
    ap.add_argument("--homogeneous-frac", type=float, default=0.0,
                    help="fraction of slates drawn from a single grade; this is "
                         "what makes c_S vary and gives anchors something to fix")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with Run("phase0-anchors", vars(args), tag=args.dataset) as run:
        rng = np.random.default_rng(args.seed)
        ds = load_beir(args.dataset)
        run.section("Setup")
        run.log(f"{ds}")
        doc_ids = list(ds.docs)
        index = BM25Index.build(doc_ids, (ds.docs[d].full for d in doc_ids))
        run.log(f"{index}")

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
        sigbuild = make_signature_builder(args.signature, counter=counter,
                                          idf_lookup=lambda t: idf.get(t, 0.0))
        packer = (OptionsPacker(token_counter=counter) if args.layout == "options"
                  else StatePacker(token_counter=counter))
        info = packer.pack("q", ["x"] * args.slate_size).info
        per_cand = info.get("option_tokens", info.get("per_candidate_tokens"))
        run.log(f"layout={args.layout}, {per_cand} tokens/candidate at M={args.slate_size}")
        run.log(f"slate composition: homogeneous_frac={args.homogeneous_frac} "
                f"({'c_S varies -- anchors have something to fix' if args.homogeneous_frac > 0 else 'uniform draws -- c_S barely moves'})")

        teacher = None
        if args.anchor_spread == "teacher":
            import os
            import pathlib as _pl
            from systemone.teachers import CachedTeacher, JevTeacher
            if not os.environ.get("OPENROUTER_API_KEY"):
                envf = _pl.Path(".env")
                if envf.exists():
                    for line in envf.read_text().splitlines():
                        if line.startswith("OPENROUTER_API_KEY="):
                            os.environ["OPENROUTER_API_KEY"] = \
                                line.split("=", 1)[1].strip().strip("\"'")
            teacher = CachedTeacher(JevTeacher(max_candidates=40),
                                    "data/cache/teacher", backend_tag="jev-openrouter")
            run.log("pivot utilities: Jev fractional expected grade (continuous)")
            run.log("evaluation truth: qrel grades -- independent of the pivots")

        rows: dict[int, list[dict]] = {a: [] for a in args.anchor_counts}
        used = 0
        for qid in ds.query_ids[: args.queries]:
            query = ds.queries[qid]
            ranked, _ = index.search(query, args.depth)
            by_grade: dict[int, list[str]] = {}
            for d in ranked:
                by_grade.setdefault(ds.grade(qid, d), []).append(d)
            probes, grades = [], []
            for g in sorted(by_grade):
                take = by_grade[g][: args.probes_per_grade]
                probes += take
                grades += [g] * len(take)
            grades = np.asarray(grades, dtype=float)
            if len(set(grades.tolist())) < 3 or len(probes) < args.slate_size + 6:
                continue

            sigs = [sigbuild.build(query, ds.docs[d].title, ds.docs[d].text, per_cand)
                    for d in probes]

            # a free additive fit gives each probe a latent logit l_i, which is
            # the best available stand-in for "true utility on this query"
            free = make_block_design(grades, args.slate_size,
                                     args.slates_per_query, rng)
            fit = fit_additive(free, scorer.score_many(
                [packer.pack(query, [sigs[i] for i in s]) for s in free.slates]))
            latent = fit.l_hat

            # what the pivots claim to know. Teacher grades are fractional, so
            # they give the affine fit real leverage where three qrel integers
            # give it almost none -- the limit F8 kept hitting.
            if args.anchor_spread == "teacher":
                pivot_utility = np.asarray(
                    teacher.grade(query, sigs).expected_grade, dtype=float)
            else:
                pivot_utility = grades
            prior = estimate_slope_prior(
                [np.asarray(x) for x in [fit.y]], pivot_utility, free)
            for A in args.anchor_counts:
                if args.slate_size - A < 3:
                    continue
                # spread over the KNOWN utility, not the latent: pivots picked
                # for latent spread can all carry the same grade, leaving the
                # regression with one distinct x-value and no leverage
                if args.anchor_spread == "teacher":
                    anc = spread_anchors(pivot_utility, A)
                elif args.anchor_spread == "grade":
                    anc = spread_anchors(grades, A)
                else:
                    anc = spread_anchors(latent, A)
                design = anchored_slate_design(
                    len(probes), anc, args.slate_size, args.slates_per_query, rng,
                    grades=grades, homogeneous_frac=args.homogeneous_frac)
                logps = scorer.score_many(
                    [packer.pack(query, [sigs[i] for i in s]) for s in design.slates])
                # anchors carry their KNOWN utility; here that is the graded
                # label, which is what a production pivot set actually has
                # pivots carry their measured utility; the evaluation truth
                # stays the qrel grade, so nothing is circular
                r = cross_slate_scale(design, logps, anc, pivot_utility[anc],
                                      grades, slope_prior=prior)
                r["distinct_anchor_utilities"] = int(
                    np.unique(np.round(pivot_utility[anc], 3)).size)
                r["slope_prior"] = prior
                r["A"] = A
                rows[A].append(r)
            used += 1
            run.log(f"  [{used}] {qid[:14]:<14} probes={len(probes):>2} "
                    f"latent spread={np.ptp(latent):.2f}")

        if not used:
            run.log("no usable queries")
            return

        run.section("Falsification test 4 -- cross-slate absolute scale")
        run.log("Pooled correlation with the graded label, across all slates.")
        run.log("Naive log-probabilities have no absolute scale; anchoring must beat them.")
        run.log("")
        run.log(f"{'A':>3} {'cands/slate':>12} {'r naive':>9} {'r anchored':>11} "
                f"{'delta':>8} {'RMSE':>7} {'slope':>13} {'resid':>7} {'uniq u':>7}")
        run.log("-" * 86)
        summary = {}
        for A in args.anchor_counts:
            rs = rows.get(A) or []
            if not rs:
                continue
            g = lambda k: float(np.nanmean([r[k] for r in rs]))  # noqa: E731
            summary[A] = {k: g(k) for k in
                          ("r_naive", "r_anchored", "delta_r", "rmse_anchored",
                           "anchor_residual_mean", "slope_mean", "slope_std",
                           "distinct_anchor_utilities", "slope_prior")}
            summary[A]["delta_r_std"] = float(np.std([r["delta_r"] for r in rs]))
            summary[A]["n_queries"] = len(rs)
            run.log(f"{A:>3} {args.slate_size - A:>12} {g('r_naive'):>9.4f} "
                    f"{g('r_anchored'):>11.4f} {g('delta_r'):>+8.4f} "
                    f"{g('rmse_anchored'):>7.3f} "
                    f"{g('slope_mean'):>7.2f}+-{g('slope_std'):<4.2f} "
                    f"{g('anchor_residual_mean'):>7.3f} "
                    f"{g('distinct_anchor_utilities'):>7.1f}")
        run.metric("by_anchor_count", summary)
        run.metric("n_queries", used)

        best = max(summary, key=lambda a: summary[a]["r_anchored"]) if summary else None
        run.section("Read")
        if best is not None:
            b = summary[best]
            run.log(f"  best A = {best}: r {b['r_naive']:.4f} -> {b['r_anchored']:.4f} "
                    f"({b['delta_r']:+.4f})")
            deltas = np.array([summary[a]["delta_r"] for a in summary])
            # The uncertainty that matters is across QUERIES, not across anchor
            # counts: the A cells share the same queries, so their spread is not
            # a standard error and using it overstates significance badly.
            per_query_se = float(np.mean([summary[a]["delta_r_std"] for a in summary])
                                 / max(1.0, np.sqrt(used)))
            run.log(f"  mean delta_r across A : {deltas.mean():+.4f}")
            run.log(f"  per-query SE          : {per_query_se:.4f}  (n={used} queries)")
            run.log(f"  distinct anchor utilities available : "
                    f"{max(summary[a]['distinct_anchor_utilities'] for a in summary):.1f}")
            if deltas.mean() <= 2 * per_query_se:
                run.log("")
                run.log("  NOT SIGNIFICANT. The effect is smaller than the per-query")
                run.log("  standard error, so falsification test 4 neither passes nor")
                run.log("  fails here -- it is underpowered.")
                run.log("")
                run.log("  The binding limit is visible in the last column: a 3-level")
                run.log("  grade scale gives the anchor regression at most 3 distinct")
                run.log("  utility values, against a ~0.38-nat measurement floor. The")
                run.log("  plan's simulation assumed continuous latent utilities, which")
                run.log("  no BEIR-style corpus provides.")
                run.log("")
                run.log("  To make this test decisive, give the pivots continuous")
                run.log("  utilities -- Jev's Score returns a FRACTIONAL expected grade,")
                run.log("  so teacher-graded pivots have real leverage where qrel grades")
                run.log("  do not. That is also what production would use anyway.")
                run.log(f"  Cost side regardless: A={best} is "
                        f"{args.slate_size / (args.slate_size - best):.2f}x the passes.")
            else:
                run.log("  Anchoring improves the pooled scale. The cost is "
                        f"{best} of {args.slate_size} option slots, i.e. "
                        f"{args.slate_size / (args.slate_size - best):.2f}x the passes.")
            run.metric("best_anchor_count", best)
            run.metric("best_delta_r", b["delta_r"])
        run.log(f"  throughput: {rt.throughput()}")


if __name__ == "__main__":
    main()
