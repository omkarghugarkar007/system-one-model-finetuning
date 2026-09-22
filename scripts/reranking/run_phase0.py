"""Phase 0 gate on a real corpus with the real Laya checkpoint.

    python scripts/run_phase0.py --dataset trec-covid --queries 8

Writes runs/<date>-phase0-<dataset>/ with the manifest, metrics and report.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from systemone.model import LayaRuntime
from systemone.reranking.data import BM25Index, load_beir, make_signature_builder
from systemone.reranking.data.signatures import signature_recall
from systemone.reranking.experiments.phase0_identifiability import (
    anchor_recovery,
    fit_additive,
    interaction_test,
    make_block_design,
    order_noise_floor,
)
from systemone.reranking.packing import OptionsPacker, StatePacker
from systemone.reranking.scorer import LayaScorer
from systemone.utils.runner import Run


def build_probes(ds, qid, ranked_ids, n_per_grade=7):
    """Probe candidates with known grades, balanced across the grade range."""
    by_grade: dict[int, list[str]] = {}
    for d in ranked_ids:
        by_grade.setdefault(ds.grade(qid, d), []).append(d)
    probes, grades = [], []
    for g in sorted(by_grade):
        take = by_grade[g][:n_per_grade]
        probes += take
        grades += [g] * len(take)
    return probes, np.asarray(grades)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="trec-covid")
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--depth", type=int, default=100, help="first-stage pool")
    ap.add_argument("--slate-size", type=int, default=10)
    ap.add_argument("--probes-per-grade", type=int, default=10)
    ap.add_argument("--slates-per-query", type=int, default=40)
    ap.add_argument("--perms", type=int, default=8, help="order-noise permutations")
    ap.add_argument("--layout", default="options", choices=["options", "state"])
    ap.add_argument("--signature", default="title+lexical")
    ap.add_argument("--model", default="data/cache/laya")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = vars(args)
    with Run("phase0", cfg, tag=f"{args.dataset}-{args.layout}") as run:
        rng = np.random.default_rng(args.seed)

        run.section("Setup")
        ds = load_beir(args.dataset)
        run.log(f"{ds}")
        run.log(f"grades: {ds.grade_histogram()}")

        t0 = time.time()
        doc_ids = list(ds.docs)
        index = BM25Index.build(doc_ids, (ds.docs[d].full for d in doc_ids))
        run.log(f"{index}  built in {time.time() - t0:.1f}s")

        rt = LayaRuntime(args.model, apply_temperature=False)
        scorer = LayaScorer(rt, max_batch_slates=16)
        counter = scorer.token_counter()
        run.log(f"Laya on {rt.device}, temperature OFF (Phase 0 measures raw logits)")

        idf = {t: index.idf[j] for t, j in index.vocab.items()}
        sigbuild = make_signature_builder(args.signature, counter=counter,
                                          idf_lookup=lambda t: idf.get(t, 0.0))
        packer = (OptionsPacker(token_counter=counter) if args.layout == "options"
                  else StatePacker(token_counter=counter))
        budget = packer.pack("q", ["x"] * args.slate_size).info
        per_cand = budget.get("option_tokens", budget.get("per_candidate_tokens"))
        run.log(f"layout={args.layout}, signature={args.signature}, "
                f"{per_cand} tokens per candidate at M={args.slate_size}")

        qids = ds.query_ids[: args.queries]

        # ---------------------------------------------------------- per query
        all_fits, all_inter, all_anchor, floors, sig_recalls = [], [], [], [], []
        for qi, qid in enumerate(qids):
            query = ds.queries[qid]
            ranked, _ = index.search(query, args.depth)
            probes, grades = build_probes(ds, qid, ranked, args.probes_per_grade)
            if len(set(grades.tolist())) < 2 or len(probes) < args.slate_size:
                run.log(f"  [{qid}] skipped: {len(probes)} probes, "
                        f"grades {sorted(set(grades.tolist()))}")
                continue

            sigs = [sigbuild.build(query, ds.docs[d].title, ds.docs[d].text, per_cand)
                    for d in probes]
            sig_recalls += [signature_recall(query, s, ds.docs[d].full)
                            for s, d in zip(sigs, probes)]

            design = make_block_design(grades, args.slate_size,
                                       args.slates_per_query, rng)
            if not design.is_connected():
                run.log(f"  [{qid}] skipped: disconnected design")
                continue

            packed = [packer.pack(query, [sigs[i] for i in s]) for s in design.slates]
            logps = scorer.score_many(packed)
            fit = fit_additive(design, logps)
            all_fits.append(fit)
            all_inter.append(interaction_test(fit, design))

            # anchors = a spread of probes, held out of nothing here since
            # Phase 0 asks only whether the recovery machinery works
            anc = np.array([int(np.flatnonzero(grades == g)[0])
                            for g in sorted(set(grades.tolist()))])
            all_anchor.append(anchor_recovery(fit, design, anc))

            # T1 order noise on this query's first slate
            members = design.slates[0]
            perms = [np.arange(len(members))] + [
                rng.permutation(len(members)) for _ in range(args.perms - 1)]
            lp_perm = scorer.score_many(
                [packer.pack(query, [sigs[members[p]] for p in perm])
                 for perm in perms])
            floors.append(order_noise_floor(lp_perm, perms))

            run.log(f"  [{qi + 1}/{len(qids)}] {qid[:20]:<20} "
                    f"probes={len(probes):>2} rmse={fit.rmse:.4f} r2={fit.r2:.4f} "
                    f"c_range={np.ptp(fit.c_hat):.3f}")

        if not all_fits:
            run.log("\nNo usable queries. Gate cannot be evaluated.")
            return

        # ------------------------------------------------------------ results
        floor_mean = float(np.mean([f["per_candidate_std_mean"] for f in floors]))
        rmse = float(np.mean([f.rmse for f in all_fits]))
        r2 = float(np.mean([f.r2 for f in all_fits]))
        c_range = float(np.mean([np.ptp(f.c_hat) for f in all_fits]))
        ratio = rmse / max(floor_mean, 1e-9)

        run.section("T0/T1 -- measurement floor")
        run.log(f"  order-permutation std per candidate : {floor_mean:.4f} nats")
        run.log(f"  argmax changed by reordering       : "
                f"{np.mean([f['argmax_changed_frac'] for f in floors]):.1%}")
        run.log("  (Laya is deterministic, so this IS the noise floor --")
        run.log("   the plan's 'marker noise' is not a real quantity.)")

        run.section("T2 -- additive fit  log p_iS = l_i - c_S   [THE GATE]")
        run.log(f"  residual RMSE      : {rmse:.4f} nats")
        run.log(f"  R^2                : {r2:.4f}")
        run.log(f"  c_S range (signal) : {c_range:.4f} nats")
        run.log(f"  residual / floor   : {ratio:.2f}x")

        inter = {k: float(np.mean([d[k] for d in all_inter])) for k in all_inter[0]}
        run.section("T3 -- interaction (is the leftover systematic?)")
        run.log(f"  R^2 of residual on grade x slate-mean : {inter['r2_of_residual']:.4f}")
        run.log(f"  interaction coefficient               : {inter['beta_interaction']:+.4f}")

        anc = {k: float(np.nanmean([d.get(k, np.nan) for d in all_anchor]))
               for k in all_anchor[0]}
        run.section("T4 -- anchor recovery of the offset")
        run.log(f"  usable slates/query : {anc['n_slates_usable']:.0f}")
        run.log(f"  corr(recovered c_S, fitted c_S) : "
                f"{anc['corr_recovered_vs_fitted_offset']:.4f}")
        run.log(f"  offset RMSE         : {anc['offset_rmse']:.4f}")
        run.log(f"  anchor fit residual : {anc['anchor_fit_residual_mean']:.4f}")

        run.section("Evidence slicer")
        run.log(f"  signature recall (query terms kept) : {np.mean(sig_recalls):.3f}")
        run.log(f"  budget report: {scorer.budget_report()}")
        run.log(f"  throughput  : {rt.throughput()}")

        for k, v in [("order_noise_floor", floor_mean), ("additive_rmse", rmse),
                     ("additive_r2", r2), ("c_range", c_range),
                     ("residual_over_floor", ratio),
                     ("interaction_r2", inter["r2_of_residual"]),
                     ("anchor_offset_corr", anc["corr_recovered_vs_fitted_offset"]),
                     ("anchor_offset_rmse", anc["offset_rmse"]),
                     ("signature_recall", float(np.mean(sig_recalls))),
                     ("n_queries_used", len(all_fits))]:
            run.metric(k, v)
        run.metric("throughput", rt.throughput())
        run.metric("budget", scorer.budget_report())

        run.section("GATE")
        # Power first. R^2 is NOT a valid criterion here: it is
        # 1 - SS_res/SS_tot, and SS_tot is inflated by the spread of l_i, which
        # has nothing to do with whether c_S is additive. A design where c_S
        # barely moves produces a low R^2 no matter how well the model holds,
        # so the question "did the experiment have contrast?" has to be
        # answered before the fit is read at all.
        power = c_range / max(floor_mean, 1e-9)
        run.log(f"  design power (c_S range / noise floor) : {power:.2f}x")
        if power < 2.0:
            verdict = "INCONCLUSIVE"
            run.log(f"  {verdict}: c_S varies by {c_range:.3f} nats against a "
                    f"{floor_mean:.3f} floor.")
            run.log("  The slates were not different enough to move the offset, so")
            run.log("  this run cannot distinguish an additive c_S from no c_S at all.")
            run.log("  Use a corpus with an explicit grade-0 pool (trec-covid, DL19).")
        else:
            verdict = ("PASS" if ratio < 1.5 else
                       "MARGINAL" if ratio < 3.0 else "FAIL")
            run.log(f"  {verdict}: residual is {ratio:.2f}x the measurement floor "
                    f"against {power:.2f}x of signal.")
        run.log("")
        run.log("  PASS         -> residual at the floor: c_S is a per-slate shift,")
        run.log("                  and anchoring recovers what the theory says.")
        run.log("  MARGINAL     -> an affine map absorbs most of it; read T3 for shape.")
        run.log("  FAIL         -> composition-dependent beyond an additive term;")
        run.log("                  Part XI's first named risk has fired.")
        run.log("  INCONCLUSIVE -> the design lacked contrast. Not evidence either way.")
        run.metric("gate", verdict)
        run.metric("design_power", power)


if __name__ == "__main__":
    main()
