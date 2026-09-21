"""Phase 1: fine-tune Laya into an anchored reranker.

    python scripts/run_phase1_train.py --train nfcorpus --eval trec-covid

Trains on one corpus and evaluates on another by default, because the
interesting question is not "can it fit trec-covid" -- with 50 queries it
certainly can -- but whether anchored scoring transfers. nfcorpus supplies 323
queries of training signal; trec-covid supplies graded, deeply judged
evaluation on a different topic.

Gate (plan Part X, Phase 1): anchored beats naive by a large margin, and beats
a pointwise rubric at equal or lower pass count. Both are reported every eval.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from frontierrank.core.packing import OptionsPacker, StatePacker
from frontierrank.data import BM25Index, load_beir, make_signature_builder
from frontierrank.experiments.runner import Run
from frontierrank.models.laya import LayaRuntime
from frontierrank.models.laya.heads import FrontierRankModel
from frontierrank.training.anchors import AnchorPool
from frontierrank.training.slate_dataset import (SlateDataset, SlateSpec,
                                                 collate_slates)
from frontierrank.training.trainer import (TrainConfig, Trainer, TrainedScorer,
                                           make_eval_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="nfcorpus")
    ap.add_argument("--eval", default="trec-covid")
    ap.add_argument("--train-queries", type=int, default=250)
    ap.add_argument("--eval-queries", type=int, default=15)
    ap.add_argument("--slates-per-query", type=int, default=8)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--max-options", type=int, default=10)
    ap.add_argument("--n-anchors", type=int, default=4)
    ap.add_argument("--layout", default="state", choices=["options", "state"])
    ap.add_argument("--signature", default="title+lexical")
    ap.add_argument("--anchors", default="teacher",
                    choices=["templated", "judged", "teacher"])
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--full-epochs", type=int, default=2)
    ap.add_argument("--batch-slates", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--model", default="data/cache/laya")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with Run("phase1-train", vars(args), tag=f"{args.train}-to-{args.eval}") as run:
        rng = np.random.default_rng(args.seed)

        run.section("Setup")
        rt = LayaRuntime(args.model, apply_temperature=False)
        base = rt.model
        model = FrontierRankModel(base, n_levels=4).to(rt.device)
        run.log(f"Laya on {rt.device}; {model.trainable_summary()}")

        scorer_for_tokens = TrainedScorer(model, rt)
        counter = scorer_for_tokens.token_counter()
        packer = (OptionsPacker(token_counter=counter) if args.layout == "options"
                  else StatePacker(token_counter=counter))
        info = packer.pack("q", ["x"] * args.max_options).info
        per_cand = info.get("option_tokens", info.get("per_candidate_tokens"))
        run.log(f"layout={args.layout}, M={args.max_options} "
                f"({args.max_options - args.n_anchors} candidates + "
                f"{args.n_anchors} pivots), {per_cand} tokens/candidate")

        # ------------------------------------------------------------- corpora
        tr = load_beir(args.train)
        tr_ids = list(tr.docs)
        tr_index = BM25Index.build(tr_ids, (tr.docs[d].full for d in tr_ids))
        tr_idf = {t: tr_index.idf[j] for t, j in tr_index.vocab.items()}
        tr_sig = make_signature_builder(args.signature, counter=counter,
                                        idf_lookup=lambda t: tr_idf.get(t, 0.0))
        run.log(f"train {tr}")

        ev = load_beir(args.eval)
        ev_ids = list(ev.docs)
        ev_index = BM25Index.build(ev_ids, (ev.docs[d].full for d in ev_ids))
        ev_idf = {t: ev_index.idf[j] for t, j in ev_index.vocab.items()}
        ev_sig = make_signature_builder(args.signature, counter=counter,
                                        idf_lookup=lambda t: ev_idf.get(t, 0.0))
        run.log(f"eval  {ev}")

        anchor_pool = AnchorPool(strategy=args.anchors)
        if args.anchors == "teacher":
            from frontierrank.models.teachers import CachedTeacher, JevTeacher
            import os, pathlib as _pl
            if not os.environ.get("OPENROUTER_API_KEY"):
                envf = _pl.Path(".env")
                if envf.exists():
                    for line in envf.read_text().splitlines():
                        if line.startswith("OPENROUTER_API_KEY="):
                            os.environ["OPENROUTER_API_KEY"] = \
                                line.split("=", 1)[1].strip().strip("\"'")
            jev = CachedTeacher(JevTeacher(max_candidates=30),
                                "data/cache/teacher", backend_tag="jev-openrouter")
            anchor_pool.teacher = jev
            anchor_pool.index = tr_index
            anchor_pool.signature_builder = tr_sig
            run.log("anchors: teacher-graded (Jev), continuous utilities, cached")
        else:
            run.log(f"anchors: {args.anchors}, utilities "
                    f"{anchor_pool.utilities(args.n_anchors)}, "
                    f"spread {anchor_pool.spread(args.n_anchors)}")

        # -------------------------------------------------------------- slates
        spec = SlateSpec(max_options=args.max_options, n_anchors=args.n_anchors,
                         depth=args.depth, slates_per_query=args.slates_per_query)
        train_qids = tr.query_ids[: args.train_queries]
        ds_train = SlateDataset(tr, tr_index, tr_sig, anchor_pool, spec,
                                query_ids=train_qids,
                                tokens_per_candidate=per_cand, seed=args.seed)
        examples = ds_train.build()
        stats = ds_train.stats(examples)
        run.log(f"train slates: {stats}")
        run.metric("train_slate_stats", stats)

        def collate(batch):
            return collate_slates(batch, rt, packer)

        eval_qids = ev.query_ids[: args.eval_queries]
        eval_pool = anchor_pool
        if args.anchors == "teacher":
            # pivots are drawn from the corpus being scored, so eval needs its
            # own pool over the eval index -- sharing one would put training
            # documents into the evaluation slates
            eval_pool = AnchorPool(strategy="teacher", teacher=anchor_pool.teacher,
                                   index=ev_index, signature_builder=ev_sig)
        eval_fn = make_eval_fn(TrainedScorer(model, rt), ev, ev_index, ev_sig,
                               packer, eval_pool, eval_qids, depth=args.depth,
                               max_options=args.max_options,
                               n_anchors=args.n_anchors,
                               tokens_per_candidate=per_cand, seed=args.seed)

        # ------------------------------------------------------------- training
        run.section("Training")
        cfg = TrainConfig(warmup_epochs=args.warmup_epochs,
                          full_epochs=args.full_epochs,
                          batch_slates=args.batch_slates,
                          grad_accum=args.grad_accum,
                          lr_encoder=args.lr_encoder, lr_head=args.lr_head,
                          eval_every=args.eval_every, seed=args.seed)
        trainer = Trainer(model, rt, packer, cfg, log=run.log)
        history = trainer.fit(examples, collate, eval_fn)
        ckpt = trainer.save(f"{run.dir}/checkpoint.pt")

        run.section("Result")
        base_row = next(h for h in history if h["stage"] == "base")
        final = history[-1]
        run.log(f"{'stage':>8} {'step':>6} {'anchored':>10} {'naive':>8} {'BM25':>8}")
        run.log("-" * 46)
        for h in history:
            run.log(f"{h['stage']:>8} {h['step']:>6} {h['ndcg_anchored']:>10.4f} "
                    f"{h['ndcg_naive']:>8.4f} {h['ndcg_bm25']:>8.4f}")
        run.metric("history", history)
        run.metric("checkpoint", str(ckpt))

        d_anchor = final["ndcg_anchored"] - base_row["ndcg_anchored"]
        d_vs_naive = final["ndcg_anchored"] - final["ndcg_naive"]
        run.log("")
        run.log(f"  fine-tuning moved anchored nDCG@10 by {d_anchor:+.4f}")
        run.log(f"  anchored vs naive at the end          {d_vs_naive:+.4f}")
        run.log(f"  BM25 floor                            {final['ndcg_bm25']:.4f}")
        run.metric("delta_from_finetuning", d_anchor)
        run.metric("anchored_minus_naive", d_vs_naive)
        run.log("")
        run.log("  Phase 1 gate: anchored must beat naive by a large margin. A small")
        run.log("  or negative margin means the calibration is not earning its option")
        run.log("  slots, and the honest move is the pointwise rubric instead.")


if __name__ == "__main__":
    main()
