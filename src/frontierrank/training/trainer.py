"""Fine-tuning loop, sized for 16 GB of Apple unified memory.

Two stages, and the order is not cosmetic. The ordinal head starts from noise;
letting its gradients into a pretrained encoder on step one destroys what the
backbone already knows. So:

    warmup   encoder frozen. Trains the head stack, the scorer MLP and the new
             ordinal head. Cheap, fast, and it gets the new head to something
             sane before it is allowed to touch the backbone.
    full     everything unfrozen, with gradient checkpointing and a much lower
             encoder learning rate.

Memory, measured on an M4: ModernBERT-large in fp32 is 1.58 GB of parameters,
the same again in gradients, and twice that in Adam state -- about 6.3 GB
before a single activation. At 512 tokens the activations are what actually
bites, so gradient checkpointing is not optional and the batch is small with
accumulation making up the effective size.

fp32 throughout. Autocast on MPS is not worth the numerical risk for a model
whose entire value proposition is calibrated probabilities.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..eval import ndcg_at_k
from .losses import CompositeRankingLoss

__all__ = ["TrainConfig", "Trainer", "TrainedScorer"]


@dataclass
class TrainConfig:
    warmup_epochs: int = 1
    full_epochs: int = 2
    batch_slates: int = 4
    grad_accum: int = 8              # effective batch = 32 slates
    lr_encoder: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    eval_every: int = 200            # optimiser steps
    n_levels: int = 4
    loss_weights: dict = field(default_factory=dict)
    out_dir: str = "runs/checkpoints"
    seed: int = 0


class TrainedScorer:
    """Adapts a `FrontierRankModel` to `ScorerProtocol` so eval reuses AnchoredScorer.

    Eval must go through exactly the inference path -- same packer, same slate
    construction, same calibrator -- or it measures a pipeline nobody will ship.
    """

    def __init__(self, model, runtime, qtype: str = "choice",
                 max_batch_slates: int = 8):
        self.model, self.rt = model, runtime
        self.qtype, self.max_batch_slates = qtype, max_batch_slates

    def token_counter(self):
        tok = self.rt.tok
        return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def _logits(self, built):
        from ..models.laya.vendored import QTYPES
        n = len(built)
        L = max(len(b[0]) for b in built)
        kmax = max(len(b[1]) for b in built)
        pad = self.rt.tok.pad_token_id
        ids = torch.full((n, L), pad, dtype=torch.long)
        att = torch.zeros((n, L), dtype=torch.long)
        mpos = torch.zeros((n, kmax), dtype=torch.long)
        mmask = torch.zeros((n, kmax), dtype=torch.bool)
        for i, (seq, mk, _) in enumerate(built):
            ids[i, :len(seq)] = torch.tensor(seq)
            att[i, :len(seq)] = 1
            mpos[i, :len(mk)] = torch.tensor(mk)
            mmask[i, :len(mk)] = True
        dev = self.rt.device
        out = self.model(ids.to(dev), att.to(dev), mpos.to(dev), mmask.to(dev),
                         torch.full((n,), QTYPES[self.qtype]).to(dev))
        return out["slate_logits"].float().cpu().numpy()

    def score_many(self, slates):
        was_training = self.model.training
        self.model.eval()
        out = []
        for i in range(0, len(slates), self.max_batch_slates):
            chunk = list(slates[i:i + self.max_batch_slates])
            built = [self.rt.build(s.state, self.qtype, s.instructions,
                                   list(s.options), rendered=True) for s in chunk]
            logits = self._logits(built)
            for j, (_, mk, _) in enumerate(built):
                out.append(self.rt.log_softmax(logits[j, :len(mk)]))
        if was_training:
            self.model.train()
        return out

    def choice_logprobs(self, instructions, options, state):
        from ..core.packing import PackedSlate
        return self.score_many([PackedSlate(instructions, list(options), state)])[0]


class Trainer:
    def __init__(self, model, runtime, packer, cfg: TrainConfig | None = None,
                 log=print):
        self.model, self.rt, self.packer = model, runtime, packer
        self.cfg = cfg or TrainConfig()
        self.log = log
        self.device = runtime.device
        self.loss_fn = CompositeRankingLoss(n_levels=self.cfg.n_levels,
                                            **self.cfg.loss_weights)
        self.history: list[dict] = []
        torch.manual_seed(self.cfg.seed)

    # ------------------------------------------------------------------ steps
    def _batch_loss(self, batch, teacher_probs=None):
        dev = self.device
        out = self.model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev),
                         batch["marker_pos"].to(dev), batch["marker_mask"].to(dev),
                         batch["qtype"].to(dev))
        grades = batch["grades"].to(dev)
        is_anchor = batch["is_anchor"].to(dev)
        # graded, present, and not a pivot: pivots are the ruler, not the data
        mask = batch["marker_mask"].to(dev) & ~is_anchor & (grades >= 0)
        total, parts = self.loss_fn(
            slate_logits=out["slate_logits"], rel=grades.clamp_min(0), mask=mask,
            teacher_probs=teacher_probs, ordinal_logits=out["ordinal_logits"],
            level_probs=out["level_probs"],
            anchor_utility=batch["anchor_utility"].to(dev), is_anchor=is_anchor)
        return total, {k: float(v) for k, v in parts.items()}

    def _run_stage(self, name: str, examples, collate, epochs: int,
                   freeze_encoder: bool, eval_fn=None):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import OneCycleLR

        self.model.freeze_encoder(freeze_encoder)
        if self.cfg.gradient_checkpointing and not freeze_encoder:
            self.model.enable_gradient_checkpointing()
        self.log(f"\n[{name}] {self.model.trainable_summary()}")

        opt = AdamW(self.model.param_groups(self.cfg.lr_encoder, self.cfg.lr_head),
                    weight_decay=self.cfg.weight_decay)
        bs, accum = self.cfg.batch_slates, self.cfg.grad_accum
        steps = max(1, epochs * len(examples) // (bs * accum))
        sched = OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                           total_steps=steps, pct_start=self.cfg.warmup_frac)

        rng = np.random.default_rng(self.cfg.seed)
        step, t0 = 0, time.time()
        for ep in range(epochs):
            order = rng.permutation(len(examples))
            # option order is re-randomised every epoch: the mechanism is
            # position-robust, but F4 measured 33% top-1 movement on the base
            # checkpoint, so robust is not invariant
            shuffled = [examples[i].permuted(rng) for i in order]
            opt.zero_grad(set_to_none=True)
            running: dict[str, float] = {}
            for b in range(0, len(shuffled) - bs + 1, bs):
                batch = collate(shuffled[b:b + bs])
                if batch is None:
                    continue
                loss, parts = self._batch_loss(batch)
                (loss / accum).backward()
                for k, v in parts.items():
                    running[k] = running.get(k, 0.0) + v
                if (b // bs + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.cfg.max_grad_norm)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 10 == 0:
                        n = max(1, accum * 10)
                        self.log(f"  [{name}] ep{ep} step {step}/{steps} "
                                 + " ".join(f"{k}={v / n:.4f}"
                                            for k, v in sorted(running.items()))
                                 + f"  {time.time() - t0:.0f}s")
                        running = {}
                    if eval_fn and step % self.cfg.eval_every == 0:
                        m = eval_fn()
                        self.history.append({"stage": name, "step": step, **m})
                        self.log(f"  [{name}] eval @ {step}: {m}")
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
        return step

    # -------------------------------------------------------------------- fit
    def fit(self, examples, collate, eval_fn=None):
        self.model.to(self.device).train()
        if eval_fn:
            m = eval_fn()
            self.history.append({"stage": "base", "step": 0, **m})
            self.log(f"[base] before training: {m}")
        if self.cfg.warmup_epochs:
            self._run_stage("warmup", examples, collate, self.cfg.warmup_epochs,
                            freeze_encoder=True, eval_fn=eval_fn)
        if self.cfg.full_epochs:
            self._run_stage("full", examples, collate, self.cfg.full_epochs,
                            freeze_encoder=False, eval_fn=eval_fn)
        if eval_fn:
            m = eval_fn()
            self.history.append({"stage": "final", "step": -1, **m})
            self.log(f"[final] {m}")
        return self.history

    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(),
                    "config": self.cfg.__dict__, "history": self.history}, p)
        return p


def make_eval_fn(scorer, ds, index, sigbuilder, packer, anchor_pool, query_ids,
                 depth: int = 100, k: int = 10, max_options: int = 10,
                 n_anchors: int = 4, tokens_per_candidate: int = 37,
                 seed: int = 0):
    """nDCG@k over a realistic first-stage pool, through the full inference path.

    Reports anchored *and* naive so the ablation that matters -- does the
    calibration earn its slots? -- is visible on every eval, not just at the end.
    """
    from ..core.scoring import AnchoredScorer

    def _eval():
        nd_a, nd_n, nd_bm = [], [], []
        for qid in query_ids:
            query = ds.queries[qid]
            ranked, _ = index.search(query, depth)
            ranked = [d for d in ranked if d not in anchor_pool.held_out]
            if len(ranked) < max_options:
                continue
            grades = np.array([ds.grade(qid, d) for d in ranked], dtype=float)
            sigs = [sigbuilder.build(query, ds.docs[d].title, ds.docs[d].text,
                                     tokens_per_candidate) for d in ranked]
            anchors = anchor_pool.for_query(query, ds, qid, n_anchors)
            pool = AnchoredScorer(scorer, anchors, packer=packer,
                                  max_options=max_options, n_anchors=n_anchors
                                  ).score(query, sigs,
                                          rng=np.random.default_rng(seed))
            nd_a.append(ndcg_at_k(grades[pool.order], grades, k))
            nd_n.append(ndcg_at_k(grades[pool.naive_order], grades, k))
            nd_bm.append(ndcg_at_k(grades, grades, k))
        return {"ndcg_anchored": round(float(np.mean(nd_a)), 4),
                "ndcg_naive": round(float(np.mean(nd_n)), 4),
                "ndcg_bm25": round(float(np.mean(nd_bm)), 4),
                "n_queries": len(nd_a)}

    return _eval
