"""Fine-tuning a System One model on typed decisions. Sized for a laptop.

The loop is ordinary; what matters is the order and the guardrails, each of
which cost us a run to learn.

**Two stages, and the order is not cosmetic.** Any head you add starts from
noise. Letting its gradients into a pretrained encoder on step one destroys
what the backbone already knows, so the encoder is frozen for a warm-up and
unfrozen after.

**Check the budget before you train.** The most common silent failure is that
the state or the option text did not fit, so the model trains on evidence it
never saw. `fit()` refuses to start if the state is being truncated on more
than `max_truncation` of examples, because that is a data bug wearing a model
bug's clothing.

**Memory is the binding constraint on a laptop.** AdamW keeps two fp32 moments
per trainable parameter: a 422M model costs ~6.8 GB before activations, which
on 16 GB of unified memory means swap and a 7x slowdown. `freeze_lower_layers`
is the lever, and `fit()` prints the estimate before it starts.

**Calibrate afterwards, separately.** Training does not make a model calibrated
and can make it worse. Run `systemone.calibrate` on held-out data when this is
done; it is minutes and it is the difference between probabilities you can
threshold and numbers that merely rank.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..data.typed import collate_typed
from .losses import TypedDecisionLoss

__all__ = ["TypedTrainConfig", "TypedTrainer"]


@dataclass
class TypedTrainConfig:
    warmup_epochs: int = 1           # encoder frozen
    full_epochs: int = 2             # everything trainable
    batch_size: int = 8
    grad_accum: int = 4
    lr_encoder: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    freeze_lower_layers: int = 0     # the memory lever; 14 of 28 halves optimiser state
    eval_every: int = 200
    save_every: int = 0
    out_dir: str = "runs/checkpoints"
    max_truncation: float = 0.25     # refuse to train if more state is being cut
    loss: dict = field(default_factory=dict)
    seed: int = 0


class TypedTrainer:
    def __init__(self, model, tokenizer, cfg: TypedTrainConfig | None = None,
                 device=None, log=print):
        from ..model.runtime import pick_device
        self.model = model
        self.tok = tokenizer
        self.cfg = cfg or TypedTrainConfig()
        self.device = device or pick_device()
        self.log = log
        self.loss_fn = TypedDecisionLoss(**self.cfg.loss)
        self.history: list[dict] = []
        torch.manual_seed(self.cfg.seed)

    # ------------------------------------------------------------------ guards
    def preflight(self, dataset) -> dict:
        """Look at the data before training it. Returns the budget report."""
        budget = dataset.budget_report()
        labels = dataset.label_report()
        self.log("preflight")
        self.log(f"  data   : {labels}")
        self.log(f"  budget : {budget}")
        trunc = budget.get("state_truncated_frac", 0.0)
        if trunc > self.cfg.max_truncation:
            raise ValueError(
                f"state is truncated on {trunc:.1%} of examples (limit "
                f"{self.cfg.max_truncation:.0%}). The model cannot read that "
                f"evidence, so training on it teaches nothing. Shorten the "
                f"state, or select evidence before it reaches the model.")
        if budget.get("option_text_lost_frac", 0.0) > 0.5:
            self.log("  WARNING: over half the option text is being truncated. "
                     "Shorten option descriptions or use fewer options -- the "
                     "head budget is 192 tokens shared across ALL options.")
        maj = labels.get("majority_class_baseline", 0.0)
        self.log(f"  the number to beat: majority-class baseline = {maj:.4f}")
        return {"budget": budget, "labels": labels}

    # ------------------------------------------------------------------- steps
    def _batch_loss(self, batch):
        d = self.device
        out = self.model(batch["input_ids"].to(d), batch["attention_mask"].to(d),
                         batch["marker_pos"].to(d), batch["marker_mask"].to(d),
                         batch["qtype"].to(d))
        logits = out["slate_logits"] if isinstance(out, dict) else out[0]
        return self.loss_fn(logits=logits, target=batch["target"].to(d),
                            mask=batch["marker_mask"].to(d),
                            ordinal=batch["ordinal"].to(d),
                            weight=batch["weight"].to(d))

    def _stage(self, name, dataset, epochs, freeze_encoder, eval_fn):
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import OneCycleLR

        self.model.freeze_encoder(freeze_encoder)
        if not freeze_encoder and self.cfg.freeze_lower_layers:
            self.model.freeze_lower_layers(self.cfg.freeze_lower_layers)
        if self.cfg.gradient_checkpointing and not freeze_encoder:
            self.model.enable_gradient_checkpointing()
        mem = self.model.memory_estimate()
        self.log(f"\n[{name}] {self.model.trainable_summary()}")
        self.log(f"[{name}] ~{mem['total_GB_excl_activations']} GB before "
                 f"activations (params {mem['params_GB']}, grads "
                 f"{mem['grads_GB']}, optimiser {mem['optimizer_GB']})")

        opt = AdamW(self.model.param_groups(self.cfg.lr_encoder, self.cfg.lr_head),
                    weight_decay=self.cfg.weight_decay)
        bs, accum = self.cfg.batch_size, self.cfg.grad_accum
        steps = max(1, epochs * len(dataset) // (bs * accum))
        sched = OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups],
                           total_steps=steps, pct_start=self.cfg.warmup_frac)

        step, t0 = 0, time.time()
        for ep in range(epochs):
            examples = list(dataset.epoch(shuffle=True, permute_options=True))
            opt.zero_grad(set_to_none=True)
            running, seen = {}, 0
            for b in range(0, len(examples) - bs + 1, bs):
                batch = collate_typed(examples[b:b + bs], self.tok, dataset)
                if batch is None:
                    continue
                loss, parts = self._batch_loss(batch)
                (loss / accum).backward()
                for k, v in parts.items():
                    running[k] = running.get(k, 0.0) + float(v.detach())
                seen += 1
                if (b // bs + 1) % accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.cfg.max_grad_norm)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 10 == 0:
                        self.log(f"  [{name}] ep{ep} {step}/{steps} "
                                 + " ".join(f"{k}={v / max(seen,1):.4f}"
                                            for k, v in sorted(running.items()))
                                 + f"  {time.time() - t0:.0f}s")
                        running, seen = {}, 0
                    if self.cfg.save_every and step % self.cfg.save_every == 0:
                        self.save(f"{self.cfg.out_dir}/{name}-step{step}.pt")
                    if eval_fn and step % self.cfg.eval_every == 0:
                        m = eval_fn()
                        self.history.append({"stage": name, "step": step, **m})
                        self.log(f"  [{name}] eval @ {step}: {m}")
                    if self.device.type == "mps":
                        torch.mps.empty_cache()
        return step

    # --------------------------------------------------------------------- fit
    def fit(self, dataset, eval_fn=None, skip_preflight: bool = False):
        if not skip_preflight:
            self.preflight(dataset)
        self.model.to(self.device).train()
        if eval_fn:
            m = eval_fn()
            self.history.append({"stage": "base", "step": 0, **m})
            self.log(f"\n[base] before training: {m}")
        if self.cfg.warmup_epochs:
            self._stage("warmup", dataset, self.cfg.warmup_epochs, True, eval_fn)
        if self.cfg.full_epochs:
            self._stage("full", dataset, self.cfg.full_epochs, False, eval_fn)
        if eval_fn:
            m = eval_fn()
            self.history.append({"stage": "final", "step": -1, **m})
            self.log(f"\n[final] {m}")
        return self.history

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(),
                    "config": self.cfg.__dict__, "history": self.history}, p)
        return p

    # -------------------------------------------------------------------- eval
    @torch.no_grad()
    def predict(self, dataset, examples=None, batch_size: int = 16):
        """Probabilities per example, in input order. Uncalibrated by design."""
        self.model.eval()
        items = list(examples if examples is not None else dataset.examples)
        out = []
        for i in range(0, len(items), batch_size):
            batch = collate_typed(items[i:i + batch_size], self.tok, dataset)
            if batch is None:
                continue
            d = self.device
            o = self.model(batch["input_ids"].to(d), batch["attention_mask"].to(d),
                           batch["marker_pos"].to(d), batch["marker_mask"].to(d),
                           batch["qtype"].to(d))
            logits = (o["slate_logits"] if isinstance(o, dict) else o[0]).float().cpu()
            for r, ex in enumerate(batch["examples"]):
                k = ex.question.n_options
                z = logits[r, :k].numpy()
                e = np.exp(z - z.max())
                out.append({"example": ex, "logits": z, "probs": e / e.sum()})
        self.model.train()
        return out
