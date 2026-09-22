"""The three heads the reranker needs, on one Laya backbone.

Laya ships two of them and neither is quite right for ranking:

  slate head   the existing shared scalar MLP over [MASK] markers, softmaxed
               within the slate. This is the ranking signal and it stays, but
               it is trained listwise here rather than as a classification.

  ordinal head NEW. Laya's `score` primitive is not an ordinal head -- it
               renders levels as "level %d: %s" and treats them as ordinary
               Choice options, with only the RPS reward term knowing they are
               ordered. So the model can happily put mass on levels 0 and 3
               and none on 1 and 2. A CORAL head emits K-1 cumulative logits
               P(y > k) with a shared slope, which makes the grades monotone by
               construction and gives a globally comparable pointwise score --
               the thing a slate softmax structurally cannot provide.

  act head     already present, reading [pooled, top1, top1-top2, entropy,
               k/255]. Laya trains it for "escalate to a human". The controller
               needs a different question: "will a teacher call change the
               top-k?". Same inputs, different target, so it is retargeted
               rather than replaced.

The ordinal head is initialised from scratch, so it needs a warm-up where the
backbone is frozen -- otherwise its random gradients wash through an encoder
that already knows something.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .runtime import DecisionModel

__all__ = ["OrdinalHead", "SystemOneModel", "coral_probs"]


class OrdinalHead(nn.Module):
    """CORAL: K-1 binary tasks P(y > k) sharing one projection, with per-level bias.

    Sharing the projection and giving each threshold only a bias is what forces
    monotonicity: P(y > 0) >= P(y > 1) >= ... holds for every input, because the
    logits differ only by an additive constant that is ordered at init and stays
    ordered under training.
    """

    def __init__(self, hidden: int, n_levels: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_levels = n_levels
        self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                  nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1, bias=False))
        # descending init so the thresholds start ordered
        self.thresholds = nn.Parameter(torch.linspace(1.0, -1.0, n_levels - 1))

    def forward(self, marker_hidden: torch.Tensor) -> torch.Tensor:
        """(B, M, d) -> (B, M, K-1) cumulative logits."""
        return self.proj(marker_hidden) + self.thresholds


def coral_probs(cum_logits: torch.Tensor) -> torch.Tensor:
    """Cumulative logits -> a proper distribution over levels.

    P(y=0) = 1 - P(y>0);  P(y=k) = P(y>k-1) - P(y>k);  P(y=K-1) = P(y>K-2).
    Clamped because the subtraction can go slightly negative when the
    thresholds drift out of order early in training.
    """
    cum = torch.sigmoid(cum_logits)
    ones = torch.ones_like(cum[..., :1])
    zeros = torch.zeros_like(cum[..., :1])
    upper = torch.cat([ones, cum], dim=-1)
    lower = torch.cat([cum, zeros], dim=-1)
    return (upper - lower).clamp_min(1e-6)


class SystemOneModel(nn.Module):
    """A System One decision model plus an ordinal head, sharing marker states.

    Deliberately wraps rather than subclasses: the shipped checkpoint must keep
    loading with `strict=True` into the original module tree, so that a
    mismatch after a vendor update is a loud error instead of a silent
    re-initialisation.
    """

    def __init__(self, base: DecisionModel, n_levels: int = 4):
        super().__init__()
        self.base = base
        d = base.encoder.config.hidden_size
        self.ordinal = OrdinalHead(d, n_levels)
        self.n_levels = n_levels

    # ------------------------------------------------------------------ parts
    def marker_states(self, input_ids, attention_mask, marker_pos, qtype):
        """The shared computation: encoder -> head -> gather markers."""
        b = self.base
        h = b.encoder(input_ids=input_ids,
                      attention_mask=attention_mask).last_hidden_state
        h = h + b.type_emb(qtype)[:, None, :]
        if b.head is not None:
            pad = ~attention_mask.bool()
            for layer in b.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        return torch.gather(h, 1, idx), h[:, 0]

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        m, pooled = self.marker_states(input_ids, attention_mask, marker_pos, qtype)

        slate_logits = self.base.scorer(m).squeeze(-1).float()
        slate_logits = slate_logits.masked_fill(~marker_mask, -1e4)
        ordinal_logits = self.ordinal(m).float()

        p = torch.softmax(slate_logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        # a single-option slate is legitimate -- the ordinal head is pointwise
        # and gets probed one candidate at a time -- so topk(2) must not assume
        # a second option exists
        n_opt = p.shape[-1]
        top2 = p.topk(min(2, n_opt), -1).values
        margin = (top2[:, 0] - top2[:, 1]) if n_opt >= 2 else top2[:, 0]
        feats = torch.stack([top2[:, 0], margin, ent, k / 255.0], -1)
        act_logits = self.base.act_head(torch.cat([pooled.float(), feats], -1))

        return {"slate_logits": slate_logits,
                "ordinal_logits": ordinal_logits,
                "level_probs": coral_probs(ordinal_logits),
                "act_logits": act_logits}

    # ------------------------------------------------------------- parameters
    def freeze_encoder(self, frozen: bool = True):
        """Warm-up setting. The ordinal head starts from noise; letting its
        gradients into a pretrained encoder on step one is how you lose the
        backbone's existing competence."""
        for p in self.base.encoder.parameters():
            p.requires_grad = not frozen
        return self

    def freeze_lower_layers(self, n_frozen: int):
        """Freeze the bottom `n_frozen` encoder layers. The memory lever on a Mac.

        Optimiser state dominates: AdamW keeps two fp32 moments per trainable
        parameter, so 422M trainable params cost ~3.4 GB of state on top of
        1.7 GB of parameters and 1.7 GB of gradients. On a 16 GB M4 shared with
        the OS that is what pushes the machine into swap, and a swapping
        training run goes from 27 s/step to 210 s/step.

        Freezing the lower half is the cheapest fix and costs little for
        reranking: the bottom layers of a pretrained encoder do general
        language, and the task-specific work happens near the top. It is a
        quality/memory trade, so it is reported rather than applied silently.
        """
        layers = getattr(self.base.encoder, "layers", None)
        if layers is None:
            enc = getattr(self.base.encoder, "encoder", None)
            layers = getattr(enc, "layer", None) if enc is not None else None
        if layers is None:
            raise AttributeError("could not find the encoder's layer list")
        for i, layer in enumerate(layers):
            if i < n_frozen:
                for p in layer.parameters():
                    p.requires_grad = False
        # embeddings go with the bottom of the stack
        if n_frozen > 0:
            emb = getattr(self.base.encoder, "embeddings", None)
            if emb is not None:
                for p in emb.parameters():
                    p.requires_grad = False
        return self

    def memory_estimate(self, optimizer: str = "adamw") -> dict:
        """Rough training footprint in GB, so a config can be checked before it swaps."""
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        states = {"adamw": 2, "sgd": 0, "sgd_momentum": 1}.get(optimizer, 2)
        gb = lambda n: round(n * 4 / 1e9, 2)          # noqa: E731 - fp32
        return {"trainable_M": round(tr / 1e6, 1),
                "params_GB": gb(tot), "grads_GB": gb(tr),
                "optimizer_GB": gb(tr * states),
                "total_GB_excl_activations": gb(tot + tr + tr * states)}

    def enable_gradient_checkpointing(self):
        """Non-optional on 16 GB of unified memory at 512 tokens."""
        if hasattr(self.base.encoder, "gradient_checkpointing_enable"):
            self.base.encoder.gradient_checkpointing_enable()
        return self

    def param_groups(self, lr_encoder: float = 2e-5, lr_head: float = 1e-4):
        """Lower LR on the pretrained encoder than on the from-scratch heads."""
        enc = list(self.base.encoder.parameters())
        enc_ids = {id(p) for p in enc}
        rest = [p for p in self.parameters() if id(p) not in enc_ids]
        return [{"params": [p for p in enc if p.requires_grad], "lr": lr_encoder},
                {"params": [p for p in rest if p.requires_grad], "lr": lr_head}]

    def trainable_summary(self) -> dict:
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.parameters())
        return {"trainable_M": round(tr / 1e6, 1), "total_M": round(tot / 1e6, 1),
                "frozen_frac": round(1 - tr / tot, 3)}
