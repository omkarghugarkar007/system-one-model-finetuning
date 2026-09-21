"""Composite objective for fine-tuning a typed-decision reranker.

    L = w_point * BCE            pointwise calibrated relevance
      + w_pair  * lambda-RankNet top-heavy pairwise, weighted by |dNDCG|
      + w_list  * ListNet/PL     listwise over the slate
      + w_kd    * KL(teacher)    distribution distillation, not hard labels
      + w_ord   * CORAL          ordinal head over the 0-3 TREC scale
      + w_cal   * RPS            ranked probability score, the ordinal-aware
                                 strictly proper rule Laya's own RLCD uses

Distilling the teacher's DISTRIBUTION is the point. "Jev says relevant" throws
away almost everything; P(0)=.01, P(1)=.04, P(2)=.21, P(3)=.74 carries the
teacher's uncertainty, which is exactly the signal the student needs to learn
where its own uncertainty should live.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["delta_ndcg_weights", "lambda_ranknet_loss", "listnet_loss",
           "distill_kl", "coral_loss", "rps_loss", "CompositeRankingLoss"]


def _gain(rel):
    return torch.pow(2.0, rel) - 1.0


def delta_ndcg_weights(rel: torch.Tensor, scores: torch.Tensor, eps: float = 1e-9):
    """|dNDCG| for every pair if their current positions were swapped.

    Confusing rank 1 with rank 100 must cost more than rank 76 with rank 77.
    """
    n = rel.shape[-1]
    order = torch.argsort(scores, dim=-1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(n, device=rel.device).expand_as(order))
    disc = 1.0 / torch.log2(rank.float() + 2.0)
    g = _gain(rel)

    ideal = torch.sort(rel, dim=-1, descending=True).values
    idisc = 1.0 / torch.log2(torch.arange(n, device=rel.device).float() + 2.0)
    idcg = (_gain(ideal) * idisc).sum(-1, keepdim=True).clamp_min(eps)

    dg = g.unsqueeze(-1) - g.unsqueeze(-2)          # g_i - g_j
    dd = disc.unsqueeze(-1) - disc.unsqueeze(-2)    # d_i - d_j
    return (dg * dd).abs() / idcg.unsqueeze(-1)


def lambda_ranknet_loss(scores, rel, mask=None):
    """Top-heavy pairwise. Only pairs with a strict preference contribute."""
    s = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    pref = (rel.unsqueeze(-1) > rel.unsqueeze(-2)).float()
    if mask is not None:
        m = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        pref = pref * m
    w = delta_ndcg_weights(rel, scores).detach()
    loss = -F.logsigmoid(s) * pref * w
    denom = pref.sum().clamp_min(1.0)
    return loss.sum() / denom


def listnet_loss(scores, rel, tau: float = 1.0, mask=None):
    """Cross-entropy between the score softmax and the gain softmax (top-1 PL)."""
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), -1e4)
        rel = rel.masked_fill(~mask.bool(), -1e4)
    p = F.log_softmax(scores / tau, dim=-1)
    q = F.softmax(_gain(rel.clamp_min(0)) / tau, dim=-1)
    return -(q * p).sum(-1).mean()


def distill_kl(student_logits, teacher_probs, tau: float = 1.0, mask=None):
    """KL(teacher || student) over the slate. The core of frontier distillation."""
    if mask is not None:
        student_logits = student_logits.masked_fill(~mask.bool(), -1e4)
    logp = F.log_softmax(student_logits / tau, dim=-1)
    t = teacher_probs.clamp_min(1e-8)
    t = t / t.sum(-1, keepdim=True)
    return (t * (t.log() - logp)).sum(-1).mean() * (tau ** 2)


def coral_loss(ordinal_logits, rel, n_levels: int = 4):
    """CORAL: K-1 shared-slope binary tasks, P(y > k). Keeps ranks monotone."""
    tgt = torch.stack([(rel > k).float() for k in range(n_levels - 1)], dim=-1)
    return F.binary_cross_entropy_with_logits(ordinal_logits, tgt)


def rps_loss(probs, rel, n_levels: int = 4):
    """Ranked probability score -- strictly proper AND ordinal-aware.

    This is the rule Laya's own RLCD applies to `score` questions
    (w_rps = 1.0). Being off by one level should cost less than being off by
    three; log-loss alone does not know that.
    """
    cdf_p = torch.cumsum(probs, dim=-1)
    onehot = F.one_hot(rel.long().clamp(0, n_levels - 1), n_levels).float()
    cdf_t = torch.cumsum(onehot, dim=-1)
    return (((cdf_p - cdf_t) ** 2).sum(-1) / (n_levels - 1)).mean()


class CompositeRankingLoss(torch.nn.Module):
    def __init__(self, w_point=0.5, w_pair=1.0, w_list=1.0, w_kd=2.0,
                 w_ord=0.5, w_cal=1.0, tau=1.0, n_levels=4):
        super().__init__()
        self.w = dict(point=w_point, pair=w_pair, list=w_list,
                      kd=w_kd, ord=w_ord, cal=w_cal)
        self.tau, self.n_levels = tau, n_levels

    def forward(self, *, slate_logits, rel, mask=None,
                teacher_probs=None, ordinal_logits=None, level_probs=None):
        out, total = {}, slate_logits.new_zeros(())

        out["pair"] = lambda_ranknet_loss(slate_logits, rel, mask)
        out["list"] = listnet_loss(slate_logits, rel, self.tau, mask)
        total = total + self.w["pair"] * out["pair"] + self.w["list"] * out["list"]

        if teacher_probs is not None:
            out["kd"] = distill_kl(slate_logits, teacher_probs, self.tau, mask)
            total = total + self.w["kd"] * out["kd"]
        if ordinal_logits is not None:
            out["ord"] = coral_loss(ordinal_logits, rel, self.n_levels)
            total = total + self.w["ord"] * out["ord"]
        if level_probs is not None:
            out["cal"] = rps_loss(level_probs, rel, self.n_levels)
            total = total + self.w["cal"] * out["cal"]

        out["total"] = total
        return total, out
