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
           "distill_kl", "coral_loss", "rps_loss", "CompositeRankingLoss",
           "anchor_monotone_loss", "anchor_affine_loss"]


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
    """Cross-entropy between the score softmax and the gain softmax (top-1 PL).

    Masking here has to be done on the TARGET as well as the scores, and it is
    easy to get wrong in a way that does not crash. Filling `rel` with -1e4 and
    then clamping at 0 leaves masked positions with gain 2^0 - 1 = 0, which is
    a finite logit, so they keep a share of the target softmax -- while their
    log-probability has been pushed to -1e4. The product is then a loss in the
    hundreds that is pure masking artefact, and it silently dominates training.

    Anchors are masked positions in every slate here, so this is not a corner
    case: it is every batch.
    """
    p = F.log_softmax(scores.masked_fill(~mask.bool(), -1e4) / tau
                      if mask is not None else scores / tau, dim=-1)
    g = _gain(rel.clamp_min(0)) / tau
    if mask is None:
        return -(F.softmax(g, dim=-1) * p).sum(-1).mean()
    m = mask.bool()
    q = F.softmax(g.masked_fill(~m, -1e4), dim=-1) * m.float()
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-9)
    return -(q * p.masked_fill(~m, 0.0)).sum(-1).mean()


def distill_kl(student_logits, teacher_probs, tau: float = 1.0, mask=None):
    """KL(teacher || student) over the slate. The core of frontier distillation."""
    if mask is not None:
        student_logits = student_logits.masked_fill(~mask.bool(), -1e4)
    logp = F.log_softmax(student_logits / tau, dim=-1)
    t = teacher_probs.clamp_min(1e-8)
    t = t / t.sum(-1, keepdim=True)
    return (t * (t.log() - logp)).sum(-1).mean() * (tau ** 2)


def coral_loss(ordinal_logits, rel, n_levels: int = 4, mask=None):
    """CORAL: K-1 shared-slope binary tasks, P(y > k). Keeps ranks monotone.

    `mask` excludes positions with no graded label. Without it the anchors --
    which carry rel = -1 as a sentinel -- are trained as genuine grade-0
    documents, teaching the ordinal head that a direct-answer pivot is
    irrelevant. That is the opposite of what the pivots are for.
    """
    tgt = torch.stack([(rel > k).float() for k in range(n_levels - 1)], dim=-1)
    per = F.binary_cross_entropy_with_logits(ordinal_logits, tgt, reduction="none")
    if mask is None:
        return per.mean()
    m = mask.bool().unsqueeze(-1).expand_as(per)
    return (per * m).sum() / m.sum().clamp_min(1)


def rps_loss(probs, rel, n_levels: int = 4, mask=None):
    """Ranked probability score -- strictly proper AND ordinal-aware.

    This is the rule Laya's own RLCD applies to `score` questions
    (w_rps = 1.0). Being off by one level should cost less than being off by
    three; log-loss alone does not know that.
    """
    cdf_p = torch.cumsum(probs, dim=-1)
    onehot = F.one_hot(rel.long().clamp(0, n_levels - 1), n_levels).float()
    cdf_t = torch.cumsum(onehot, dim=-1)
    per = ((cdf_p - cdf_t) ** 2).sum(-1) / (n_levels - 1)
    if mask is None:
        return per.mean()
    m = mask.bool()
    return (per * m).sum() / m.sum().clamp_min(1)


class CompositeRankingLoss(torch.nn.Module):
    def __init__(self, w_point=0.5, w_pair=1.0, w_list=1.0, w_kd=2.0,
                 w_ord=0.5, w_cal=1.0, w_anchor=1.0, w_anchor_mono=0.5,
                 tau=1.0, n_levels=4):
        super().__init__()
        self.w = dict(point=w_point, pair=w_pair, list=w_list, kd=w_kd,
                      ord=w_ord, cal=w_cal, anchor=w_anchor,
                      anchor_mono=w_anchor_mono)
        self.tau, self.n_levels = tau, n_levels

    def forward(self, *, slate_logits, rel, mask=None,
                teacher_probs=None, ordinal_logits=None, level_probs=None,
                anchor_utility=None, is_anchor=None):
        out, total = {}, slate_logits.new_zeros(())

        out["pair"] = lambda_ranknet_loss(slate_logits, rel, mask)
        out["list"] = listnet_loss(slate_logits, rel, self.tau, mask)
        total = total + self.w["pair"] * out["pair"] + self.w["list"] * out["list"]

        if teacher_probs is not None:
            out["kd"] = distill_kl(slate_logits, teacher_probs, self.tau, mask)
            total = total + self.w["kd"] * out["kd"]
        if ordinal_logits is not None:
            out["ord"] = coral_loss(ordinal_logits, rel, self.n_levels, mask)
            total = total + self.w["ord"] * out["ord"]
        if level_probs is not None:
            out["cal"] = rps_loss(level_probs, rel, self.n_levels, mask)
            total = total + self.w["cal"] * out["cal"]

        # anchors are the ruler, not the thing measured: they are excluded
        # from `rel` by the caller's mask and enter only through these terms
        if is_anchor is not None and anchor_utility is not None and is_anchor.any():
            out["anchor"] = anchor_affine_loss(slate_logits, anchor_utility, is_anchor)
            out["anchor_mono"] = anchor_monotone_loss(slate_logits, anchor_utility,
                                                      is_anchor)
            total = (total + self.w["anchor"] * out["anchor"]
                     + self.w["anchor_mono"] * out["anchor_mono"])

        out["total"] = total
        return total, out


# ==========================================================================
# Anchor alignment -- not in plan.md, and arguably the point of the whole thing
# ==========================================================================
#
# The plan treats anchoring as an inference-time correction: score the slate,
# then regress the pivots to recover c_S. Nothing in the training objective
# asks the model to make that regression *work*.
#
# It should. The affine recovery is only as good as the relationship between a
# pivot's known utility and its observed logit, and that relationship is a
# property of the weights. F8 found the recovery underpowered at realistic
# noise levels; training for it attacks that directly, where adding more pivots
# only buys sqrt(A).
#
# Two terms, both cheap because the anchors are already in every slate:
#
#   monotone   pairwise hinge: a pivot with higher known utility must get a
#              higher logit. This is what makes the pivots a usable ruler.
#   affine     penalise the residual of the least-squares fit from pivot logits
#              to pivot utilities. This is literally the quantity the inference
#              calibrator minimises, so optimising it trains the model for the
#              estimator it will actually be scored with.

def anchor_monotone_loss(slate_logits, anchor_utility, is_anchor, margin=0.5):
    """Pivots must be ordered by their known utility, with a margin.

    Without this the pivots can be ordered correctly on average and scrambled
    on any given slate, which is exactly what makes a per-slate affine fit
    noisy -- and the fit is per-slate.
    """
    u = anchor_utility.masked_fill(~is_anchor, float("nan"))
    du = u.unsqueeze(-1) - u.unsqueeze(-2)          # u_i - u_j
    ds = slate_logits.unsqueeze(-1) - slate_logits.unsqueeze(-2)
    pair = is_anchor.unsqueeze(-1) & is_anchor.unsqueeze(-2)
    pair = pair & torch.isfinite(du) & (du > 0)
    if not pair.any():
        return slate_logits.new_zeros(())
    viol = F.relu(margin - ds)[pair]
    return viol.mean()


def anchor_affine_loss(slate_logits, anchor_utility, is_anchor, eps=1e-6):
    """Residual of the per-slate least-squares fit  u ~ a * logit + b.

    This is the inference-time estimator's own objective. Minimising it during
    training is the difference between "the pivots happen to be informative"
    and "the model was trained to make them informative".

    Scale-free: the residual is normalised by the pivots' utility variance, so
    a slate whose pivots span a wider range is not penalised for it.
    """
    total = slate_logits.new_zeros(())
    n = 0
    for b in range(slate_logits.shape[0]):
        m = is_anchor[b]
        if int(m.sum()) < 2:
            continue
        x = slate_logits[b][m].float()
        y = anchor_utility[b][m].float()
        if not torch.isfinite(y).all():
            continue
        xc, yc = x - x.mean(), y - y.mean()
        denom = (xc * xc).sum().clamp_min(eps)
        slope = (xc * yc).sum() / denom
        resid = yc - slope * xc
        total = total + (resid * resid).mean() / (yc * yc).mean().clamp_min(eps)
        n += 1
    return total / max(n, 1)
