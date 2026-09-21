"""Slate construction under a hard option-slot budget.

Laya ships `temperature_by_options` with `choice:11+ -> T = 0.1006`, a 10x
sharpening that only fits if the raw logits are near-uniform past ~10 options.
Treat M = 10 total option slots as a hard cap for the local model. Anchors
occupy slots candidates could have used, so the budget is:

    candidates per slate = M - A
    slates per query     = ceil(N / (M - A))

Slate assignment matters more than it looks. Slating by first-stage rank
("blocked") is the obvious choice and is the WORST case for an uncalibrated
slate softmax, because slate composition then correlates with quality and the
hidden offset c_S varies maximally between slates. Shuffling hides most of the
bug without fixing it -- which is probably why nobody has written it down.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Slate", "SlatePlan", "build_slates", "slates_per_query"]


def slates_per_query(n_candidates: int, max_options: int = 10, n_anchors: int = 2) -> int:
    per = max_options - n_anchors
    if per < 1:
        raise ValueError("anchors consume the entire slate budget")
    return math.ceil(n_candidates / per)


@dataclass
class Slate:
    candidate_ids: list           # indices into the candidate pool
    anchor_ids: list              # indices into the anchor pool
    index: int = 0

    @property
    def n_options(self) -> int:
        return len(self.candidate_ids) + len(self.anchor_ids)


@dataclass
class SlatePlan:
    slates: list = field(default_factory=list)
    max_options: int = 10
    n_anchors: int = 2

    @property
    def n_passes(self) -> int:
        return len(self.slates)

    def validate(self) -> None:
        for s in self.slates:
            if s.n_options > self.max_options:
                raise ValueError(
                    f"slate {s.index} has {s.n_options} options, cap is {self.max_options}")
            if len(s.anchor_ids) != self.n_anchors:
                raise ValueError(f"slate {s.index} carries {len(s.anchor_ids)} anchors")


def build_slates(n_candidates: int,
                 n_anchor_pool: int,
                 max_options: int = 10,
                 n_anchors: int = 2,
                 order: np.ndarray | None = None,
                 stratify: bool = True,
                 rng: np.random.Generator | None = None) -> SlatePlan:
    """Partition candidates into slates, each carrying `n_anchors` pivots.

    order      candidate indices in first-stage rank order (best first).
    stratify   interleave first-stage ranks across slates instead of blocking
               them. This does not fix the offset problem -- calibration does --
               but it reduces the variance the calibrator has to absorb, and it
               keeps every slate's difficulty roughly comparable.
    """
    rng = rng or np.random.default_rng()
    if order is None:
        order = np.arange(n_candidates)
    order = np.asarray(order)
    per = max_options - n_anchors
    if per < 1:
        raise ValueError("anchors consume the entire slate budget")
    n_slates = math.ceil(n_candidates / per)

    if stratify:
        # round-robin deal: slate k gets ranks k, k+n_slates, k+2*n_slates, ...
        groups = [order[k::n_slates].tolist() for k in range(n_slates)]
    else:
        groups = [order[i:i + per].tolist() for i in range(0, n_candidates, per)]

    if n_anchors > n_anchor_pool:
        raise ValueError(f"need {n_anchors} anchors, pool holds {n_anchor_pool}")

    slates = []
    for k, g in enumerate(groups):
        if not g:
            continue
        # rotate through the anchor pool so no single pivot dominates the fit
        picks = [(k * n_anchors + j) % n_anchor_pool for j in range(n_anchors)]
        slates.append(Slate(candidate_ids=g, anchor_ids=picks, index=k))

    plan = SlatePlan(slates=slates, max_options=max_options, n_anchors=n_anchors)
    plan.validate()
    return plan
