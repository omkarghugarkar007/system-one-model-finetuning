"""Anchored slates -> one globally comparable utility per candidate.

This is the scoring substrate. It takes a pool of candidate signatures and a
first-stage order, deals them into slates that each carry a fixed set of pivot
documents, scores every slate, and uses the pivots to put every slate's
candidates on one utility scale.

Three things here that the simulation did not need and real text does:

1. **Layout.** `core.budget` shows a Laya slate cannot carry rich candidate
   text in its options. The packer decides whether candidates live in the
   options or in the state, and it is a measured choice (`experiments/phase0`).

2. **Anchor positions are shuffled.** Putting the pivots last in every slate
   would confound any residual position effect with the anchor/candidate
   distinction -- and that is exactly the fit everything downstream depends on.
   Positions are permuted with a seeded RNG and unpermuted after scoring.

3. **Batching.** Laya does not share state across questions, so a 100-candidate
   query is 17 independent 512-token sequences. A scorer exposing `score_many`
   gets them in one forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .calibrate import AnchorCalibrator
from .packing import DEFAULT_INSTRUCTIONS, OptionsPacker, PackedSlate, SlatePacker
from .slates import SlatePlan, build_slates

__all__ = ["Anchor", "ScoredPool", "AnchoredScorer"]


@dataclass
class Anchor:
    """A pivot document with a known utility on the global scale.

    Pick pivots that span the utility range, hold them out of every qrel, and
    keep the set fixed across the whole corpus. Their utilities are the only
    thing shared between slates, and therefore the only thing making slates
    comparable at all.
    """
    text: str
    utility: float
    grade: int | None = None
    doc_id: str = ""


@dataclass
class ScoredPool:
    utility: np.ndarray                 # global scale, comparable across slates
    sigma: np.ndarray                   # per-candidate standard error
    raw_logp: np.ndarray                # uncalibrated, for the naive ablation
    plan: SlatePlan | None = None
    fits: list = field(default_factory=list)
    suspect_slates: list = field(default_factory=list)
    slate_of: np.ndarray | None = None  # which slate scored each candidate
    anchor_logp: dict = field(default_factory=dict)   # slate -> anchor log-probs
    pack_info: list = field(default_factory=list)

    @property
    def order(self) -> np.ndarray:
        return np.argsort(-self.utility)

    @property
    def naive_order(self) -> np.ndarray:
        """Ranking by raw slate log-probability -- the bug, for the ablation."""
        return np.argsort(-self.raw_logp)


class AnchoredScorer:
    def __init__(self, model, anchors: Sequence[Anchor],
                 packer: SlatePacker | None = None,
                 max_options: int = 10, n_anchors: int = 4,
                 method: str = "ridge", lam: float = 1.0,
                 base_sigma: float = 0.35, max_residual: float = 0.75,
                 stratify: bool = True, shuffle_positions: bool = True,
                 instructions: str = DEFAULT_INSTRUCTIONS):
        if n_anchors > len(anchors):
            raise ValueError(f"n_anchors={n_anchors} exceeds anchor pool {len(anchors)}")
        if max_options - n_anchors < 1:
            raise ValueError("anchors consume the entire slate budget")
        self.model = model
        self.anchors = list(anchors)
        self.packer = packer or OptionsPacker(instructions=instructions)
        self.max_options, self.n_anchors = max_options, n_anchors
        self.base_sigma, self.stratify = base_sigma, stratify
        self.shuffle_positions = shuffle_positions
        self.method, self.lam, self.max_residual = method, lam, max_residual

    # ------------------------------------------------------------------ score
    def score(self, query: str, signatures: Sequence[str],
              first_stage_order: np.ndarray | None = None,
              rng: np.random.Generator | None = None) -> ScoredPool:
        rng = rng or np.random.default_rng(0)
        n = len(signatures)
        plan = build_slates(n, len(self.anchors), self.max_options,
                            self.n_anchors, first_stage_order,
                            stratify=self.stratify, rng=rng)

        # --- pack every slate, recording where the anchors landed
        packed: list[PackedSlate] = []
        placements = []          # (slate, perm, n_cands)
        for s in plan.slates:
            anc = [self.anchors[i] for i in s.anchor_ids]
            texts = [signatures[i] for i in s.candidate_ids] + [a.text for a in anc]
            perm = (rng.permutation(len(texts)) if self.shuffle_positions
                    else np.arange(len(texts)))
            packed.append(self.packer.pack(query, [texts[p] for p in perm]))
            placements.append((s, perm, len(s.candidate_ids)))

        # --- one batched call if the model offers it
        if hasattr(self.model, "score_many"):
            logps = self.model.score_many(packed)
        else:
            logps = [self.model.choice_logprobs(p.instructions, p.options, p.state)
                     for p in packed]

        # --- calibrate each slate against its own anchors
        u = np.full(n, np.nan)
        sig = np.full(n, self.base_sigma)
        raw = np.full(n, np.nan)
        slate_of = np.full(n, -1, dtype=int)
        fits, suspect, anchor_logp = [], [], {}

        for (s, perm, k), p, lp in zip(placements, packed, logps):
            lp = np.asarray(lp, dtype=float)
            if lp.shape[0] != len(perm):
                raise ValueError(f"scorer returned {lp.shape[0]} logprobs for "
                                 f"{len(perm)} options in slate {s.index}")
            # undo the position shuffle: original index j sat at perm-position i
            inv = np.empty_like(perm)
            inv[perm] = np.arange(len(perm))
            lp_orig = lp[inv]
            lp_c, lp_a = lp_orig[:k], lp_orig[k:]

            anc_u = [self.anchors[i].utility for i in s.anchor_ids]
            cal = AnchorCalibrator(anc_u, method=self.method, lam=self.lam,
                                   max_residual=self.max_residual)
            vals, fit = cal.calibrate(lp_c, lp_a)

            idx = np.asarray(s.candidate_ids)
            u[idx] = vals
            raw[idx] = lp_c
            slate_of[idx] = s.index
            # a badly fitting slate is not a confident slate: inflate sigma
            # rather than silently trusting numbers whose pivots misbehaved
            sig[idx] = self.base_sigma * abs(fit.slope) * (1.0 + fit.residual)
            fits.append(fit)
            anchor_logp[s.index] = lp_a
            if cal.suspect(fit):
                suspect.append(s.index)

        if np.isnan(u).any():
            raise RuntimeError("some candidates were never placed in a slate")
        return ScoredPool(u, sig, raw, plan, fits, suspect, slate_of,
                          anchor_logp, [p.info for p in packed])
