"""The top-k regret frontier.

The object the controller reasons about is not a score. It is the set of
candidates whose uncertainty can still change the requested top-k.

Given calibrated utilities u_i with standard errors s_i, candidate i is on the
frontier if its plausible interval overlaps the interval of the candidate
currently sitting at rank K. Everything strictly above stays in; everything
strictly below stays out; nothing you spend on them changes nDCG@K.

`expected_regret` estimates what nDCG@K you lose by stopping now. That number,
not a confidence threshold, is what the controller compares against the price
of a teacher call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Frontier", "compute_frontier", "expected_regret", "swap_probability"]


def _gain(rel):
    return np.exp2(np.asarray(rel, dtype=float)) - 1.0


def _discount(n):
    return 1.0 / np.log2(np.arange(2, n + 2))


@dataclass
class Frontier:
    members: np.ndarray        # candidate indices whose rank-K membership is in doubt
    boundary_rank: int         # K
    order: np.ndarray          # current ranking, best first
    margin: np.ndarray         # per-candidate |u_i - u_K| / joint sigma
    p_swap: np.ndarray         # per-candidate P(crosses the K boundary)

    @property
    def size(self) -> int:
        return int(self.members.size)

    def head(self, m: int) -> np.ndarray:
        """The m most contested members -- what you actually send to the teacher."""
        if self.size == 0:
            return np.array([], dtype=int)
        o = np.argsort(-self.p_swap[self.members])
        return self.members[o[:m]]


def swap_probability(u: np.ndarray, sigma: np.ndarray, k: int) -> np.ndarray:
    """P(candidate i ends up on the other side of the rank-k boundary).

    Gaussian approximation: the boundary is the k-th order statistic, which we
    approximate by the current k-th ranked candidate's utility and its own
    uncertainty. Crude, but it is a probability, and it is calibrated enough to
    rank actions by -- which is all the controller needs.
    """
    u = np.asarray(u, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = u.size
    if k >= n:
        return np.zeros(n)
    order = np.argsort(-u)
    kth = order[k - 1]
    d = u - u[kth]
    joint = np.sqrt(np.clip(sigma**2 + sigma[kth] ** 2, 1e-12, None))
    z = d / joint
    # P(wrong side): above the line -> P(falls below); below -> P(rises above)
    from math import erf, sqrt
    ncdf = np.vectorize(lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0))))
    p = np.where(d >= 0, ncdf(-z), ncdf(z))
    p[kth] = 0.5
    return np.clip(p, 0.0, 1.0)


def compute_frontier(u, sigma, k: int = 10, z: float = 1.64,
                     p_floor: float = 0.02) -> Frontier:
    """Members are candidates whose interval overlaps the rank-k candidate's.

    z=1.64 is a 90% one-sided band. p_floor additionally admits anything with a
    non-trivial swap probability, which catches heavy-tailed cases the Gaussian
    band misses.
    """
    u = np.asarray(u, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    order = np.argsort(-u)
    n = u.size
    if k >= n:
        members = np.arange(n)
        return Frontier(members, k, order, np.zeros(n), np.full(n, 0.5))

    kth = order[k - 1]
    lo_i, hi_i = u - z * sigma, u + z * sigma
    lo_k, hi_k = u[kth] - z * sigma[kth], u[kth] + z * sigma[kth]
    overlap = (hi_i >= lo_k) & (lo_i <= hi_k)

    p = swap_probability(u, sigma, k)
    members = np.flatnonzero(overlap | (p >= p_floor))
    joint = np.sqrt(np.clip(sigma**2 + sigma[kth] ** 2, 1e-12, None))
    margin = np.abs(u - u[kth]) / joint
    return Frontier(members, k, order, margin, p)


def expected_regret(u, sigma, rel_hat, k: int = 10, n_samples: int = 256,
                    rng: np.random.Generator | None = None) -> float:
    """E[nDCG@k(oracle) - nDCG@k(current)] under the model's own uncertainty.

    rel_hat is the expected graded relevance per candidate (e.g. from the
    ordinal head). We sample utilities, re-rank, and compare the nDCG of the
    ranking we would ship against the nDCG of the ranking that sample implies.
    """
    rng = rng or np.random.default_rng()
    u = np.asarray(u, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    rel = np.asarray(rel_hat, dtype=float)
    n = u.size
    k = min(k, n)
    disc = _discount(k)

    shipped = np.argsort(-u)[:k]
    ideal = np.sort(_gain(rel))[::-1][:k]
    idcg = float(np.sum(ideal * disc))
    if idcg <= 0:
        return 0.0

    dcg_shipped = float(np.sum(_gain(rel[shipped]) * disc))
    draws = u[None, :] + rng.normal(0.0, 1.0, size=(n_samples, n)) * sigma[None, :]
    top = np.argsort(-draws, axis=1)[:, :k]
    dcg_alt = np.sum(_gain(rel)[top] * disc[None, :], axis=1)
    return float(np.mean(np.maximum(dcg_alt - dcg_shipped, 0.0)) / idcg)
