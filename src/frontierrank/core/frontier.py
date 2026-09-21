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


def expected_regret(u, sigma, rel_hat=None, k: int = 10, n_samples: int = 256,
                    rng: np.random.Generator | None = None,
                    rel_from_utility=None) -> float:
    """E[nDCG@k lost by shipping the current order instead of the true one].

    The definition matters, and the obvious implementation is wrong. Scoring
    both the shipped ranking and each sampled alternative with the *point
    estimate* of relevance makes the shipped ranking optimal by construction --
    it is the argsort of the very quantity being scored -- so the difference is
    never positive and the regret is identically zero. A controller reading
    that number can never justify buying anything.

    The correct quantity treats each draw as a hypothesis about the truth:

        regret = E_{u~ ~ N(u_hat, sigma)} [ nDCG@k(rank by u~; truth u~)
                                          - nDCG@k(rank by u_hat; truth u~) ]

    Under any draw, ranking by that draw is optimal, so the gap is >= 0, and it
    shrinks to 0 as sigma -> 0. That is what "how much might my uncertainty be
    costing me?" actually means.

    `rel_from_utility` maps a utility onto graded relevance for the gain
    function; the default clips at zero, which keeps 2^rel - 1 well behaved and
    treats negative-utility candidates as irrelevant rather than as harmful.
    `rel_hat` is accepted for callers that have a separate relevance estimate
    (an ordinal head); it only sets the SCALE of the gain, never the ordering,
    because the ordering is what is uncertain.
    """
    rng = rng or np.random.default_rng()
    u = np.asarray(u, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = u.size
    k = min(k, n)
    if k < 1:
        return 0.0
    disc = _discount(k)
    to_rel = rel_from_utility or (lambda x: np.clip(x, 0.0, None))

    if rel_hat is not None:
        # rescale utilities onto the caller's relevance range, preserving order
        rel_hat = np.asarray(rel_hat, dtype=float)
        lo, hi = float(np.min(rel_hat)), float(np.max(rel_hat))
        span = max(hi - lo, 1e-9)
        u_lo, u_hi = float(np.min(u)), float(np.max(u))
        u_span = max(u_hi - u_lo, 1e-9)

        def to_rel(x, _lo=lo, _span=span, _ulo=u_lo, _uspan=u_span):
            return _lo + (np.asarray(x) - _ulo) / _uspan * _span

    shipped = np.argsort(-u)[:k]
    draws = u[None, :] + rng.normal(0.0, 1.0, size=(n_samples, n)) * sigma[None, :]

    losses = np.empty(n_samples)
    for i in range(n_samples):
        rel = _gain(to_rel(draws[i]))
        best = np.argsort(-draws[i])[:k]
        idcg = float(np.sum(np.sort(rel)[::-1][:k] * disc))
        if idcg <= 0:
            losses[i] = 0.0
            continue
        losses[i] = (float(np.sum(rel[best] * disc))
                     - float(np.sum(rel[shipped] * disc))) / idcg
    return float(np.mean(np.maximum(losses, 0.0)))
