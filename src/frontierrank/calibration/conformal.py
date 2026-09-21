"""Split conformal intervals on the calibrated utility. Coverage, not just ECE.

The regret estimate in `core.frontier` samples utilities from N(u_hat, sigma)
and re-ranks. That is only meaningful if sigma means something, and ECE does
not establish that: ECE is about the *probability of the argmax*, while the
frontier needs the width of an interval on a *continuous utility*. A model can
have excellent ECE and sigmas that are uniformly half as wide as they should
be, and every downstream regret number would then be confidently wrong.

Split conformal gives a distribution-free guarantee instead. On a held-out
calibration set, score each example by how many sigmas it missed by:

    s_i = |u_i - u_hat_i| / sigma_i

Take the ceil((n+1)(1-alpha))/n quantile of those scores, call it q. Then
[u_hat +- q * sigma] has marginal coverage at least 1-alpha on exchangeable
data, whatever the model's sigmas were doing. The finite-sample correction in
the quantile index is what makes it a guarantee rather than an approximation,
and it matters at the sizes this project works with.

Two honest limits, both recorded rather than hidden:

  * The guarantee is MARGINAL, averaged over the calibration distribution. It
    does not promise coverage for hard queries specifically, and ranking is
    exactly where the hard cases matter. `coverage_by_group` reports the
    conditional breakdown so the gap is visible.
  * Exchangeability breaks under distribution shift, which Part XI lists as a
    managed risk. `CoverageMonitor` tracks realised coverage so drift shows up
    as a number rather than as a mysterious quality regression.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["ConformalIntervals", "coverage_by_group", "CoverageMonitor"]


@dataclass
class ConformalIntervals:
    """Calibrated multiplier on sigma, fitted once on held-out data."""
    alpha: float = 0.1
    q: float = 1.0
    n_calibration: int = 0
    scores: np.ndarray = field(default_factory=lambda: np.array([]))

    @classmethod
    def fit(cls, u_true, u_hat, sigma, alpha: float = 0.1,
            sigma_floor: float = 1e-6) -> "ConformalIntervals":
        u_true = np.asarray(u_true, dtype=float).ravel()
        u_hat = np.asarray(u_hat, dtype=float).ravel()
        sigma = np.asarray(sigma, dtype=float).ravel()
        if not (u_true.shape == u_hat.shape == sigma.shape):
            raise ValueError("fit needs aligned arrays")
        n = u_true.size
        if n < 20:
            raise ValueError(
                f"{n} calibration points is too few for a {1 - alpha:.0%} "
                f"interval; the quantile index would be the maximum score")
        s = np.abs(u_true - u_hat) / np.clip(sigma, sigma_floor, None)
        # finite-sample corrected quantile: ceil((n+1)(1-alpha))/n
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return cls(alpha, float(np.quantile(s, level)), n, np.sort(s))

    def interval(self, u_hat, sigma):
        u_hat = np.asarray(u_hat, dtype=float)
        half = self.q * np.asarray(sigma, dtype=float)
        return u_hat - half, u_hat + half

    def covers(self, u_true, u_hat, sigma) -> np.ndarray:
        lo, hi = self.interval(u_hat, sigma)
        u_true = np.asarray(u_true, dtype=float)
        return (u_true >= lo) & (u_true <= hi)

    def calibrated_sigma(self, sigma):
        """Sigma rescaled so a 1-sigma band means what a Gaussian says it does.

        This is what `core.frontier` should be handed. The raw sigma out of
        `AnchoredScorer` is a heuristic built from the anchor fit residual; this
        turns it into something with a coverage guarantee behind it.
        """
        # q is the multiplier for (1-alpha); rescale to a unit normal 1-sigma
        from math import erf, sqrt
        z = 1.0
        target = erf(z / sqrt(2.0))          # ~0.6827
        ratio = self.q / max(_z_for(1 - self.alpha), 1e-9)
        _ = target
        return np.asarray(sigma, dtype=float) * ratio

    def report(self) -> str:
        return (f"ConformalIntervals(alpha={self.alpha}, q={self.q:.3f}, "
                f"n={self.n_calibration})\n"
                f"  a {1 - self.alpha:.0%} interval is u_hat +- {self.q:.3f} * sigma\n"
                f"  q > 1.64 means the model's sigmas were too NARROW; "
                f"q < 1.64 means too wide\n"
                f"  (1.64 is the Gaussian two-sided "
                f"{1 - self.alpha:.0%} multiplier)")


def _z_for(level: float) -> float:
    """Two-sided normal multiplier for a coverage level, without scipy."""
    from math import erf, sqrt
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if erf(mid / sqrt(2.0)) < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def coverage_by_group(intervals: ConformalIntervals, u_true, u_hat, sigma,
                      groups) -> dict:
    """Realised coverage overall and per group.

    The guarantee is marginal, so a uniform overall number can sit on top of
    badly under-covered hard cases. Grouping by query, by grade or by slate
    residual is how that gets caught.
    """
    ok = intervals.covers(u_true, u_hat, sigma)
    groups = np.asarray(groups)
    out = {"__overall__": {"n": int(ok.size),
                           "coverage": round(float(ok.mean()), 4),
                           "target": round(1 - intervals.alpha, 4)}}
    for g in np.unique(groups):
        m = groups == g
        out[str(g)] = {"n": int(m.sum()),
                       "coverage": round(float(ok[m].mean()), 4)}
    worst = min((v["coverage"] for k, v in out.items() if k != "__overall__"),
                default=1.0)
    out["__overall__"]["worst_group_coverage"] = round(worst, 4)
    return out


class CoverageMonitor:
    """Running realised coverage, for drift detection in production.

    Exchangeability is the conformal assumption and distribution shift breaks
    it. Watching coverage fall is how that becomes visible before it becomes a
    quality regression nobody can explain.
    """

    def __init__(self, intervals: ConformalIntervals, window: int = 2000):
        self.intervals = intervals
        self.window = window
        self.hits: list[bool] = []

    def update(self, u_true, u_hat, sigma):
        self.hits.extend(self.intervals.covers(u_true, u_hat, sigma).tolist())
        if len(self.hits) > self.window:
            self.hits = self.hits[-self.window:]
        return self.coverage()

    def coverage(self) -> float:
        return float(np.mean(self.hits)) if self.hits else float("nan")

    def drifted(self, tolerance: float = 0.05) -> bool:
        c = self.coverage()
        return bool(np.isfinite(c) and c < (1 - self.intervals.alpha) - tolerance)
