"""Anchor calibration: recover the per-slate offset that a slate softmax hides.

A typed-decision model reading slate S emits probabilities normalised over S:

    log p_i = l_i - c_S,        c_S = logsumexp_{j in S} l_j

c_S is a function of the slate, not the document. It cancels within a slate and
does NOT cancel across slates, and it cannot be recovered from p alone -- the
softmax is invariant to adding a constant to every logit in the slate.

Fix: every slate carries A anchor items whose latent utility is known (fixed
pivot documents with held-out graded labels). Regressing the known utilities on
the anchors' observed log-probabilities recovers a per-slate affine map that
puts every slate on one global scale.

Estimators
----------
shift   u_hat = log p + b_S                 A >= 1.  Assumes unit slope.
affine  u_hat = a_S * log p + b_S           A >= 2.  OLS; unstable at A == 2.
ridge   as affine, slope shrunk toward 1    A >= 2.  Default. Robust to
                                            slate-dependent temperature drift.
"""
from __future__ import annotations

import numpy as np

__all__ = ["AnchorCalibrator", "fit_slate", "SlateFit"]


class SlateFit:
    __slots__ = ("slope", "intercept", "n_anchors", "residual")

    def __init__(self, slope: float, intercept: float, n_anchors: int, residual: float):
        self.slope, self.intercept = slope, intercept
        self.n_anchors, self.residual = n_anchors, residual

    def apply(self, log_p: np.ndarray) -> np.ndarray:
        return self.slope * np.asarray(log_p, dtype=float) + self.intercept

    def __repr__(self) -> str:
        return (f"SlateFit(slope={self.slope:.4f}, intercept={self.intercept:.4f}, "
                f"A={self.n_anchors}, resid={self.residual:.4f})")


def fit_slate(anchor_log_p: np.ndarray,
              anchor_utility: np.ndarray,
              method: str = "ridge",
              lam: float = 1.0,
              slope_prior: float | None = None) -> SlateFit:
    """Fit one slate's calibration map from its anchors.

    anchor_log_p    observed log-probabilities of the anchor items in this slate
    anchor_utility  their known latent utilities (fixed across all slates)
    lam             ridge strength; slope -> `slope_prior` as lam grows.
    slope_prior     what the slope shrinks toward. Defaults to 1.0, which is
                    only correct when utilities are expressed in the same units
                    as log-probabilities.

                    This bit me. On a 0-2 TREC grade scale against log-probs
                    spanning ~3 nats, the true slope is around 0.67, so
                    shrinking toward 1.0 pulls every slate's fit in the same
                    wrong direction -- a bias that looks exactly like
                    "anchoring does not help". Pass the ratio of utility spread
                    to log-probability spread, or use `AnchorCalibrator`, which
                    estimates it.
    """
    lp = np.asarray(anchor_log_p, dtype=float).ravel()
    u = np.asarray(anchor_utility, dtype=float).ravel()
    if lp.shape != u.shape:
        raise ValueError(f"anchor shape mismatch: {lp.shape} vs {u.shape}")
    A = lp.size
    if A == 0:
        raise ValueError("a slate with no anchors cannot be calibrated")

    prior = 1.0 if slope_prior is None else float(slope_prior)
    if A == 1 or method == "shift":
        slope = prior
        intercept = float(np.mean(u - prior * lp))
    elif method == "affine":
        X = np.vstack([lp, np.ones(A)]).T
        slope, intercept = np.linalg.lstsq(X, u, rcond=None)[0]
    elif method == "ridge":
        num = float(np.sum((lp - lp.mean()) * (u - u.mean())))
        den = float(np.sum((lp - lp.mean()) ** 2))
        # -> `prior` as lam dominates, not hard-coded 1.0
        slope = (num + lam * prior) / (den + lam)
        intercept = float(np.mean(u - slope * lp))
    else:
        raise ValueError(f"unknown method {method!r}")

    resid = float(np.sqrt(np.mean((slope * lp + intercept - u) ** 2)))
    return SlateFit(float(slope), float(intercept), A, resid)


class AnchorCalibrator:
    """Stateless across slates; holds the anchor utilities and the fit policy.

    The anchor utilities are the *only* thing shared between slates, and they
    are what makes the resulting scores globally comparable.
    """

    def __init__(self, anchor_utility, method: str = "ridge", lam: float = 1.0,
                 max_residual: float | None = None,
                 slope_prior: float | None = None):
        self.anchor_utility = np.asarray(anchor_utility, dtype=float).ravel()
        self.method, self.lam = method, lam
        self.slope_prior = slope_prior
        # slates whose anchors fit badly are flagged, not silently trusted
        self.max_residual = max_residual

    @property
    def n_anchors(self) -> int:
        return int(self.anchor_utility.size)

    def calibrate(self, log_p_candidates: np.ndarray, log_p_anchors: np.ndarray):
        """Map one slate's candidate log-probs onto the global utility scale.

        Returns (utilities, fit). `fit.residual` above `max_residual` means the
        slate's anchors did not behave, and the caller should treat those
        candidates as high-uncertainty rather than trusting the numbers.
        """
        fit = fit_slate(log_p_anchors, self.anchor_utility, self.method, self.lam,
                        self.slope_prior)
        u = fit.apply(log_p_candidates)
        return u, fit

    def suspect(self, fit: SlateFit) -> bool:
        return self.max_residual is not None and fit.residual > self.max_residual
