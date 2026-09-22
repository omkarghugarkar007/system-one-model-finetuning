"""Phase 0 -- the go/no-go gate. Does the slate softmax behave as modelled?

Everything in FrontierRank rests on one assumption: that Laya's marker softmax
is a Luce model, so the observed log-probability of candidate i in slate S is

    log p_{i,S} = l_i - c_S

with l_i a property of the candidate alone and c_S a property of the slate
alone. If that holds, anchors identify c_S and every slate lands on one scale.
If a candidate's score depends on *which* other candidates share its slate --
not merely how many or how good -- then no affine correction can recover it and
the architecture does not survive.

Five tests, cheapest first.

  T0 determinism    Is the model a function? Establishes whether repeated
                    measurement noise exists at all.
  T1 order noise    Permute the options of a fixed slate. Laya is deterministic,
                    so the plan's "marker noise" is not a real quantity -- order
                    variation is the actual measurement floor, and every later
                    residual has to be judged against it.
  T2 additive fit   THE TEST. Fit y_{i,S} = l_i - c_S by least squares over an
                    incomplete block design and compare the residual against the
                    T1 floor. Residual ~ floor: the model is as good as the
                    measurement allows. Residual >> floor: composition matters
                    beyond an additive term, and anchoring is weaker than the
                    simulation claims.
  T3 interaction    If T2 leaves structure, is it systematic? Regress the
                    residual on candidate grade x slate mean grade. A
                    significant interaction is the specific failure Part XI
                    names as fatal.
  T4 anchor recovery  Falsification test 2: do known-grade pivots actually
                    recover c_S, and are the residuals flat or curved? Curved
                    residuals mean the affine map needs a monotone spline.

T2 is the one that decides whether the rest of the project is worth building.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["BlockDesign", "make_block_design", "fit_additive", "AdditiveFit",
           "order_noise_floor", "interaction_test", "anchor_recovery"]


# --------------------------------------------------------------------- design
@dataclass
class BlockDesign:
    """Which probe candidates appear in which slates.

    Composition is varied on purpose: slates drawn only from high-grade
    candidates have a very different c_S from slates drawn only from low-grade
    ones, and if c_S never varies the test has no power.
    """
    slates: list[list[int]]          # slate -> probe indices
    probe_grades: np.ndarray
    n_probes: int

    @property
    def n_slates(self) -> int:
        return len(self.slates)

    def appearances(self) -> np.ndarray:
        c = np.zeros(self.n_probes, dtype=int)
        for s in self.slates:
            c[np.asarray(s)] += 1
        return c

    def is_connected(self) -> bool:
        """The additive model is only identifiable on a connected design.

        Two disjoint groups of slates sharing no candidate can each be shifted
        by a different constant without changing the fit, so `l_i` would not be
        comparable across them -- exactly the pathology the whole project is
        about, appearing in the test design itself.
        """
        if not self.slates:
            return False
        seen_p, seen_s = set(self.slates[0]), {0}
        changed = True
        while changed:
            changed = False
            for k, s in enumerate(self.slates):
                if k in seen_s:
                    continue
                if seen_p & set(s):
                    seen_s.add(k)
                    seen_p |= set(s)
                    changed = True
        return len(seen_s) == self.n_slates


def make_block_design(probe_grades, slate_size: int = 10, n_slates: int = 60,
                      rng: np.random.Generator | None = None,
                      homogeneous_frac: float = 0.5) -> BlockDesign:
    """Slates with deliberately varied composition, every probe seen several times.

    `homogeneous_frac` of the slates are drawn from a single grade where
    possible, which is what makes c_S swing. The rest are mixed.
    """
    rng = rng or np.random.default_rng(0)
    g = np.asarray(probe_grades)
    n = g.size
    by_grade = {int(v): np.flatnonzero(g == v) for v in np.unique(g)}
    slates = []
    for k in range(n_slates):
        if rng.random() < homogeneous_frac:
            pool = by_grade[int(rng.choice(list(by_grade)))]
            if pool.size < slate_size:
                pool = np.arange(n)
        else:
            pool = np.arange(n)
        slates.append(sorted(rng.choice(pool, size=min(slate_size, pool.size),
                                        replace=False).tolist()))
    return BlockDesign(slates, g, n)


# ------------------------------------------------------------- additive model
@dataclass
class AdditiveFit:
    l_hat: np.ndarray          # per-candidate logit, up to a global constant
    c_hat: np.ndarray          # per-slate offset, up to the same constant
    residuals: np.ndarray      # one per observation
    rmse: float
    r2: float
    obs_probe: np.ndarray
    obs_slate: np.ndarray
    y: np.ndarray

    def summary(self) -> dict:
        return {"rmse": self.rmse, "r2": self.r2,
                "n_obs": int(self.y.size),
                "c_range": float(np.ptp(self.c_hat)),
                "c_std": float(np.std(self.c_hat)),
                "l_std": float(np.std(self.l_hat)),
                "resid_max_abs": float(np.max(np.abs(self.residuals)))}


def fit_additive(design: BlockDesign, logp: list[np.ndarray]) -> AdditiveFit:
    """Least-squares fit of y_{i,S} = l_i - c_S.

    Rank-deficient by one (adding a constant to every l and every c is
    invisible), so the minimum-norm solution from lstsq is used and only
    *differences* are ever interpreted.
    """
    rows_p, rows_s, y = [], [], []
    for k, (members, lp) in enumerate(zip(design.slates, logp)):
        lp = np.asarray(lp, dtype=float)
        if lp.size != len(members):
            raise ValueError(f"slate {k}: {lp.size} log-probs for {len(members)} members")
        for i, v in zip(members, lp):
            rows_p.append(i)
            rows_s.append(k)
            y.append(v)
    rows_p = np.asarray(rows_p)
    rows_s = np.asarray(rows_s)
    y = np.asarray(y, dtype=float)

    n_obs = y.size
    X = np.zeros((n_obs, design.n_probes + design.n_slates))
    X[np.arange(n_obs), rows_p] = 1.0
    X[np.arange(n_obs), design.n_probes + rows_s] = -1.0

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return AdditiveFit(beta[:design.n_probes], beta[design.n_probes:], resid,
                       float(np.sqrt(np.mean(resid ** 2))),
                       1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
                       rows_p, rows_s, y)


# ------------------------------------------------------------- noise floor
def order_noise_floor(logp_by_perm: list[np.ndarray],
                      perms: list[np.ndarray]) -> dict:
    """Spread of log p for the same candidate in the same slate, across orders.

    This is the measurement floor. A deterministic model re-scored on identical
    input gives identical numbers, so the only honest baseline for "how much
    should a residual be?" is how much the score moves when nothing about the
    slate's *content* changes.
    """
    n = len(perms[0])
    vals = np.full((len(perms), n), np.nan)
    for r, (lp, perm) in enumerate(zip(logp_by_perm, perms)):
        inv = np.empty_like(perm)
        inv[perm] = np.arange(n)
        vals[r] = np.asarray(lp)[inv]
    per_cand_std = np.nanstd(vals, axis=0, ddof=1)
    argmax_each = [int(np.nanargmax(v)) for v in vals]
    mode = max(set(argmax_each), key=argmax_each.count)
    return {"per_candidate_std_mean": float(np.mean(per_cand_std)),
            "per_candidate_std_max": float(np.max(per_cand_std)),
            "range_mean": float(np.mean(np.nanmax(vals, 0) - np.nanmin(vals, 0))),
            "argmax_changed_frac": float(np.mean([a != mode for a in argmax_each])),
            "n_permutations": len(perms), "values": vals}


# ------------------------------------------------------------- interaction
def interaction_test(fit: AdditiveFit, design: BlockDesign) -> dict:
    """Is what the additive model missed systematic in composition?

    Regress the residual on the candidate's own grade, the slate's mean grade,
    and their product. A non-trivial product term is the specific failure mode
    Part XI names: a candidate scoring differently because of *which* others
    share its slate.
    """
    slate_mean = np.array([design.probe_grades[np.asarray(s)].mean()
                           for s in design.slates])
    g_i = design.probe_grades[fit.obs_probe].astype(float)
    m_s = slate_mean[fit.obs_slate]
    X = np.column_stack([np.ones_like(g_i), g_i, m_s, g_i * m_s])
    beta, *_ = np.linalg.lstsq(X, fit.residuals, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((fit.residuals - pred) ** 2))
    ss_tot = float(np.sum((fit.residuals - fit.residuals.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"beta_intercept": float(beta[0]), "beta_grade": float(beta[1]),
            "beta_slate_mean": float(beta[2]), "beta_interaction": float(beta[3]),
            "r2_of_residual": float(r2),
            "residual_std_explained": float(np.std(pred))}


# ---------------------------------------------------------- anchor recovery
def anchor_recovery(fit: AdditiveFit, design: BlockDesign,
                    anchor_probes: np.ndarray, lam: float = 1.0) -> dict:
    """Falsification test 2: do known-grade pivots recover the fitted c_S?

    For each slate containing at least two anchors, fit the affine map from the
    anchors' observed log-probabilities to their known utilities, then ask how
    well the recovered offset matches the offset the full additive fit found.
    A curved residual pattern means the affine map is the wrong family.
    """
    from ..anchor_fit import fit_slate

    anchor_probes = np.asarray(anchor_probes)
    # the anchors' "known utility" is their fitted latent logit, centred --
    # in production these are graded pivots; here we ask the weaker question
    # of whether the recovery machinery works at all given known values
    known = fit.l_hat[anchor_probes] - fit.l_hat[anchor_probes].mean()

    recovered, truth, resids, slopes, used = [], [], [], [], []
    for k, members in enumerate(design.slates):
        members = np.asarray(members)
        present = [j for j, m in enumerate(members) if m in set(anchor_probes.tolist())]
        if len(present) < 2:
            continue
        sel = members[present]
        obs = np.array([fit.y[(fit.obs_slate == k) & (fit.obs_probe == m)][0]
                        for m in sel])
        util = np.array([known[np.flatnonzero(anchor_probes == m)[0]] for m in sel])
        sf = fit_slate(obs, util, method="ridge", lam=lam)
        # under the model obs = l - c_S, so an intercept of +c_S recovers it
        recovered.append(sf.intercept)
        truth.append(fit.c_hat[k] - fit.l_hat[anchor_probes].mean())
        resids.append(sf.residual)
        slopes.append(sf.slope)
        used.append(len(present))

    if not recovered:
        return {"n_slates_usable": 0}
    recovered = np.asarray(recovered)
    truth = np.asarray(truth)
    return {"n_slates_usable": len(recovered),
            "anchors_per_slate_mean": float(np.mean(used)),
            "corr_recovered_vs_fitted_offset": float(
                np.corrcoef(recovered, truth)[0, 1]) if len(recovered) > 2 else float("nan"),
            "offset_rmse": float(np.sqrt(np.mean((recovered - truth) ** 2))),
            "anchor_fit_residual_mean": float(np.mean(resids)),
            "anchor_fit_residual_max": float(np.max(resids)),
            "slope_mean": float(np.mean(slopes)),
            "slope_std": float(np.std(slopes))}


# ==========================================================================
# Falsification tests 3 and 4: does anchoring actually buy the global scale?
# ==========================================================================
#
# T2 established that c_S exists and is additive. That is necessary, not
# sufficient. The claim the architecture rests on is stronger: that a handful
# of pivots *recover* c_S well enough to make scores comparable ACROSS slates.
#
# That can fail even when T2 passes, for a reason the plan's simulation could
# not see. The simulation assumed marker noise sigma = 0.35 and slate drift
# sigma = 0.15 -- noise four times smaller than the signal. Real Laya has an
# order-noise floor around 0.38 nats against a c_S spread around 1.0, so each
# anchor measures the offset with error comparable to the thing being measured.
# With A = 2 the estimator is averaging two noisy numbers. Whether anchoring
# works is therefore an SNR question, and the answer is a curve in A, not a
# yes or no.

def anchored_slate_design(n_candidates: int, anchor_ids, slate_size: int = 10,
                          n_slates: int = 40,
                          rng: np.random.Generator | None = None,
                          grades=None, homogeneous_frac: float = 0.0) -> BlockDesign:
    """Production-shaped slates: the same anchors in EVERY slate, plus candidates.

    `homogeneous_frac` is the variable that decides whether this test has
    anything to detect. Anchors correct the per-slate offset c_S, so they can
    only help to the extent that c_S actually varies -- and c_S varies with
    slate *composition*. Drawing candidates uniformly gives every slate roughly
    the same difficulty, c_S barely moves, and anchoring has nothing to fix.

    That is not a rigged test, it is the production regime: plan.md Part VIII
    recommends dealing candidates round-robin precisely because it makes slate
    composition near-constant. But it means "anchoring does not help" measured
    at homogeneous_frac=0 is a statement about the easy regime only.

    Setting it above zero draws a fraction of slates from a single grade, which
    is the streaming / blocked-slate case where Part VIII reports anchors as
    non-optional (+0.223 nDCG@10). Sweeping it is how you find out whether
    anchoring is useless or merely unnecessary when you slate well.
    """
    rng = rng or np.random.default_rng(0)
    anchor_ids = np.asarray(anchor_ids)
    others = np.setdiff1d(np.arange(n_candidates), anchor_ids)
    per = slate_size - anchor_ids.size
    if per < 1:
        raise ValueError("anchors consume the entire slate")

    by_grade = {}
    if grades is not None and homogeneous_frac > 0:
        g = np.asarray(grades)
        for v in np.unique(g[others]):
            pool = others[g[others] == v]
            if pool.size >= per:
                by_grade[float(v)] = pool

    slates = []
    for _ in range(n_slates):
        if by_grade and rng.random() < homogeneous_frac:
            pool = by_grade[float(rng.choice(list(by_grade)))]
        else:
            pool = others
        take = rng.choice(pool, size=min(per, pool.size), replace=False)
        slates.append(sorted(np.concatenate([anchor_ids, take]).tolist()))
    return BlockDesign(slates, np.zeros(n_candidates), n_candidates)


def spread_anchors(values, n: int) -> np.ndarray:
    """Pick n indices whose values are as evenly spread as possible.

    Anchor choice is not incidental. The affine fit's leverage is the spread of
    the anchors' utilities, so pivots bunched together give an ill-conditioned
    slope no amount of regularisation repairs. Part VIII's fifth falsification
    test -- "does anchor choice matter more than it should?" -- is about exactly
    this, and picking by quantile is the honest default to measure against.
    """
    v = np.asarray(values, dtype=float)
    targets = np.quantile(v, np.linspace(0.05, 0.95, n))
    picks, used = [], set()
    for t in targets:
        order = np.argsort(np.abs(v - t))
        for j in order:
            if int(j) not in used:
                used.add(int(j))
                picks.append(int(j))
                break
    return np.asarray(sorted(picks))


def estimate_slope_prior(logp: list[np.ndarray], truth, design: BlockDesign) -> float:
    """Ratio of utility spread to log-probability spread.

    The ridge estimator shrinks the slope toward this. Getting it wrong biases
    every slate's fit in the same direction, which is indistinguishable from
    the mechanism not working.
    """
    lp = np.concatenate([np.asarray(x, dtype=float) for x in logp])
    t = np.asarray(truth, dtype=float)
    lp_spread = float(np.std(lp))
    t_spread = float(np.std(t))
    return t_spread / lp_spread if lp_spread > 1e-9 else 1.0


def cross_slate_scale(design: BlockDesign, logp: list[np.ndarray],
                      anchor_ids, anchor_utility, truth,
                      method: str = "ridge", lam: float = 1.0,
                      slope_prior: float | None = None) -> dict:
    """Falsification test 4: pool calibrated utilities and correlate with truth.

    Naive log-probabilities have no absolute scale, so pooling them across
    slates compares one slate's apples with another's. If anchoring is worth
    anything, the pooled correlation must rise -- and that single number is what
    the frontier, the thresholds and the multi-round re-scoring all depend on.
    """
    from ..anchor_fit import fit_slate

    anchor_ids = np.asarray(anchor_ids)
    anchor_set = set(anchor_ids.tolist())
    anchor_utility = np.asarray(anchor_utility, dtype=float)
    truth = np.asarray(truth, dtype=float)

    naive_v, cal_v, truth_v, resid, slopes = [], [], [], [], []
    for members, lp in zip(design.slates, logp):
        members = np.asarray(members)
        lp = np.asarray(lp, dtype=float)
        a_pos = [j for j, m in enumerate(members) if m in anchor_set]
        c_pos = [j for j, m in enumerate(members) if m not in anchor_set]
        if len(a_pos) < 1 or not c_pos:
            continue
        util = np.array([anchor_utility[np.flatnonzero(anchor_ids == members[j])[0]]
                         for j in a_pos])
        sf = fit_slate(lp[a_pos], util, method=method, lam=lam,
                       slope_prior=slope_prior)
        naive_v.append(lp[c_pos])
        cal_v.append(sf.apply(lp[c_pos]))
        truth_v.append(truth[members[c_pos]])
        resid.append(sf.residual)
        slopes.append(sf.slope)

    if not cal_v:
        return {"n_slates": 0}
    naive_v = np.concatenate(naive_v)
    cal_v = np.concatenate(cal_v)
    truth_v = np.concatenate(truth_v)

    def _r(a, b):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    return {"n_slates": len(resid), "n_pooled": int(truth_v.size),
            "r_naive": _r(naive_v, truth_v),
            "r_anchored": _r(cal_v, truth_v),
            "delta_r": _r(cal_v, truth_v) - _r(naive_v, truth_v),
            "rmse_anchored": float(np.sqrt(np.mean((cal_v - truth_v) ** 2))),
            "anchor_residual_mean": float(np.mean(resid)),
            "slope_mean": float(np.mean(slopes)),
            "slope_std": float(np.std(slopes))}
