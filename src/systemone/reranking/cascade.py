"""The system: score locally, find the frontier, buy a judgement only where it pays.

This is where the parts assembled elsewhere become FrontierRank. One query in,
a ranking out, plus a full account of what was spent and why.

    signatures -> AnchoredScorer  -> calibrated utility + sigma
                -> compute_frontier -> the candidates whose rank-k membership
                                       is still in doubt
                -> Controller      -> stop / read_more / widen / teacher
                -> TeacherProtocol -> absolute grades on the frontier
                -> fuse            -> a ranking, and a training example

Two things here are easy to get wrong and are handled explicitly.

**Fusing the teacher's answer.** The teacher returns grades on an absolute
rubric; the student returns a calibrated utility. They are on different scales,
and averaging them directly would let the rubric's coarseness overwrite the
student's fine ordering. The teacher's expected grade is mapped onto the
utility scale by the same affine convention the anchors use, then blended with
a weight that reflects which one we actually believe -- and on the evidence in
Part VI, that is not automatically the teacher.

**Not re-ranking what was never judged.** Escalation covers the frontier, not
the pool. Candidates the teacher never saw keep their student utility, and the
fusion must not accidentally reorder them relative to judged ones. The teacher
is applied as a *correction on the frontier*, with everything else untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .controller import Action, Controller
from .frontier import compute_frontier, expected_regret

__all__ = ["CascadeResult", "Cascade", "fuse_teacher"]


def fuse_teacher(utility: np.ndarray, sigma: np.ndarray, targets: np.ndarray,
                 teacher_grade: np.ndarray, teacher_weight: float = 0.7,
                 grade_to_utility=lambda g: g * 2.0 - 3.0,
                 sigma_after: float = 0.12):
    """Blend teacher grades into the student's utilities, on the frontier only.

    `teacher_weight` is a belief, not a constant of nature. Part VI's matched
    protocol found Jev and Laya statistically indistinguishable on ordinary
    classification, so a weight of 1.0 -- treating the teacher as ground truth
    -- is not supported by the evidence. It is exposed so the ablation can be
    run rather than assumed.
    """
    u = np.array(utility, dtype=float, copy=True)
    s = np.array(sigma, dtype=float, copy=True)
    if targets.size == 0:
        return u, s
    t_u = np.asarray(grade_to_utility(np.asarray(teacher_grade, dtype=float)))
    u[targets] = (1.0 - teacher_weight) * u[targets] + teacher_weight * t_u
    # a judged candidate is a more certain candidate, but not a certain one
    s[targets] = np.minimum(s[targets], sigma_after)
    return u, s


@dataclass
class CascadeResult:
    order: np.ndarray
    utility: np.ndarray
    sigma: np.ndarray
    actions: list = field(default_factory=list)
    frontier_size: int = 0
    regret_before: float = 0.0
    regret_after: float = 0.0
    escalated: bool = False
    teacher_targets: np.ndarray = field(default_factory=lambda: np.array([], int))
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    training_examples: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"frontier": self.frontier_size, "escalated": self.escalated,
                "actions": [a for a, _ in self.actions],
                "regret_before": round(self.regret_before, 5),
                "regret_after": round(self.regret_after, 5),
                "regret_resolved": round(self.regret_before - self.regret_after, 5),
                "cost_usd": round(self.cost_usd, 8),
                "latency_ms": round(self.latency_ms, 1)}


class Cascade:
    """One query, start to finish, with the budget accounted for.

    `max_rounds` exists because `read_more` can otherwise loop: more evidence
    changes the scores, which changes the frontier, which can ask for more
    evidence. The controller's value test should stop it, but a hard bound is
    cheaper than trusting a fitted model in production.
    """

    def __init__(self, scorer, teacher=None, controller: Controller | None = None,
                 k: int = 10, teacher_batch: int = 30, max_rounds: int = 2,
                 teacher_weight: float = 0.7, collect_training: bool = True):
        self.scorer = scorer
        self.teacher = teacher
        self.controller = controller or Controller(lambda_cost=1000.0)
        self.k = k
        self.teacher_batch = teacher_batch
        self.max_rounds = max_rounds
        self.teacher_weight = teacher_weight
        self.collect_training = collect_training

    def run(self, query: str, signatures, rel_hat=None,
            first_stage_order=None, rng=None) -> CascadeResult:
        rng = rng or np.random.default_rng(0)
        # Regret is a Monte Carlo estimate, so `before` and `after` must be
        # drawn from the SAME stream or the difference is dominated by sampling
        # noise -- which showed up as regret rising after an action that could
        # only reduce it.
        regret_seed = int(rng.integers(0, 2**31 - 1))

        def _regret(uu, ss, rr):
            return expected_regret(uu, ss, rr, k=self.k,
                                   rng=np.random.default_rng(regret_seed))

        pool = self.scorer.score(query, list(signatures),
                                 first_stage_order=first_stage_order, rng=rng)
        u, s = pool.utility.copy(), pool.sigma.copy()
        # without an ordinal head, the utility is the best available stand-in
        # for expected graded relevance; the regret estimate inherits that.
        rel = np.asarray(rel_hat, dtype=float) if rel_hat is not None \
            else np.clip(u, 0.0, None)

        res = CascadeResult(np.argsort(-u), u, s)
        res.regret_before = _regret(u, s, rel)
        res.regret_after = res.regret_before

        for _ in range(self.max_rounds):
            fr = compute_frontier(u, s, k=self.k)
            res.frontier_size = fr.size
            decision = self.controller.decide(u, s, rel, fr,
                                              pool_quality=float(np.max(u)))
            res.actions.append((decision.action, round(decision.value, 6)))
            if decision.action == Action.STOP:
                break

            if decision.action in (Action.TEACHER_RUBRIC, Action.TEACHER_CHOICE):
                if self.teacher is None:
                    break
                targets = fr.head(self.teacher_batch)
                if targets.size == 0:
                    break
                verdict = self.teacher.grade(query, [signatures[i] for i in targets])
                u, s = fuse_teacher(u, s, targets, verdict.expected_grade,
                                    self.teacher_weight)
                rel = np.clip(u, 0.0, None) if rel_hat is None else rel
                res.escalated = True
                res.teacher_targets = targets
                res.cost_usd += float(verdict.cost_usd)
                res.latency_ms += float(verdict.latency_ms)
                if self.collect_training:
                    # every escalation is a labelled example -- this is the
                    # flywheel, and it costs nothing extra because the call
                    # was already paid for
                    res.training_examples.append({
                        "query": query,
                        "signatures": [signatures[i] for i in targets],
                        "teacher_level_probs": verdict.level_probs,
                        "student_utility": pool.utility[targets],
                        "student_sigma": pool.sigma[targets],
                    })
                break                       # one teacher call per query
            # read_more / widen are the caller's to implement: they need the
            # corpus, which this layer deliberately does not have
            break

        res.utility, res.sigma = u, s
        res.order = np.argsort(-u)
        res.regret_after = _regret(u, s, rel)
        return res
