"""Escalation as a decision problem, not a confidence threshold.

An uncertain prediction is not automatically worth resolving. An uncertain
prediction NEAR RANK K is. The controller compares the expected nDCG@k recovered
by an action against what that action costs:

    a* = argmax_a  E[dNDCG(a)] - lambda_C * cost(a) - lambda_L * latency(a)
    stop if the best value is below zero.

Start with the supervised-counterfactual estimator, not RL. Log every action's
realised outcome offline, fit a regressor to V(s, a), and only reach for bandits
once production feedback exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frontier import Frontier, expected_regret

__all__ = ["Action", "ActionSpec", "Controller", "ControllerDecision", "DEFAULT_ACTIONS"]


class Action:
    STOP = "stop"
    READ_MORE = "read_more"          # pull more evidence windows from a document
    TEACHER_RUBRIC = "teacher_rubric"  # one batched Jev Score call over the frontier
    TEACHER_CHOICE = "teacher_choice"  # one Jev Choice over the frontier
    WIDEN = "widen_retrieval"        # the pool itself is bad; go get more candidates


@dataclass
class ActionSpec:
    name: str
    cost_usd: float                  # marginal $ per invocation
    latency_ms: float
    # expected fraction of the current regret this action resolves, if taken.
    # Seed from priors; replace with a fitted model as soon as logs exist.
    resolves: float = 0.5
    applicable_min_frontier: int = 1


DEFAULT_ACTIONS = [
    # 30 frontier candidates x 250 tok + 500 overhead = 8000 tok @ $0.042/M
    ActionSpec(Action.TEACHER_RUBRIC, cost_usd=0.000336, latency_ms=300.0, resolves=0.70),
    ActionSpec(Action.TEACHER_CHOICE, cost_usd=0.000252, latency_ms=300.0, resolves=0.55),
    # local: a few extra Laya slate passes, ~14 ms each on T4
    ActionSpec(Action.READ_MORE, cost_usd=0.0000042, latency_ms=45.0, resolves=0.30,
               applicable_min_frontier=1),
    ActionSpec(Action.WIDEN, cost_usd=0.00002, latency_ms=120.0, resolves=0.25),
]


@dataclass
class ControllerDecision:
    action: str
    value: float
    regret_before: float
    targets: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    detail: dict = field(default_factory=dict)


class Controller:
    def __init__(self, actions=None, lambda_cost: float = 1.0, lambda_latency: float = 0.0,
                 teacher_batch: int = 30, min_value: float = 0.0,
                 rng: np.random.Generator | None = None):
        """lambda_cost converts dollars into nDCG points.

        Setting it is the single most consequential knob in the system: it *is*
        the operating point. lambda_cost = 1000 means you will pay $0.001 for
        one nDCG@10 point per query. Sweep it to draw the Pareto curve; do not
        tune it to a target escalation rate, which inverts cause and effect.

        The useful range is much lower than it looks. A teacher call costs
        $0.000336 and resolves roughly 0.7 of the current regret, so it is
        worth buying when

            lambda_cost < 0.7 * regret / 0.000336  ~=  2000 * regret

        At a typical regret of 0.03 that is lambda_cost < ~67. Sweeping
        1e2..1e6, which looks like a natural range, sits entirely inside the
        "never escalate" regime and produces a flat curve that looks like a
        broken controller. `break_even_lambda` computes the boundary so the
        sweep can be centred on it.
        """
        self.actions = list(actions or DEFAULT_ACTIONS)
        self.lambda_cost = lambda_cost
        self.lambda_latency = lambda_latency
        self.teacher_batch = teacher_batch
        self.min_value = min_value
        self.rng = rng or np.random.default_rng()

    def break_even_lambda(self, regret: float, action: str | None = None) -> float:
        """The lambda_cost at which an action stops paying for itself.

        Centre a Pareto sweep on this. A sweep that never crosses it produces a
        flat curve and looks like a bug in the controller.
        """
        specs = [a for a in self.actions if action is None or a.name == action]
        vals = [a.resolves * regret / max(a.cost_usd, 1e-12) for a in specs]
        return float(max(vals)) if vals else float("inf")

    def decide(self, u, sigma, rel_hat, frontier: Frontier,
               pool_quality: float | None = None) -> ControllerDecision:
        regret = expected_regret(u, sigma, rel_hat, k=frontier.boundary_rank, rng=self.rng)

        best, best_val, best_targets = Action.STOP, 0.0, np.array([], dtype=int)
        detail = {}
        for spec in self.actions:
            if frontier.size < spec.applicable_min_frontier:
                continue
            if spec.name == Action.WIDEN:
                # only sensible when the whole pool looks weak, not when the
                # ordering is merely uncertain
                if pool_quality is None or pool_quality > 0.35:
                    continue
            gain = spec.resolves * regret
            val = gain - self.lambda_cost * spec.cost_usd - self.lambda_latency * spec.latency_ms
            detail[spec.name] = val
            if val > best_val:
                best, best_val = spec.name, val
                best_targets = frontier.head(self.teacher_batch)

        if best_val <= self.min_value:
            best, best_val, best_targets = Action.STOP, 0.0, np.array([], dtype=int)
        return ControllerDecision(best, best_val, regret, best_targets, detail)
