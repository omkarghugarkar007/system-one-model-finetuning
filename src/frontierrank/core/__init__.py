"""The theory layer: numpy only, no model, no network, no corpus.

Everything here runs on CPU in seconds and is what the falsification tests in
Part VIII exercise. If a claim cannot be stated against this layer, it is not
yet a claim about the mechanism.
"""
from .budget import (DEFAULT_BUDGET, LayaBudget, SlateBudget, SlateLayout,
                     layout_comparison, option_tokens_available, plan_slate_budget)
from .calibrate import AnchorCalibrator, SlateFit, fit_slate
from .controller import (Action, ActionSpec, Controller, ControllerDecision,
                         DEFAULT_ACTIONS)
from .frontier import Frontier, compute_frontier, expected_regret, swap_probability
from .packing import (DEFAULT_INSTRUCTIONS, OptionsPacker, PackedSlate,
                      SlatePacker, StatePacker, make_packer)
from .scoring import Anchor, AnchoredScorer, ScoredPool
from .slates import Slate, SlatePlan, build_slates, slates_per_query

__all__ = [
    "LayaBudget", "SlateBudget", "SlateLayout", "DEFAULT_BUDGET",
    "option_tokens_available", "plan_slate_budget", "layout_comparison",
    "AnchorCalibrator", "SlateFit", "fit_slate",
    "Slate", "SlatePlan", "build_slates", "slates_per_query",
    "PackedSlate", "SlatePacker", "OptionsPacker", "StatePacker", "make_packer",
    "DEFAULT_INSTRUCTIONS",
    "Anchor", "AnchoredScorer", "ScoredPool",
    "Frontier", "compute_frontier", "expected_regret", "swap_probability",
    "Action", "ActionSpec", "Controller", "ControllerDecision", "DEFAULT_ACTIONS",
]
