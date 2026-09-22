"""The worked example: reranking with anchored slates and a regret frontier.

This is the research the recipe came out of. It is kept because it is the
honest demonstration of what a fine-tuned System One model can and cannot do --
see FINDINGS.md, including the baseline row where a 22M cross-encoder beats it
on quality.

Nothing in `systemone.data`, `.model`, `.train` or `.calibrate` depends on
anything here.
"""
from .anchor_fit import AnchorCalibrator, SlateFit, fit_slate
from .cascade import Cascade, CascadeResult
from .controller import Action, ActionSpec, Controller, DEFAULT_ACTIONS
from .frontier import Frontier, compute_frontier, expected_regret
from .packing import OptionsPacker, PackedSlate, StatePacker, make_packer
from .scoring import Anchor, AnchoredScorer, ScoredPool
from .slates import Slate, SlatePlan, build_slates, slates_per_query

__all__ = ["Anchor", "AnchoredScorer", "ScoredPool", "AnchorCalibrator",
           "SlateFit", "fit_slate", "build_slates", "slates_per_query",
           "Slate", "SlatePlan", "PackedSlate", "OptionsPacker", "StatePacker",
           "make_packer", "Frontier", "compute_frontier", "expected_regret",
           "Controller", "Action", "ActionSpec", "DEFAULT_ACTIONS",
           "Cascade", "CascadeResult"]
