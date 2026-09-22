"""The System One model: sequence construction, decision head, token budget.

`sequence.py` is a faithful re-implementation of the shipped
`build_sequence`, and `budget.py` is the arithmetic that governs it. Read
`budget.py` before designing any question -- it is the constraint that decides
how much evidence reaches the model, and it is not documented upstream.
"""
from .budget import (DEFAULT_BUDGET, LayaBudget, SlateBudget, SlateLayout,
                     layout_comparison, max_options_for_option_text,
                     option_tokens_available, plan_slate_budget)
from .heads import SystemOneModel, OrdinalHead, coral_probs
from .runtime import DecisionModel, LayaConfig, LayaRuntime, pick_device
from .sequence import (QTYPES, build_sequence, render_options, temp_bucket)

__all__ = ["LayaRuntime", "LayaConfig", "DecisionModel", "pick_device",
           "OrdinalHead", "SystemOneModel", "coral_probs",
           "build_sequence", "render_options", "temp_bucket", "QTYPES",
           "LayaBudget", "SlateBudget", "SlateLayout", "DEFAULT_BUDGET",
           "option_tokens_available", "max_options_for_option_text",
           "plan_slate_budget", "layout_comparison"]
