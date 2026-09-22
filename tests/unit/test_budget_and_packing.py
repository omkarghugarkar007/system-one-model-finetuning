"""The token budget is load-bearing: it decided the slate design. Pin it down."""
import numpy as np
import pytest

from systemone.model.budget import (DEFAULT_BUDGET, SlateLayout,
                                      max_options_for_option_text,
                                      option_tokens_available, plan_slate_budget)
from systemone.reranking.packing import OptionsPacker, StatePacker


def test_the_number_that_drove_the_design():
    # At the plan's M=10 a candidate signature gets 16 tokens of option text.
    # If this ever changes, the OPTIONS layout has to be re-argued from scratch.
    assert option_tokens_available(10) == 16


def test_shrink_fires_from_four_options():
    # 3 x 49 = 147 <= 176, so M=3 never shrinks; 4 x 49 = 196 does.
    assert option_tokens_available(3) == DEFAULT_BUDGET.option_token_cap
    assert option_tokens_available(4) < DEFAULT_BUDGET.option_token_cap


def test_option_budget_is_monotone_decreasing_in_slate_size():
    got = [option_tokens_available(m) for m in range(2, 40)]
    assert all(a >= b for a, b in zip(got, got[1:]))


def test_option_budget_never_falls_below_the_floor():
    # max(4, ...) INCLUDES the [MASK], so 3 tokens of text is the floor
    assert min(option_tokens_available(m) for m in range(2, 256)) == 3


def test_short_options_are_not_shrunk():
    # bare ids are cheap; the shrink is about long option text, not option count
    assert option_tokens_available(10, desired_option_tokens=2) == 2


def test_max_options_inverts_the_budget():
    for want in (5, 10, 16, 28):
        m = max_options_for_option_text(want)
        assert option_tokens_available(m, want) >= want
        assert option_tokens_available(m + 1, want) < want


def test_state_layout_buys_more_tokens_per_candidate():
    a = plan_slate_budget(10, SlateLayout.OPTIONS)
    b = plan_slate_budget(10, SlateLayout.STATE)
    assert b.tokens_per_candidate > 2 * a.tokens_per_candidate


def test_state_packer_pre_truncates_evenly():
    # the trap: build_sequence cuts the state from the RIGHT, so without
    # pre-truncation the last candidates would be scored on nothing
    texts = ["word " * 400 for _ in range(10)]
    packed = StatePacker().pack("a query", texts)
    lens = packed.info["per_candidate_chars"]
    assert max(lens) - min(lens) <= 5, "truncation must be even across candidates"
    assert all(n > 0 for n in lens), "no candidate may be emptied"
    assert all(packed.info["truncated"])


def test_options_packer_keeps_short_text_intact():
    texts = [f"candidate {i}" for i in range(10)]
    packed = OptionsPacker().pack("q", texts)
    assert packed.options == texts
    assert not any(packed.info["truncated"])
    assert packed.state == "q"


def test_packers_preserve_count_and_order():
    texts = [f"cand-{i}-{'x' * i}" for i in range(8)]
    for packer in (OptionsPacker(), StatePacker()):
        p = packer.pack("q", texts)
        assert len(p.options) == len(texts)
    # the state layout must keep ids aligned with insertion order
    p = StatePacker().pack("q", texts)
    assert list(p.state["candidates"].keys()) == [f"c{i}" for i in range(8)]
