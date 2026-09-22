"""Token accounting for Laya's `build_sequence`, and what it costs a slate.

This module exists because reading the shipped source changed the design. The
plan assumed the binding constraint was Laya's ~320-token *state* budget. It is
not. The binding constraint is the **head** budget, and it bites the option
text, which is where a candidates-as-options slate puts its evidence.

The arithmetic, transcribed from `rl_common.build_sequence` (Apache-2.0,
convaiinnovations/laya) with `head_max_len=192`, `max_len=512`:

    opt_ids[i]  = [MASK] + tokens(" " + option_i)[:48]      # 48-token hard cap
    opt_budget  = head_max_len - sum(len(opt_ids))
    if opt_budget < 16:                                     # options overflow
        per     = max(4, (head_max_len - 16) // n_options)  # shrink EVERY option
        opt_ids = [o[:per] for o in opt_ids]                # `per` INCLUDES [MASK]
    head_ids    = instructions[:max(8, opt_budget)]
    room        = max_len - (1 + len(head_ids) + 1 + sum(len(opt_ids)) + 1) - 1
    state       = tokens(state)[:room]

Two consequences the design has to respect:

1. The shrink fires at **M >= 4** options (4 x 49 = 196 > 176). At the plan's
   M = 10 it leaves **16 tokens of option text per candidate** and 22 tokens of
   instructions. A "signature" with title, section path, metadata and two
   evidence windows does not fit in 16 tokens. Not approximately -- not at all.

2. Therefore there are two slate layouts, and they are not stylistic variants:

   OPTIONS   candidates are the options; the marker sits on the candidate's own
             text. Discrimination is local, but each candidate gets ~16 tokens.
   STATE     candidates live in the state and the options are bare ids ("c0");
             each candidate gets ~40 tokens, but the marker sits on a
             semantically empty token and all discrimination must arrive by
             attention from the state.

   Which one wins is an empirical question, not a design choice, and it is the
   first thing Phase 0 measures. `layout_comparison()` prints the budgets;
   `scripts/reranking/run_phase0_signal.py` measures the ranking.

Everything here is pure arithmetic on token counts, so it runs without the
tokenizer. `systemone.model.budget_probe` checks these predictions
against the real tokenizer and fails loudly if a Laya release changes them.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "LayaBudget", "SlateBudget", "SlateLayout", "DEFAULT_BUDGET",
    "option_tokens_available", "plan_slate_budget", "layout_comparison",
    "max_options_for_option_text",
]


class SlateLayout:
    """Where a slate puts the candidate evidence."""
    OPTIONS = "options"   # candidates ARE the options; ~16 tok each at M=10
    STATE = "state"       # candidates live in the state; options are bare ids


@dataclass(frozen=True)
class LayaBudget:
    """The three numbers from `rl_agent_config.json` plus the two literals."""
    max_len: int = 512
    head_max_len: int = 192
    option_token_cap: int = 48    # the `[:48]` in build_sequence
    min_opt_budget: int = 16      # the `< 16` that triggers the shrink
    min_per_option: int = 4       # the `max(4, ...)` floor
    min_head_tokens: int = 8      # the `max(8, opt_budget)` floor


DEFAULT_BUDGET = LayaBudget()


def option_tokens_available(n_options: int,
                            desired_option_tokens: int = 10_000,
                            budget: LayaBudget = DEFAULT_BUDGET) -> int:
    """Tokens of *option text* each option actually keeps (excludes the [MASK]).

    `desired_option_tokens` is how long the untruncated option text is; pass a
    large number to ask "what is the ceiling at this M?".
    """
    if n_options < 1:
        raise ValueError("a slate needs at least one option")
    # every option is first capped at 48, then gets +1 for its [MASK] marker
    per_option = 1 + min(desired_option_tokens, budget.option_token_cap)
    opt_budget = budget.head_max_len - n_options * per_option
    if opt_budget >= budget.min_opt_budget:
        return per_option - 1
    per = max(budget.min_per_option,
              (budget.head_max_len - budget.min_opt_budget) // n_options)
    # the shrink truncates, it never extends
    return max(0, min(per, per_option) - 1)


def max_options_for_option_text(min_option_tokens: int,
                                budget: LayaBudget = DEFAULT_BUDGET) -> int:
    """Largest M that still leaves each option at least `min_option_tokens`.

    The inverse question, and the one a slate designer actually asks: "I need
    30 tokens per candidate -- how many fit?"
    """
    # option_tokens_available is monotone decreasing in M, so the first failure
    # is the boundary. 255 is Jev's Choice cap and a sane place to stop.
    best = 0
    for m in range(1, 256):
        if option_tokens_available(m, min_option_tokens, budget) >= min_option_tokens:
            best = m
        else:
            break
    return best


@dataclass(frozen=True)
class SlateBudget:
    """What one slate forward pass can actually carry."""
    layout: str
    n_options: int
    option_text_tokens: int      # per option, excluding the [MASK]
    instruction_tokens: int
    state_tokens: int
    tokens_per_candidate: int    # the number that matters: evidence per candidate
    truncation_fired: bool

    def __repr__(self) -> str:
        return (f"SlateBudget({self.layout}, M={self.n_options}, "
                f"opt={self.option_text_tokens}, ins={self.instruction_tokens}, "
                f"state={self.state_tokens}, per_cand={self.tokens_per_candidate})")


def plan_slate_budget(n_options: int,
                      layout: str = SlateLayout.OPTIONS,
                      option_id_tokens: int = 3,
                      instruction_tokens: int = 40,
                      budget: LayaBudget = DEFAULT_BUDGET) -> SlateBudget:
    """Exact budget for one slate pass, without touching a tokenizer.

    layout=OPTIONS  candidates are the options. `option_id_tokens` is ignored;
                    the option text is the candidate signature and takes
                    whatever `option_tokens_available` leaves it.
    layout=STATE    options are bare ids of `option_id_tokens` tokens, and every
                    candidate's evidence is carved out of the state budget.
    """
    if layout == SlateLayout.OPTIONS:
        opt_text = option_tokens_available(n_options, budget.option_token_cap, budget)
        fired = opt_text < min(budget.option_token_cap, budget.option_token_cap)
        per_opt_total = opt_text + 1
        opt_budget = budget.head_max_len - n_options * per_opt_total
        ins = max(budget.min_head_tokens, min(instruction_tokens, max(opt_budget, 0)))
        head_len = 1 + ins + 1 + n_options * per_opt_total + 1
        state = max(0, budget.max_len - head_len - 1)
        # the candidate's evidence is its option text; the state holds the query
        return SlateBudget(layout, n_options, opt_text, ins, state, opt_text, fired)

    if layout == SlateLayout.STATE:
        opt_text = option_tokens_available(n_options, option_id_tokens, budget)
        fired = opt_text < option_id_tokens
        per_opt_total = opt_text + 1
        opt_budget = budget.head_max_len - n_options * per_opt_total
        ins = max(budget.min_head_tokens, min(instruction_tokens, max(opt_budget, 0)))
        head_len = 1 + ins + 1 + n_options * per_opt_total + 1
        state = max(0, budget.max_len - head_len - 1)
        # the state holds the query AND every candidate; reserve ~20 for the query
        per_cand = max(0, (state - 20)) // max(1, n_options)
        return SlateBudget(layout, n_options, opt_text, ins, state, per_cand, fired)

    raise ValueError(f"unknown layout {layout!r}")


def layout_comparison(sizes=(2, 4, 6, 8, 10, 12, 16, 20, 30),
                      budget: LayaBudget = DEFAULT_BUDGET) -> str:
    """The table that justifies testing both layouts. Pure arithmetic."""
    lines = [
        "Evidence per candidate, by slate size and layout (Laya: max_len=512, head_max_len=192)",
        "",
        f"{'M':>4} {'OPTIONS: opt tok':>17} {'state':>7} | {'STATE: per cand':>16} {'state':>7} "
        f"{'ins':>5} | {'winner':>8}",
        "-" * 82,
    ]
    for m in sizes:
        a = plan_slate_budget(m, SlateLayout.OPTIONS, budget=budget)
        b = plan_slate_budget(m, SlateLayout.STATE, budget=budget)
        win = "STATE" if b.tokens_per_candidate > a.tokens_per_candidate else "OPTIONS"
        lines.append(
            f"{m:>4} {a.option_text_tokens:>17} {a.state_tokens:>7} | "
            f"{b.tokens_per_candidate:>16} {b.state_tokens:>7} {b.instruction_tokens:>5} | "
            f"{win:>8}")
    lines += [
        "",
        "OPTIONS: the candidate signature IS the option text, so it is capped by the",
        "  head budget. At the plan's M=10 that is 16 tokens -- a title, and not a long one.",
        "STATE:   options are bare ids, so the head is cheap and the state carries the",
        "  evidence. More tokens per candidate, but the [MASK] marker sits on an empty",
        "  token and every discriminating signal has to arrive by attention.",
        "",
        "Phase 0 measures which one ranks better. Neither is obviously right.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(layout_comparison())
