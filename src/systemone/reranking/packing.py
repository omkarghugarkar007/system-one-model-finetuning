"""Turning a slate of candidates into one model request. Two layouts, both tested.

`model.budget` shows that a Laya slate cannot carry rich candidate text in its
options: at M = 10 each option keeps 16 tokens. That leaves two ways to put
evidence in front of the model, and they are genuinely different experiments,
not formatting choices.

OPTIONS   options = the candidate signatures, state = the query.
          The [MASK] marker sits inside the candidate's own text, so the
          marker's hidden state is directly about that candidate. Cheap,
          local, and the way the mechanism was designed -- but 16 tokens.

STATE     options = bare ids ("c0"..."c9"), state = query + every candidate.
          ~40 tokens per candidate at M = 10, but the marker sits on a
          semantically empty token and every discriminating signal has to
          arrive by attention from the state.

One trap in the STATE layout, and it is silent. `build_sequence` truncates the
state with `st[:room]` -- from the right. A candidate list longer than the
budget therefore loses its LAST candidates entirely, and they still get
markers, still get scored, and score on nothing. The packer pre-truncates every
candidate to a fair share so the loss is even and visible instead of positional
and invisible. `StatePacker` reports it in `info["per_candidate_chars"]`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from ..model.budget import DEFAULT_BUDGET, LayaBudget, SlateLayout, plan_slate_budget

__all__ = ["PackedSlate", "SlatePacker", "OptionsPacker", "StatePacker",
           "make_packer", "DEFAULT_INSTRUCTIONS"]

DEFAULT_INSTRUCTIONS = "Which passage best answers the query?"

# Rough chars-per-token for English prose under a BPE tokenizer. Only used when
# no real token counter is supplied; the model layer passes the real one.
CHARS_PER_TOKEN = 4


@dataclass
class PackedSlate:
    instructions: str
    options: list[str]
    state: object
    info: dict = field(default_factory=dict)


class SlatePacker(Protocol):
    name: str

    def pack(self, query: str, texts: Sequence[str]) -> PackedSlate: ...


def _truncate_to_tokens(text: str, n_tokens: int,
                        counter: Callable[[str], int] | None) -> str:
    """Cut `text` to about `n_tokens`. Exact when a counter is supplied."""
    if n_tokens <= 0:
        return ""
    if counter is None:
        return text[: n_tokens * CHARS_PER_TOKEN]
    if counter(text) <= n_tokens:
        return text
    # binary search on characters -- one tokenizer call per step, ~12 steps
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter(text[:mid]) <= n_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


@dataclass
class OptionsPacker:
    """Candidates are the options. The marker sits on the candidate's own text."""
    instructions: str = DEFAULT_INSTRUCTIONS
    budget: LayaBudget = DEFAULT_BUDGET
    token_counter: Callable[[str], int] | None = None
    name: str = "options"

    def pack(self, query: str, texts: Sequence[str]) -> PackedSlate:
        texts = list(texts)
        b = plan_slate_budget(len(texts), SlateLayout.OPTIONS,
                              instruction_tokens=len(self.instructions) // CHARS_PER_TOKEN,
                              budget=self.budget)
        # build_sequence truncates option text itself, but doing it here keeps
        # what the model saw inspectable rather than buried in the tokenizer.
        opts = [_truncate_to_tokens(t, b.option_text_tokens, self.token_counter)
                for t in texts]
        return PackedSlate(self.instructions, opts, query,
                           {"layout": "options", "budget": b,
                            "option_tokens": b.option_text_tokens,
                            "truncated": [len(o) < len(t) for o, t in zip(opts, texts)]})


@dataclass
class StatePacker:
    """Candidates live in the state; options are bare ids.

    `id_prefix` stays short on purpose: every character of option text is taken
    from the head budget, and the head budget is what starves the options.
    """
    instructions: str = DEFAULT_INSTRUCTIONS
    budget: LayaBudget = DEFAULT_BUDGET
    token_counter: Callable[[str], int] | None = None
    id_prefix: str = "c"
    query_tokens_reserved: int = 20
    name: str = "state"

    def pack(self, query: str, texts: Sequence[str]) -> PackedSlate:
        texts = list(texts)
        n = len(texts)
        b = plan_slate_budget(n, SlateLayout.STATE,
                              instruction_tokens=len(self.instructions) // CHARS_PER_TOKEN,
                              budget=self.budget)
        # a fair share each, so state truncation cannot silently delete the tail
        per = max(0, (b.state_tokens - self.query_tokens_reserved
                      - 6 * n) // max(1, n))     # ~6 tokens of JSON scaffolding each
        cut = [_truncate_to_tokens(t, per, self.token_counter) for t in texts]
        ids = [f"{self.id_prefix}{i}" for i in range(n)]
        state = {"query": query, "candidates": dict(zip(ids, cut))}
        return PackedSlate(self.instructions, ids, state,
                           {"layout": "state", "budget": b,
                            "per_candidate_tokens": per,
                            "per_candidate_chars": [len(c) for c in cut],
                            "truncated": [len(c) < len(t) for c, t in zip(cut, texts)]})


def make_packer(layout: str = SlateLayout.OPTIONS, **kw) -> SlatePacker:
    if layout == SlateLayout.OPTIONS:
        return OptionsPacker(**kw)
    if layout == SlateLayout.STATE:
        return StatePacker(**kw)
    raise ValueError(f"unknown layout {layout!r}")
