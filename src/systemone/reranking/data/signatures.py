"""The evidence slicer: one document -> the few tokens the model actually reads.

Under `model.budget` a Laya slate gives each candidate 16 tokens (OPTIONS
layout) or ~40 (STATE). At that size "feed it the passage" is not an option,
and the question stops being "how good is the reranker?" and becomes "did the
right 16 tokens make it into the slate?". Those are different failures with
different fixes, and the plan's `read_more` action only makes sense once they
are measured separately.

So signature construction is a first-class, swappable, testable component with
an explicit token budget -- not a formatting step. The strategies below are
deliberately all deterministic and LLM-free: the BM25 term positions and the
document structure are already in hand from retrieval.

    TitleOnly       the title, truncated. The cheapest baseline and, at 16
                    tokens, a surprisingly hard one to beat.
    Head            the document's first tokens. What a naive implementation
                    does, and the control that shows whether selection matters.
    LexicalWindow   the window of the document with the highest query-term IDF
                    mass. Query-dependent, so the same document yields
                    different evidence for different queries -- which is the
                    whole point, and also why it cannot be precomputed once.
    TitleThenWindow title first, remaining budget to the best window.

`recall` measures the slicer on its own terms: does the signature still contain
the query terms the full document matched on? That is separable from ranking
quality, and Part XI asks for exactly that separation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from .bm25 import tokenize

__all__ = ["SignatureBuilder", "TitleOnly", "Head", "LexicalWindow",
           "TitleThenWindow", "make_signature_builder", "signature_recall"]

_WORD = re.compile(r"\S+")

# chars per token under Laya's BPE tokenizer for English prose; only used when
# no exact counter is supplied
CHARS_PER_TOKEN = 4


def _cut(text: str, budget_tokens: int,
         counter: Callable[[str], int] | None) -> str:
    if budget_tokens <= 0:
        return ""
    if counter is None:
        return text[: budget_tokens * CHARS_PER_TOKEN].strip()
    if counter(text) <= budget_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter(text[:mid]) <= budget_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].strip()


class SignatureBuilder(Protocol):
    name: str

    def build(self, query: str, title: str, text: str,
              budget_tokens: int) -> str: ...


@dataclass
class TitleOnly:
    counter: Callable[[str], int] | None = None
    name: str = "title"

    def build(self, query, title, text, budget_tokens) -> str:
        return _cut(title or text, budget_tokens, self.counter)


@dataclass
class Head:
    counter: Callable[[str], int] | None = None
    name: str = "head"

    def build(self, query, title, text, budget_tokens) -> str:
        return _cut(f"{title}. {text}".strip(". "), budget_tokens, self.counter)


@dataclass
class LexicalWindow:
    """The span with the most query-term IDF mass.

    `idf` maps a term to its inverse document frequency; without it every
    matched term counts the same, which over-rewards windows dense in common
    query words. Pass `BM25Index.idf` via `idf_lookup` for the real thing.
    """
    counter: Callable[[str], int] | None = None
    idf_lookup: Callable[[str], float] | None = None
    window_words: int = 40
    stride: int = 10
    name: str = "lexical"

    def _best_window(self, query: str, text: str) -> str:
        words = _WORD.findall(text)
        if not words:
            return ""
        qterms = set(tokenize(query))
        if not qterms or len(words) <= self.window_words:
            return " ".join(words[: self.window_words])

        weight = {}
        for t in qterms:
            weight[t] = self.idf_lookup(t) if self.idf_lookup else 1.0

        norm = [tokenize(w, drop_stopwords=False) for w in words]
        hit = np.array([max((weight.get(t, 0.0) for t in ws), default=0.0)
                        for ws in norm])
        if hit.sum() == 0:
            return " ".join(words[: self.window_words])

        # rolling sum over the window, sampled at `stride`
        csum = np.concatenate([[0.0], np.cumsum(hit)])
        best_i, best_v = 0, -1.0
        for i in range(0, max(1, len(words) - self.window_words + 1), self.stride):
            v = csum[min(i + self.window_words, len(words))] - csum[i]
            if v > best_v:
                best_i, best_v = i, v
        return " ".join(words[best_i:best_i + self.window_words])

    def build(self, query, title, text, budget_tokens) -> str:
        return _cut(self._best_window(query, text), budget_tokens, self.counter)


@dataclass
class TitleThenWindow:
    """Title first, remaining budget to the best window.

    At 16 tokens this usually degenerates to the title, which is itself a
    finding worth reporting rather than a bug to hide.
    """
    counter: Callable[[str], int] | None = None
    idf_lookup: Callable[[str], float] | None = None
    title_share: float = 0.5
    window_words: int = 40
    stride: int = 10
    name: str = "title+lexical"

    def __post_init__(self):
        self._win = LexicalWindow(self.counter, self.idf_lookup,
                                  self.window_words, self.stride)

    def build(self, query, title, text, budget_tokens) -> str:
        t_budget = int(budget_tokens * self.title_share)
        head = _cut(title, t_budget, self.counter) if title else ""
        used = (self.counter(head) if self.counter
                else len(head) // CHARS_PER_TOKEN) if head else 0
        rest = max(0, budget_tokens - used - 1)
        tail = _cut(self._win._best_window(query, text), rest, self.counter)
        return f"{head} {tail}".strip() if head else tail


def make_signature_builder(kind: str = "title+lexical", **kw) -> SignatureBuilder:
    """Build a slicer by name, ignoring kwargs it has no use for.

    Callers sweep strategies in a loop and pass one kwarg bag for all of them;
    `TitleOnly` has no use for `idf_lookup` and should not have to pretend it
    does just to be swappable.
    """
    cls = {"title": TitleOnly, "head": Head, "lexical": LexicalWindow,
           "title+lexical": TitleThenWindow}[kind]
    accepted = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in kw.items() if k in accepted})


def signature_recall(query: str, signature: str, full_text: str) -> float:
    """Fraction of the document's query-term matches that survived slicing.

    1.0 means the slice kept every query term the full document had. This is
    the slicer's own metric, and it is what separates "the model could not
    tell" from "the model was never shown it".
    """
    q = set(tokenize(query))
    if not q:
        return 1.0
    in_doc = q & set(tokenize(full_text))
    if not in_doc:
        return 1.0
    return len(in_doc & set(tokenize(signature))) / len(in_doc)
