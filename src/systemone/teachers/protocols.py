"""The two interfaces every model plugs into. Nothing else in the package names a vendor.

ScorerProtocol is the student: something that turns one slate into a
distribution over its options. Laya implements it, a simulated Luce model
implements it, and a Jev `Choice` call implements it too -- which is what makes
"is the local model good enough?" a measurable question rather than an
architectural commitment.

TeacherProtocol is the escalation target. It is deliberately not the same
interface: the teacher returns *graded* judgements on an absolute rubric, not a
slate-normalised distribution, because that is the whole reason it is worth
paying for. A Choice softmax hides its offset (see `reranking.anchor_fit`); a rubric
grade does not.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["ScorerProtocol", "TeacherProtocol", "TeacherVerdict", "SlateRequest",
           "RUBRIC_4LEVEL", "expected_grade"]


# The TREC DL graded-relevance scale, verbatim. Not an analogy -- the same four
# levels, which is what lets a teacher judgement be used directly as a training
# label and compared against a qrel.
RUBRIC_4LEVEL = [
    "irrelevant: the passage has nothing to do with the query",
    "related to the query's topic but does not answer it",
    "partially answers the query, or the answer is present but buried in "
    "extraneous material",
    "directly and completely answers the query",
]


@dataclass
class SlateRequest:
    """One slate, as handed to a scorer. Carries provenance for the run log."""
    instructions: str
    options: list[str]
    state: str
    slate_index: int = 0
    query_id: str = ""

    def __post_init__(self):
        self.options = list(self.options)


@runtime_checkable
class ScorerProtocol(Protocol):
    def choice_logprobs(self, instructions: str, options: Sequence[str],
                        state: str) -> np.ndarray:
        """Log-probabilities over `options`, normalised across them.

        Must return exactly `len(options)` finite values summing (in exp space)
        to 1. Callers rely on that: `reranking.anchor_fit` fits the per-slate offset
        from anchor positions, and a scorer that silently drops an option
        silently corrupts the fit.
        """
        ...


@dataclass
class TeacherVerdict:
    """Graded judgements on an absolute rubric, plus what they cost.

    `level_probs` is the object of value, not `expected_grade`. Distilling
    P(0)=.01, P(1)=.04, P(2)=.21, P(3)=.74 tells the student where its own
    uncertainty should live; distilling the argmax throws that away and is the
    most reliable way to produce a confident, badly calibrated student.
    """
    candidate_ids: np.ndarray
    level_probs: np.ndarray              # (n_candidates, n_levels), rows sum to 1
    input_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    confidence: np.ndarray | None = None  # vendor-reported, if any
    source: str = ""                      # which backend produced this
    meta: dict = field(default_factory=dict)

    @property
    def expected_grade(self) -> np.ndarray:
        return expected_grade(self.level_probs)

    @property
    def n_levels(self) -> int:
        return int(self.level_probs.shape[1])

    def validate(self) -> TeacherVerdict:
        p = np.asarray(self.level_probs, dtype=float)
        if p.ndim != 2:
            raise ValueError(f"level_probs must be 2-D, got {p.shape}")
        if p.shape[0] != len(self.candidate_ids):
            raise ValueError(
                f"{p.shape[0]} judgements for {len(self.candidate_ids)} candidates")
        if not np.all(np.isfinite(p)):
            raise ValueError("teacher returned non-finite probabilities")
        s = p.sum(axis=1)
        if not np.allclose(s, 1.0, atol=1e-3):
            raise ValueError(f"teacher rows do not sum to 1 (min {s.min():.4f}, "
                             f"max {s.max():.4f})")
        return self


def expected_grade(level_probs) -> np.ndarray:
    p = np.asarray(level_probs, dtype=float)
    return p @ np.arange(p.shape[-1], dtype=float)


@runtime_checkable
class TeacherProtocol(Protocol):
    def grade(self, query: str, signatures: Sequence[str],
              rubric: Sequence[str] = RUBRIC_4LEVEL) -> TeacherVerdict:
        """Grade every signature against `query` on an ordered rubric."""
        ...
