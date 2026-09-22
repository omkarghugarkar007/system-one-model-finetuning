"""A teacher that is a human -- or the coding agent in the loop -- grading by hand.

Why this exists. The teacher hierarchy in Part VII is
human > consensus of several teachers > Jev > weak labels, and Jev is *not*
established as better than the student at ordinary classification (Part VI's
matched-protocol result). So for the small, high-value sets -- the Phase 0
falsification anchors, the pivot documents themselves, the confident
student/teacher disagreements -- the right teacher is a careful reader, not an
API.

The mechanism is a file round-trip, deliberately asynchronous:

    1. `grade()` hashes the request. If `<inbox>/<digest>.response.json` exists
       it is loaded and returned, and the call is indistinguishable from any
       other teacher.
    2. If it does not, a readable `<digest>.request.json` is written and
       `TeacherPending` is raised, naming the file.
    3. A human or agent fills in the response file.
    4. Re-running the experiment now completes. Nothing is recomputed, and the
       graded set accumulates as a reusable asset.

This keeps hand grading on exactly the same interface as Jev, so the ablation
"replace the teacher with a human on this subset" costs one config line.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .protocols import RUBRIC_4LEVEL, TeacherVerdict
from .base import request_digest

__all__ = ["InSessionTeacher", "TeacherPending"]


class TeacherPending(RuntimeError):
    """Raised when a hand-grading request has been written but not answered."""

    def __init__(self, path: Path, n_pending: int = 1):
        self.path = Path(path)
        self.n_pending = n_pending
        super().__init__(
            f"awaiting a hand-graded response: {self.path}\n"
            f"Fill in 'grades' (or 'level_probs') and re-run. "
            f"{n_pending} request(s) outstanding in {self.path.parent}.")


class InSessionTeacher:
    """Hand grading on the TeacherProtocol interface.

    `strict_missing=False` degrades to a neutral distribution instead of
    raising, which is useful when a large run should keep going and collect
    every outstanding request in one pass rather than stopping at the first.
    """

    def __init__(self, inbox: str | Path, rubric: Sequence[str] = RUBRIC_4LEVEL,
                 n_levels: int | None = None, strict_missing: bool = True,
                 confidence: float = 0.90):
        self.inbox = Path(inbox)
        self.inbox.mkdir(parents=True, exist_ok=True)
        self.rubric = list(rubric)
        self.n_levels = n_levels or len(self.rubric)
        self.strict_missing = strict_missing
        self.confidence = float(confidence)
        self.pending: list[Path] = []

    # ------------------------------------------------------------------ files
    def _request_path(self, digest: str) -> Path:
        return self.inbox / f"{digest}.request.json"

    def _response_path(self, digest: str) -> Path:
        return self.inbox / f"{digest}.response.json"

    def outstanding(self) -> list[Path]:
        """Requests with no matching response yet."""
        return [p for p in sorted(self.inbox.glob("*.request.json"))
                if not self._response_path(p.name.split(".")[0]).exists()]

    def _write_request(self, digest: str, query: str, sigs: list[str],
                       rubric: list[str]) -> Path:
        p = self._request_path(digest)
        p.write_text(json.dumps({
            "digest": digest,
            "how_to_answer": (
                "Write <digest>.response.json next to this file. Either give "
                "'grades': one integer per candidate on the rubric below, or "
                "'level_probs': one row of probabilities per candidate. "
                "Prefer 'level_probs' when genuinely unsure -- the student "
                "distills the distribution, and a hedged row is more useful "
                "than a confident wrong integer."),
            "rubric": {str(i): r for i, r in enumerate(rubric)},
            "query": query,
            "candidates": {f"c{i}": s for i, s in enumerate(sigs)},
            "response_template": {"grades": [0] * len(sigs)},
        }, indent=2))
        return p

    # ------------------------------------------------------------------ grade
    def _probs_from_grades(self, grades, k: int) -> np.ndarray:
        """A hand grade is a point judgement; give it a sane amount of doubt.

        A human writing "2" does not mean P(2) = 1.0. Spreading `1 - confidence`
        over the neighbouring levels keeps the ordinal structure and stops the
        student being distilled into overconfidence.
        """
        g = np.asarray(grades, dtype=int)
        if g.min() < 0 or g.max() >= k:
            raise ValueError(f"grades must lie in [0, {k - 1}], got {g.min()}..{g.max()}")
        p = np.full((g.size, k), (1.0 - self.confidence), dtype=float)
        levels = np.arange(k)
        # neighbours get the doubt, distant levels get almost none
        dist = np.abs(levels[None, :] - g[:, None])
        p = np.exp(-dist.astype(float))
        p[np.arange(g.size), g] = 0.0
        p = p / p.sum(axis=1, keepdims=True).clip(1e-9) * (1.0 - self.confidence)
        p[np.arange(g.size), g] = self.confidence
        return p / p.sum(axis=1, keepdims=True)

    def grade(self, query: str, signatures: Sequence[str],
              rubric: Sequence[str] | None = None) -> TeacherVerdict:
        rubric = list(rubric or self.rubric)
        sigs = list(signatures)
        k = len(rubric)
        digest = request_digest(query, sigs, rubric, "in_session")

        rp = self._response_path(digest)
        if rp.exists():
            d = json.loads(rp.read_text())
            if "level_probs" in d:
                p = np.asarray(d["level_probs"], dtype=float)
                p = p / p.sum(axis=1, keepdims=True).clip(1e-9)
            elif "grades" in d:
                p = self._probs_from_grades(d["grades"], k)
            else:
                raise ValueError(f"{rp} has neither 'grades' nor 'level_probs'")
            if p.shape[0] != len(sigs):
                raise ValueError(
                    f"{rp} grades {p.shape[0]} candidates, request had {len(sigs)}")
            return TeacherVerdict(np.arange(len(sigs)), p, source="in_session",
                                  meta={"digest": digest,
                                        "grader": d.get("grader", "unspecified"),
                                        "notes": d.get("notes", "")}).validate()

        path = self._write_request(digest, query, sigs, rubric)
        self.pending.append(path)
        if self.strict_missing:
            raise TeacherPending(path, len(self.outstanding()))
        # neutral: maximum entropy, so nothing downstream mistakes it for signal
        p = np.full((len(sigs), k), 1.0 / k)
        return TeacherVerdict(np.arange(len(sigs)), p, source="in_session/pending",
                              meta={"digest": digest, "pending": True}).validate()
