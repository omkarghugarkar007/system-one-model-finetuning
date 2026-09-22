"""Teacher utilities: a content-addressed cache, and a simulated teacher.

The cache is not an optimisation. A research project that pays per judgement
needs every teacher call to be replayable, or the ablation ladder in Part IX
costs as much as the experiment did. Keying by a hash of (query, signatures,
rubric, backend) means re-running a config is free and a diff in the cache is a
diff in the science.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .protocols import RUBRIC_4LEVEL, TeacherProtocol, TeacherVerdict

__all__ = ["request_digest", "CachedTeacher", "SimulatedTeacher"]


def request_digest(query: str, signatures: Sequence[str],
                   rubric: Sequence[str], backend: str = "") -> str:
    """Stable content hash of one grading request."""
    h = hashlib.sha256()
    for part in (backend, query, *signatures, *rubric):
        h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:20]


def _verdict_to_json(v: TeacherVerdict) -> dict:
    return {"candidate_ids": np.asarray(v.candidate_ids).tolist(),
            "level_probs": np.asarray(v.level_probs).tolist(),
            "input_tokens": v.input_tokens, "cost_usd": v.cost_usd,
            "latency_ms": v.latency_ms, "source": v.source, "meta": v.meta,
            "confidence": (None if v.confidence is None
                           else np.asarray(v.confidence).tolist())}


def _verdict_from_json(d: dict) -> TeacherVerdict:
    return TeacherVerdict(
        candidate_ids=np.asarray(d["candidate_ids"]),
        level_probs=np.asarray(d["level_probs"], dtype=float),
        input_tokens=int(d.get("input_tokens", 0)),
        cost_usd=float(d.get("cost_usd", 0.0)),
        latency_ms=float(d.get("latency_ms", 0.0)),
        confidence=(None if d.get("confidence") is None
                    else np.asarray(d["confidence"], dtype=float)),
        source=d.get("source", ""), meta=d.get("meta", {}) or {})


class CachedTeacher:
    """Wrap any teacher with a content-addressed on-disk cache.

    `spent_usd` counts only what this process actually paid, so a cache hit is
    visibly free in the run manifest -- which keeps the reported cost of an
    experiment honest even after the fifth re-run.
    """

    def __init__(self, inner: TeacherProtocol, cache_dir: str | Path,
                 backend_tag: str = "", read_only: bool = False):
        self.inner = inner
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.backend_tag = backend_tag or type(inner).__name__
        self.read_only = read_only
        self.hits = 0
        self.misses = 0
        self.spent_usd = 0.0

    def path_for(self, digest: str) -> Path:
        return self.dir / f"{digest}.json"

    def grade(self, query: str, signatures: Sequence[str],
              rubric: Sequence[str] = RUBRIC_4LEVEL) -> TeacherVerdict:
        sigs, rub = list(signatures), list(rubric)
        digest = request_digest(query, sigs, rub, self.backend_tag)
        p = self.path_for(digest)
        if p.exists():
            self.hits += 1
            v = _verdict_from_json(json.loads(p.read_text()))
            v.meta = dict(v.meta, cache="hit")
            return v
        if self.read_only:
            raise KeyError(f"cache miss for {digest} and read_only=True")
        v = self.inner.grade(query, sigs, rub)
        self.misses += 1
        self.spent_usd += float(v.cost_usd)
        p.write_text(json.dumps(_verdict_to_json(v), indent=1))
        return v

    def ask(self, state, qtype, instructions, criteria=None):
        """Cached passthrough for the generic typed-question path."""
        import numpy as _np
        digest = request_digest(str(state), [qtype, str(instructions)],
                                [str(criteria)], self.backend_tag + "/ask")
        p = self.path_for(digest)
        if p.exists():
            self.hits += 1
            return _np.asarray(json.loads(p.read_text())["probs"], dtype=float)
        if self.read_only:
            raise KeyError(f"cache miss for {digest} and read_only=True")
        probs = self.inner.ask(state, qtype, instructions, criteria)
        self.misses += 1
        p.write_text(json.dumps({"probs": _np.asarray(probs).tolist()}))
        return _np.asarray(probs, dtype=float)

    def stats(self) -> dict:
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / n if n else 0.0,
                "spent_usd": round(self.spent_usd, 6)}


class SimulatedTeacher:
    """An oracle with controllable error. For testing the cascade, not the model.

    `skill` in [0, 1] interpolates between a uniform distribution and a
    one-hot on the true grade, then a Dirichlet-style softening produces a
    plausible distribution. `bias` shifts the mode, which is how you test
    whether the controller survives a teacher that is wrong in a consistent
    direction -- the failure mode a real teacher actually has.
    """

    def __init__(self, true_grades: dict[str, np.ndarray] | None = None,
                 skill: float = 0.9, bias: float = 0.0, n_levels: int = 4,
                 seed: int = 0):
        self.true = true_grades or {}
        self.skill = float(np.clip(skill, 0.0, 1.0))
        self.bias = bias
        self.n_levels = n_levels
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def grade(self, query: str, signatures: Sequence[str],
              rubric: Sequence[str] = RUBRIC_4LEVEL) -> TeacherVerdict:
        sigs = list(signatures)
        k = len(rubric)
        g = np.asarray(self.true.get(query, np.zeros(len(sigs))), dtype=float)
        g = np.resize(g, len(sigs)) + self.bias
        levels = np.arange(k, dtype=float)
        # a peak at the true grade, width set by skill
        width = 0.35 + 3.0 * (1.0 - self.skill)
        logits = -((levels[None, :] - g[:, None]) ** 2) / (2 * width ** 2)
        logits += self.rng.normal(0.0, 0.25 * (1.0 - self.skill) + 1e-6, logits.shape)
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        self.calls += 1
        return TeacherVerdict(np.arange(len(sigs)), p, input_tokens=0,
                              cost_usd=0.0, source="simulated").validate()
