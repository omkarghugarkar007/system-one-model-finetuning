"""Jev as a large-slate teacher, over the TypeSafe System One API.

Two routes to the same endpoint shape, chosen by `provider`:

    openrouter  POST https://openrouter.ai/api/v1/systemone   (OPENROUTER_API_KEY)
    typesafe    POST https://api.typesafe.ai/v1/systemone     (TYPESAFE_API_KEY)

Both take `{state, model, questions}` and return `{model, answers, usage}`.
OpenRouter additionally returns `usage.cost` in dollars, which we record rather
than recompute -- a measured price beats a modelled one.

Note that Jev is NOT listed in OpenRouter's `/api/v1/models` response: its
modality is `text->decisions`, and that listing only returns `text->text`
models. Query `/api/v1/models/typesafe/jev-1.13/endpoints` directly.

One request, one shared state, one Score question per candidate. That is the
configuration measured at 0.692 nDCG@10 for $0.45/1k in jev-rerank-bench, and
it beat both the single Choice and every per-pair variant. TypeSafe's own
re-ranking cookbook uses one request per query-candidate pair -- 1,200 calls
for 40 queries. The vendor's recipe is the expensive path; do not copy it.

Latency is flat in option count, so there is no reason to send fewer candidates
than the frontier holds.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Sequence

import numpy as np

from .protocols import RUBRIC_4LEVEL, TeacherVerdict

__all__ = ["JevTeacher", "JevError", "PROVIDERS"]

PROVIDERS = {
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/systemone",
        "key_env": "OPENROUTER_API_KEY",
        "model": "typesafe/jev-1.13",
    },
    "typesafe": {
        "url": "https://api.typesafe.ai/v1/systemone",
        "key_env": "TYPESAFE_API_KEY",
        "model": "jev-latest",
    },
}

# $0.042 per 1M input tokens, output free. Used only when the provider does not
# report a cost of its own.
PRICE_PER_MTOK = 0.042


class JevError(RuntimeError):
    pass


class JevTeacher:
    """Batched rubric grading: one shared state, one Score question per candidate."""

    MAX_LEVELS = 10       # the Score primitive accepts 2-10 levels
    MAX_OPTIONS = 255     # Choice caps at 255; hard reject at 256

    def __init__(self, provider: str = "openrouter", api_key: str | None = None,
                 model: str | None = None, rubric: Sequence[str] = RUBRIC_4LEVEL,
                 max_candidates: int = 30, timeout: float = 90.0,
                 max_retries: int = 4, url: str | None = None):
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(PROVIDERS)}")
        cfg = PROVIDERS[provider]
        self.provider = provider
        self.url = url or cfg["url"]
        self.model = model or cfg["model"]
        self.api_key = api_key or os.environ.get(cfg["key_env"], "")
        if not self.api_key:
            raise JevError(
                f"no API key: set {cfg['key_env']} or pass api_key=. "
                f"Never hard-code it.")
        self.rubric = list(rubric)
        if not 2 <= len(self.rubric) <= self.MAX_LEVELS:
            raise ValueError(f"Score takes 2-{self.MAX_LEVELS} levels, "
                             f"got {len(self.rubric)}")
        self.max_candidates = min(max_candidates, self.MAX_OPTIONS)
        self.timeout = timeout
        self.max_retries = max_retries

    # ---------------------------------------------------------------- request
    def _build_payload(self, query: str, sigs: list[str],
                       rubric: list[str]) -> dict:
        # The whole shortlist travels in ONE state, so the model sees candidates
        # in each other's context. Keys are ours; TypeSafe documents that the
        # question id is not sent to the model and does not affect inference.
        state = {"query": query,
                 "candidates": {f"c{i}": s for i, s in enumerate(sigs)}}
        questions = {
            f"c{i}": {
                "type": "score",
                "instructions": {
                    "question": (
                        f"Rate how well candidate `c{i}` in `candidates` answers "
                        f"the `query`. Judge only whether it answers the query -- "
                        f"not whether it is well written, recent, or authoritative."),
                    "candidate_id": f"c{i}",
                },
                "criteria": rubric,
            }
            for i in range(len(sigs))
        }
        return {"state": state, "model": self.model, "questions": questions}

    def _post(self, payload: dict) -> tuple[dict, float]:
        body = json.dumps(payload).encode()
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        if self.provider == "openrouter":
            headers["X-Title"] = "FrontierRank"
        last = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(self.url, data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode()), (time.perf_counter() - t0) * 1e3
            except urllib.error.HTTPError as e:
                detail = e.read().decode()[:400]
                last = JevError(f"HTTP {e.code} from {self.url}: {detail}")
                # 4xx other than rate-limiting is our bug; do not burn retries
                if e.code not in (408, 409, 425, 429) and e.code < 500:
                    raise last from None
            except Exception as e:                      # noqa: BLE001 - network
                last = JevError(f"{type(e).__name__}: {e}")
            time.sleep(min(2.0 ** attempt, 8.0))
        raise last if last else JevError("request failed with no error recorded")

    # ------------------------------------------------------------------ grade
    def grade(self, query: str, signatures: Sequence[str],
              rubric: Sequence[str] | None = None) -> TeacherVerdict:
        rubric = list(rubric or self.rubric)
        if not 2 <= len(rubric) <= self.MAX_LEVELS:
            raise ValueError(f"Score takes 2-{self.MAX_LEVELS} levels")
        sigs = list(signatures)[: self.max_candidates]
        if not sigs:
            raise ValueError("nothing to grade")

        resp, latency = self._post(self._build_payload(query, sigs, rubric))
        answers = resp.get("answers")
        if not isinstance(answers, dict):
            raise JevError(f"no answers in response: {str(resp)[:300]}")

        n_levels = len(rubric)
        probs = np.zeros((len(sigs), n_levels), dtype=float)
        conf = np.full(len(sigs), np.nan)
        for i in range(len(sigs)):
            a = answers.get(f"c{i}")
            if a is None:
                raise JevError(f"missing answer for c{i}")
            p = a.get("probabilities")
            if not p:
                raise JevError(f"answer c{i} carried no distribution: {a}")
            for lvl, v in p.items():
                probs[i, int(lvl)] = float(v)
            if a.get("confidence") is not None:
                conf[i] = float(a["confidence"])
        # the API rounds to 4dp, so rows are close to but not exactly 1
        probs = probs / probs.sum(axis=1, keepdims=True).clip(1e-9)

        usage = resp.get("usage", {}) or {}
        tok = int(usage.get("input_tokens", 0))
        cost = usage.get("cost")
        cost = float(cost) if cost is not None else tok / 1e6 * PRICE_PER_MTOK

        return TeacherVerdict(
            candidate_ids=np.arange(len(sigs)),
            level_probs=probs,
            input_tokens=tok,
            cost_usd=cost,
            latency_ms=latency,
            confidence=conf,
            source=f"jev/{self.provider}/{resp.get('model', self.model)}",
            meta={"response_model": resp.get("model"), "id": resp.get("id"),
                  "cost_reported": usage.get("cost") is not None},
        ).validate()

    # ------------------------------------------------------------------ ask
    def ask(self, state, qtype: str, instructions, criteria=None) -> np.ndarray:
        """Ask the teacher the SAME typed question you will ask the student.

        This is the distillation path for ordinary fine-tuning, and it is
        distinct from `grade()` on purpose. `grade()` is reranking-shaped: many
        candidates against one rubric. Distilling a classification or rating
        question through it means passing the options as both the candidates
        and the rubric, which asks the teacher a question nobody meant and
        returns a distribution over the wrong thing.

        Returns probabilities over the question's options, in option order.
        """
        if qtype not in ("choice", "score", "noul"):
            raise ValueError(f"unknown question type {qtype!r}")
        q: dict = {"type": qtype, "instructions": instructions}
        if qtype == "choice":
            crit = (criteria if isinstance(criteria, dict)
                    else {c: None for c in (criteria or [])})
            if not 2 <= len(crit) <= self.MAX_OPTIONS:
                raise ValueError(f"choice takes 2-{self.MAX_OPTIONS} options")
            q["criteria"] = crit
            keys = list(crit)
        elif qtype == "score":
            levels = list(criteria or [])
            if not 2 <= len(levels) <= self.MAX_LEVELS:
                raise ValueError(f"score takes 2-{self.MAX_LEVELS} levels")
            q["criteria"] = levels
            keys = [str(i) for i in range(len(levels))]
        else:
            if criteria:
                q["criteria"] = criteria
            keys = ["false", "true"]

        resp, _ = self._post({"state": state, "model": self.model,
                              "questions": {"q": q}})
        a = (resp.get("answers") or {}).get("q")
        if a is None:
            raise JevError(f"no answer: {str(resp)[:300]}")

        if qtype == "noul":
            p1 = float(a.get("noul", 0.5))
            return np.array([1.0 - p1, p1])
        probs = a.get("probabilities")
        if not probs:
            raise JevError(f"answer carried no distribution: {a}")
        out = np.array([float(probs.get(k, 0.0)) for k in keys], dtype=float)
        return out / out.sum() if out.sum() > 0 else np.full(len(keys), 1.0 / len(keys))
