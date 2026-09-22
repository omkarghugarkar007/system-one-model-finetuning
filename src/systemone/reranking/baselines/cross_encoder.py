"""A pointwise cross-encoder baseline. The honest competition.

`plan.md` names this comparison itself and calls it uncomfortable: a 6-layer
MiniLM cross-encoder lifts BEIR from 0.408 to 0.475 reranking BM25 top-100 --
"the largest per-parameter gain anywhere in this document, available this
afternoon, at 1,800 documents/second". Everything FrontierRank does is chasing
points beyond that, so the row has to be measured rather than cited.

Two properties matter here and they pull in opposite directions.

**Cost.** N forward passes over (query, doc) pairs, but each pair is short and
the model is 22M parameters. Laya pays 17 passes of 512 tokens per 100
candidates; MiniLM pays 100 passes of ~180 tokens over a model 19x smaller.
Which is cheaper is an empirical question, not an obvious one.

**Calibration.** This is the axis FrontierRank actually depends on. A
cross-encoder emits a logit trained with binary cross-entropy against MS MARCO
labels; `sigmoid` of it is a probability only by courtesy, and it is fitted to a
different corpus. Whether it is calibrated *on your data* is the question, and
it is what `ece`/`brier` in the baseline runner measure.
"""
from __future__ import annotations

import time

import numpy as np
import torch

__all__ = ["CrossEncoderReranker"]


class CrossEncoderReranker:
    """HF sequence-classification cross-encoder, scored pointwise."""

    def __init__(self, model_dir: str, device=None, max_length: int = 512,
                 batch_size: int = 32, name: str = "cross-encoder"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from ...model.runtime import pick_device
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.name = name
        self.n_pairs = 0
        self.total_ms = 0.0

    @torch.no_grad()
    def score(self, query: str, documents) -> np.ndarray:
        """Raw relevance logits, one per document. Higher is better."""
        docs = list(documents)
        out = np.empty(len(docs), dtype=float)
        for i in range(0, len(docs), self.batch_size):
            chunk = docs[i:i + self.batch_size]
            enc = self.tok([query] * len(chunk), chunk, padding=True,
                           truncation=True, max_length=self.max_length,
                           return_tensors="pt").to(self.device)
            t0 = time.perf_counter()
            logits = self.model(**enc).logits
            if self.device.type == "mps":
                torch.mps.synchronize()
            self.total_ms += (time.perf_counter() - t0) * 1e3
            self.n_pairs += len(chunk)
            # single-logit heads are relevance; 2-logit heads are (neg, pos)
            z = logits.float().cpu().numpy()
            out[i:i + len(chunk)] = z[:, 0] if z.shape[1] == 1 else z[:, 1] - z[:, 0]
        return out

    def probability(self, query: str, documents) -> np.ndarray:
        """P(relevant) by sigmoid. A probability by courtesy, not by training.

        The head was fitted with binary cross-entropy on MS MARCO, so this is
        calibrated for MS MARCO's positive rate and nothing else. Measuring its
        ECE on a different corpus is the point of including it.
        """
        return 1.0 / (1.0 + np.exp(-self.score(query, documents)))

    def throughput(self) -> dict:
        return {"pairs": self.n_pairs, "total_ms": round(self.total_ms, 1),
                "ms_per_pair": round(self.total_ms / max(1, self.n_pairs), 3),
                "pairs_per_s": round(self.n_pairs / max(1e-9, self.total_ms / 1e3), 1),
                "device": str(self.device)}
