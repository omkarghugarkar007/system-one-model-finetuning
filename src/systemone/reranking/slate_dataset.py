"""Training slates that mirror inference slates, or the calibration will not transfer.

The rule that drives every choice here: a slate the model trains on must look
like a slate it will score. Same M, same anchors in the same proportion, same
packer, same signature budget. Train on 30-candidate slates and score on
10-option ones and the fitted temperature is meaningless.

Slate difficulty is varied on purpose (plan Part IX): all-hard, mixed and easy
slates in fixed proportion. An all-mixed diet teaches the model that every
slate contains one good answer, which is the one thing that is reliably false
at inference -- and it starves the act head of the contested slates it needs to
learn when the frontier is genuinely in doubt.

Option order is re-randomised every epoch. The marker mechanism is
position-robust by construction, but F4 measured 33% top-1 movement under
reordering on the base checkpoint, so "robust" is not "invariant".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["SlateExample", "SlateSpec", "SlateDataset", "collate_slates"]


@dataclass
class SlateExample:
    query_id: str
    query: str
    texts: list[str]              # candidate signatures, then anchor texts
    grades: np.ndarray            # graded relevance for candidates; -1 for anchors
    anchor_utility: np.ndarray    # utility for anchors; nan for candidates
    is_anchor: np.ndarray         # bool mask
    difficulty: str = "mixed"
    doc_ids: list = field(default_factory=list)

    @property
    def n_options(self) -> int:
        return len(self.texts)

    def permuted(self, rng: np.random.Generator) -> SlateExample:
        p = rng.permutation(self.n_options)
        return SlateExample(self.query_id, self.query, [self.texts[i] for i in p],
                            self.grades[p], self.anchor_utility[p],
                            self.is_anchor[p], self.difficulty,
                            [self.doc_ids[i] for i in p] if self.doc_ids else [])


@dataclass
class SlateSpec:
    max_options: int = 10
    n_anchors: int = 4
    depth: int = 100
    slates_per_query: int = 8
    # all-hard slates have no strong answer; easy ones have exactly one. Both
    # occur at inference and neither is the majority.
    difficulty_mix: tuple = (("hard", 0.3), ("mixed", 0.5), ("easy", 0.2))

    @property
    def candidates_per_slate(self) -> int:
        n = self.max_options - self.n_anchors
        if n < 1:
            raise ValueError("anchors consume the entire slate")
        return n


class SlateDataset:
    """Builds training slates from a corpus, a first stage and an anchor pool.

    Anchors are appended last here and shuffled by `permuted()`, so nothing
    downstream may assume a position. `is_anchor` is the only reliable way to
    tell them apart, and anchors are excluded from every ranking loss: they are
    the ruler, not the thing being measured.
    """

    def __init__(self, ds, index, signature_builder, anchor_pool,
                 spec: SlateSpec | None = None, query_ids=None,
                 tokens_per_candidate: int = 37, seed: int = 0):
        self.ds = ds
        self.index = index
        self.sig = signature_builder
        self.anchors = anchor_pool
        self.spec = spec or SlateSpec()
        self.tokens_per_candidate = tokens_per_candidate
        self.rng = np.random.default_rng(seed)
        self.query_ids = list(query_ids) if query_ids is not None else ds.query_ids
        self._pools: dict[str, tuple] = {}

    # ------------------------------------------------------------------ pools
    def pool(self, qid: str):
        """First-stage pool for a query, with grades. Cached: BM25 is not free."""
        if qid not in self._pools:
            query = self.ds.queries[qid]
            ranked, _ = self.index.search(query, self.spec.depth)
            ranked = [d for d in ranked if d not in self.anchors.held_out]
            grades = np.array([self.ds.grade(qid, d) for d in ranked], dtype=float)
            self._pools[qid] = (ranked, grades)
        return self._pools[qid]

    def _sample_candidates(self, grades: np.ndarray, difficulty: str,
                           n: int) -> np.ndarray:
        pos = np.flatnonzero(grades >= 2)
        weak = np.flatnonzero(grades == 1)
        neg = np.flatnonzero(grades <= 0)
        rng = self.rng

        def take(a, k):
            k = min(k, a.size)
            return rng.choice(a, k, replace=False) if k else np.array([], dtype=int)

        if difficulty == "easy" and pos.size:
            sel = np.concatenate([take(pos, 1), take(neg, n - 1)])
        elif difficulty == "hard":
            # no clear answer: weak and negative only, which is the slate where
            # a reranker is most likely to invent a winner
            sel = np.concatenate([take(weak, n // 2), take(neg, n - n // 2)])
        else:
            sel = np.concatenate([take(pos, max(1, n // 4)), take(weak, n // 3),
                                  take(neg, n)])
        sel = np.unique(sel)[:n]
        if sel.size < n:                      # top up from anywhere
            rest = np.setdiff1d(np.arange(grades.size), sel)
            sel = np.concatenate([sel, take(rest, n - sel.size)])
        return sel[:n]

    def _difficulty(self) -> str:
        names = [d for d, _ in self.spec.difficulty_mix]
        probs = np.array([w for _, w in self.spec.difficulty_mix], dtype=float)
        return str(self.rng.choice(names, p=probs / probs.sum()))

    # --------------------------------------------------------------- building
    def build_query(self, qid: str) -> list[SlateExample]:
        query = self.ds.queries[qid]
        ranked, grades = self.pool(qid)
        if len(ranked) < self.spec.candidates_per_slate:
            return []
        anchors = self.anchors.for_query(query, self.ds, qid,
                                         self.spec.n_anchors, self.rng)
        out = []
        for _ in range(self.spec.slates_per_query):
            diff = self._difficulty()
            sel = self._sample_candidates(grades, diff,
                                          self.spec.candidates_per_slate)
            texts, dids = [], []
            for i in sel:
                d = ranked[int(i)]
                texts.append(self.sig.build(query, self.ds.docs[d].title,
                                            self.ds.docs[d].text,
                                            self.tokens_per_candidate))
                dids.append(d)
            g = list(grades[sel])
            u = [np.nan] * len(sel)
            mask = [False] * len(sel)
            for a in anchors:
                texts.append(a.text)
                dids.append(a.doc_id)
                g.append(-1.0)
                u.append(a.utility)
                mask.append(True)
            out.append(SlateExample(qid, query, texts, np.asarray(g, float),
                                    np.asarray(u, float), np.asarray(mask, bool),
                                    diff, dids))
        return out

    def build(self, shuffle: bool = True) -> list[SlateExample]:
        ex = [e for qid in self.query_ids for e in self.build_query(qid)]
        if shuffle:
            self.rng.shuffle(ex)
        return ex

    def stats(self, examples) -> dict:
        if not examples:
            return {}
        g = np.concatenate([e.grades[~e.is_anchor] for e in examples])
        diffs = [e.difficulty for e in examples]
        return {"slates": len(examples),
                "queries": len({e.query_id for e in examples}),
                "options_per_slate": examples[0].n_options,
                "candidates_per_slate": int((~examples[0].is_anchor).sum()),
                "grade_mean": round(float(g.mean()), 3),
                "grade_hist": {int(k): int(v) for k, v in
                               zip(*np.unique(g, return_counts=True), strict=False)},
                "slates_with_a_positive": round(float(np.mean(
                    [(e.grades[~e.is_anchor] >= 2).any() for e in examples])), 3),
                "difficulty": {d: diffs.count(d) / len(diffs) for d in set(diffs)}}


def collate_slates(examples, runtime, packer, qtype: str = "choice"):
    """SlateExamples -> the tensors `FrontierRankModel.forward` wants."""
    import torch

    built, keep = [], []
    for e in examples:
        packed = packer.pack(e.query, e.texts)
        try:
            built.append(runtime.build(packed.state, qtype, packed.instructions,
                                       list(packed.options), rendered=True))
            keep.append(e)
        except ValueError:
            continue                    # options did not fit; drop, never truncate
    if not built:
        return None

    n = len(built)
    L = max(len(b[0]) for b in built)
    kmax = max(len(b[1]) for b in built)
    pad = runtime.tok.pad_token_id

    ids = torch.full((n, L), pad, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    grades = torch.full((n, kmax), -1.0)
    anchor = torch.zeros((n, kmax), dtype=torch.bool)
    util = torch.full((n, kmax), float("nan"))

    for i, ((seq, mk, _), e) in enumerate(zip(built, keep, strict=False)):
        ids[i, :len(seq)] = torch.tensor(seq)
        att[i, :len(seq)] = 1
        k = len(mk)
        mpos[i, :k] = torch.tensor(mk)
        mmask[i, :k] = True
        grades[i, :k] = torch.tensor(e.grades[:k], dtype=torch.float32)
        anchor[i, :k] = torch.tensor(e.is_anchor[:k])
        util[i, :k] = torch.tensor(e.anchor_utility[:k], dtype=torch.float32)

    from ..model.sequence import QTYPES
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos,
            "marker_mask": mmask, "grades": grades, "is_anchor": anchor,
            "anchor_utility": util,
            "qtype": torch.full((n,), QTYPES[qtype], dtype=torch.long),
            "examples": keep}
