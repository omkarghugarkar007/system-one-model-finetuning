"""Typed decisions: the unit of training data for a System One model.

A System One model answers *typed questions about a state*. Fine-tuning it
means showing it questions of the shape you will ask at inference, with labels.
That is the whole data contract, and it deliberately mirrors the Jev / Laya
request shape so anything you can ask at inference you can train on:

    choice   pick one option from a named set   -> the option
    score    rate against ordered levels        -> a level, and levels are ORDERED
    noul     a yes/no question                  -> a probability

Three rules this module enforces, each learned the hard way:

**Train the question you will ask.** The rendered option text, the instructions
and the option count all go into the same 192-token head budget, so a question
trained with three short options and asked with ten long ones is not the same
question. `TypedDataset.budget_report()` tells you what the tokenizer actually
kept.

**Option order is randomised for `choice` and `noul`, never for `score`.** The
marker mechanism is position-robust by construction, but measured order
sensitivity on a base checkpoint was 33%, so robustness is trained rather than
assumed. Score levels are ordinal: shuffling them destroys the only structure
that makes "off by one" cheaper than "off by three".

**Soft labels beat hard ones.** If you have a teacher, store its full
distribution in `soft_label`. P(0)=.01, P(1)=.04, P(2)=.21, P(3)=.74 tells the
student where its own uncertainty belongs; the argmax throws that away and is
the most reliable way to produce a confident, badly calibrated model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..model.sequence import QTYPES, build_sequence, render_options

__all__ = ["Question", "TypedExample", "TypedDataset", "choice", "score", "noul",
           "collate_typed", "load_jsonl"]


@dataclass
class Question:
    """One typed question, in the same shape the inference API takes."""
    type: str                       # "choice" | "score" | "noul"
    instructions: str | dict | list
    criteria: dict | list | None = None

    def __post_init__(self):
        if self.type not in QTYPES:
            raise ValueError(f"type must be one of {sorted(QTYPES)}, got {self.type!r}")
        if self.type == "choice" and not self.criteria:
            raise ValueError("a choice needs criteria (its options)")
        if self.type == "score":
            if not isinstance(self.criteria, (list, tuple)) or len(self.criteria) < 2:
                raise ValueError("a score needs an ordered list of >= 2 levels")
            if len(self.criteria) > 10:
                raise ValueError("the Score primitive accepts at most 10 levels")

    @property
    def options(self) -> list[str]:
        return render_options(self.type, self.criteria)

    @property
    def n_options(self) -> int:
        return len(self.options)

    @property
    def ordinal(self) -> bool:
        """Score levels are ordered; choice options are not. This decides
        whether the order may be shuffled and whether an ordinal loss applies."""
        return self.type == "score"


@dataclass
class TypedExample:
    state: str | dict | list
    question: Question
    label: int | None = None                 # index into options / levels
    soft_label: np.ndarray | None = None     # teacher distribution over options
    weight: float = 1.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        k = self.question.n_options
        if self.label is not None and not 0 <= self.label < k:
            raise ValueError(f"label {self.label} outside 0..{k - 1}")
        if self.soft_label is not None:
            p = np.asarray(self.soft_label, dtype=float)
            if p.size != k:
                raise ValueError(f"soft_label has {p.size} entries for {k} options")
            self.soft_label = p / p.sum()
        if self.label is None and self.soft_label is None:
            raise ValueError("an example needs a label or a soft_label")

    @property
    def target(self) -> np.ndarray:
        """The distribution to train against. Soft if available, else one-hot."""
        if self.soft_label is not None:
            return self.soft_label
        t = np.zeros(self.question.n_options)
        t[self.label] = 1.0
        return t

    def permuted(self, rng: np.random.Generator) -> TypedExample:
        """Shuffle option order. A no-op for ordinal questions, by design."""
        if self.question.ordinal:
            return self
        k = self.question.n_options
        perm = rng.permutation(k)
        crit = self.question.criteria
        if isinstance(crit, dict):
            keys = list(crit)
            new_crit = {keys[i]: crit[keys[i]] for i in perm}
        else:
            new_crit = [list(crit)[i] for i in perm]
        inv = np.empty(k, dtype=int)
        inv[perm] = np.arange(k)
        return TypedExample(
            self.state, Question(self.question.type, self.question.instructions,
                                 new_crit),
            None if self.label is None else int(inv[self.label]),
            None if self.soft_label is None else self.soft_label[perm],
            self.weight, self.meta)


# ----------------------------------------------------------------- shorthands
def choice(state, instructions, options, label=None, soft_label=None, **kw):
    """`options` is either {key: description} or a list of keys."""
    crit = options if isinstance(options, dict) else dict.fromkeys(options)
    if isinstance(label, str):
        label = list(crit).index(label)
    return TypedExample(state, Question("choice", instructions, crit),
                        label, soft_label, **kw)


def score(state, instructions, levels, label=None, soft_label=None, **kw):
    """`levels` is an ORDERED list, lowest first. Index is the score."""
    return TypedExample(state, Question("score", instructions, list(levels)),
                        label, soft_label, **kw)


def noul(state, instructions, label=None, soft_label=None, criteria=None, **kw):
    """Yes/no. label=1 is true. p[1] is the noul value."""
    if isinstance(label, bool):
        label = int(label)
    return TypedExample(state, Question("noul", instructions, criteria),
                        label, soft_label, **kw)


# -------------------------------------------------------------------- dataset
class TypedDataset:
    """Examples plus the tokenizer, with the token budget made visible."""

    def __init__(self, examples, tokenizer, max_len: int = 512,
                 head_max_len: int = 192, seed: int = 0):
        self.examples = list(examples)
        self.tok = tokenizer
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.rng = np.random.default_rng(seed)
        self._info: list[dict] = []

    def __len__(self):
        return len(self.examples)

    def build(self, ex: TypedExample):
        """One example -> (ids, marker positions, info). Raises if options do not fit."""
        ids, markers, info = build_sequence(
            self.tok, ex.state, ex.question.type, ex.question.instructions,
            ex.question.options, self.max_len, self.head_max_len, strict=True)
        self._info.append(info)
        return ids, markers, info

    def epoch(self, shuffle: bool = True, permute_options: bool = True):
        """Examples for one epoch, with option order re-randomised."""
        order = (self.rng.permutation(len(self.examples)) if shuffle
                 else np.arange(len(self.examples)))
        for i in order:
            ex = self.examples[int(i)]
            yield ex.permuted(self.rng) if permute_options else ex

    # ------------------------------------------------------------ diagnostics
    def budget_report(self, sample: int = 200) -> dict:
        """What the tokenizer actually kept. Run this BEFORE training.

        The single most common way a System One fine-tune quietly fails is that
        the state or the option text did not fit, so the model is trained on
        evidence it never saw. `state_truncated_frac` above zero means your
        examples are longer than the model can read.
        """
        info = self._info
        if not info:
            for ex in self.examples[:sample]:
                try:
                    self.build(ex)
                except ValueError:
                    continue
            info = self._info
        if not info:
            return {"error": "no example built successfully"}
        opt = [t for b in info for t in b["option_text_tokens"]]
        want = [t for b in info for t in b["option_text_tokens_untruncated"]]
        return {
            "examples_probed": len(info),
            "option_tokens_kept_mean": round(float(np.mean(opt)), 1),
            "option_tokens_wanted_mean": round(float(np.mean(want)), 1),
            "option_text_lost_frac": round(
                float(np.mean([a < b for a, b in zip(opt, want, strict=False)])), 3),
            "option_shrink_fired_frac": round(
                float(np.mean([b["option_shrink_fired"] for b in info])), 3),
            "instruction_tokens_mean": round(
                float(np.mean([b["instruction_tokens"] for b in info])), 1),
            "state_tokens_mean": round(
                float(np.mean([b["state_tokens"] for b in info])), 1),
            "state_truncated_frac": round(
                float(np.mean([b["state_truncated"] for b in info])), 3),
        }

    def label_report(self) -> dict:
        by_type: dict[str, int] = {}
        by_k: dict[int, int] = {}
        labels: dict[int, int] = {}
        soft = 0
        for ex in self.examples:
            by_type[ex.question.type] = by_type.get(ex.question.type, 0) + 1
            by_k[ex.question.n_options] = by_k.get(ex.question.n_options, 0) + 1
            if ex.label is not None:
                labels[ex.label] = labels.get(ex.label, 0) + 1
            soft += ex.soft_label is not None
        n = max(1, len(self.examples))
        maj = max(labels.values()) / n if labels else 0.0
        return {"examples": len(self.examples), "by_type": by_type,
                "by_option_count": dict(sorted(by_k.items())),
                "label_hist": dict(sorted(labels.items())),
                "soft_labelled_frac": round(soft / n, 3),
                "majority_class_baseline": round(maj, 4)}


def collate_typed(batch, tokenizer, dataset: TypedDataset):
    """A list of TypedExample -> the tensors the model's forward wants."""
    import torch

    built, keep = [], []
    for ex in batch:
        try:
            built.append(dataset.build(ex))
            keep.append(ex)
        except ValueError:
            continue                  # options did not fit; drop, never truncate
    if not built:
        return None

    n = len(built)
    L = max(len(b[0]) for b in built)
    K = max(len(b[1]) for b in built)
    pad = tokenizer.pad_token_id
    ids = torch.full((n, L), pad, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, K), dtype=torch.long)
    mmask = torch.zeros((n, K), dtype=torch.bool)
    target = torch.zeros((n, K), dtype=torch.float32)
    label = torch.full((n,), -1, dtype=torch.long)
    weight = torch.ones(n, dtype=torch.float32)
    qtype = torch.zeros(n, dtype=torch.long)
    ordinal = torch.zeros(n, dtype=torch.bool)

    for i, ((seq, mk, _), ex) in enumerate(zip(built, keep, strict=False)):
        ids[i, :len(seq)] = torch.tensor(seq)
        att[i, :len(seq)] = 1
        k = len(mk)
        mpos[i, :k] = torch.tensor(mk)
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(ex.target[:k], dtype=torch.float32)
        label[i] = -1 if ex.label is None else ex.label
        weight[i] = ex.weight
        qtype[i] = QTYPES[ex.question.type]
        ordinal[i] = ex.question.ordinal

    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos,
            "marker_mask": mmask, "target": target, "label": label,
            "weight": weight, "qtype": qtype, "ordinal": ordinal,
            "examples": keep}


def load_jsonl(path: str | Path) -> list[TypedExample]:
    """Load examples from JSONL.

    One object per line:
        {"state": ..., "type": "choice",
         "instructions": "...", "criteria": {...} | [...],
         "label": 2 | "billing", "soft_label": [...]}
    """
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        q = Question(d["type"], d["instructions"], d.get("criteria"))
        lab = d.get("label")
        if isinstance(lab, str):
            lab = list(d["criteria"]).index(lab)
        if isinstance(lab, bool):
            lab = int(lab)
        out.append(TypedExample(d["state"], q, lab,
                                d.get("soft_label"), d.get("weight", 1.0),
                                d.get("meta", {}) or {}))
    return out
