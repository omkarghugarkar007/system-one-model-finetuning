"""BEIR corpora in the standard layout: corpus.jsonl, queries.jsonl, qrels/*.tsv.

Which datasets, and why these first. Phase 0 asks whether the per-slate offset
c_S behaves as modelled on real text. That needs *graded* judgements (a binary
qrel cannot show a utility scale) and it needs to run in minutes, not hours.

    nfcorpus    3.6k docs, 323 test queries, grades 0-2. Minutes end to end.
    scifact     5.2k docs, 300 test queries, binary. A control: if the anchor
                mechanism needs graded labels, it should visibly struggle here.
    trec-covid  171k docs, 50 test queries, grades 0-2, deeply judged.

TREC DL19/20 land later via `data.trec_dl` -- they are the plan's headline
corpus and the only clean 4-level qrels, but they need the 8.8M-passage MS
MARCO collection, which is not where a go/no-go gate should start.

One convention enforced here and nowhere else: `min_grade_relevant`. BEIR's own
evaluation treats any positive grade as relevant, but TREC DL's grade 1 is
"related but does not answer", and binary metrics there must use
`trec_eval -l 2`. `Dataset.binary_relevant()` makes the threshold explicit at
the call site so it cannot be forgotten silently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Dataset", "Doc", "load_beir", "BEIR_URL", "AVAILABLE"]

BEIR_URL = ("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/"
            "datasets/{name}.zip")

# name -> (max grade in qrels, note)
AVAILABLE = {
    "nfcorpus": (2, "3.6k docs, 323 test queries, graded 0-2"),
    "scifact": (1, "5.2k docs, 300 test queries, binary"),
    "trec-covid": (2, "171k docs, 50 test queries, graded 0-2, deeply judged"),
    "fiqa": (1, "57k docs, binary"),
    "scidocs": (1, "25k docs, binary"),
}


@dataclass
class Doc:
    doc_id: str
    title: str
    text: str

    @property
    def full(self) -> str:
        return f"{self.title}\n{self.text}".strip() if self.title else self.text


@dataclass
class Dataset:
    name: str
    split: str
    docs: dict[str, Doc]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]          # qid -> {doc_id: grade}
    max_grade: int = 2
    meta: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        judged = sum(len(v) for v in self.qrels.values())
        return (f"Dataset({self.name}/{self.split}: {len(self.docs):,} docs, "
                f"{len(self.queries):,} queries, {judged:,} judgements, "
                f"grades 0-{self.max_grade})")

    @property
    def query_ids(self) -> list[str]:
        """Queries that actually carry judgements, in a stable order."""
        return sorted(q for q in self.queries if self.qrels.get(q))

    def grade(self, qid: str, doc_id: str) -> int:
        """Unjudged is 0. That is the TREC convention and it is not neutral --
        it makes recall optimistic on shallowly judged pools. `judged_frac`
        reports how much of a ranking rests on that assumption."""
        return self.qrels.get(qid, {}).get(doc_id, 0)

    def grades_for(self, qid: str, doc_ids) -> list[int]:
        return [self.grade(qid, d) for d in doc_ids]

    def judged_frac(self, qid: str, doc_ids) -> float:
        known = self.qrels.get(qid, {})
        ids = list(doc_ids)
        return sum(d in known for d in ids) / max(1, len(ids))

    def binary_relevant(self, qid: str, doc_id: str, min_grade: int = 1) -> bool:
        """Explicit threshold. On TREC DL pass min_grade=2: grade 1 is NOT relevant."""
        return self.grade(qid, doc_id) >= min_grade

    def grade_histogram(self) -> dict[int, int]:
        h: dict[int, int] = {}
        for rels in self.qrels.values():
            for g in rels.values():
                h[g] = h.get(g, 0) + 1
        return dict(sorted(h.items()))


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_beir(name: str, root: str | Path = "data/raw", split: str = "test",
              max_docs: int | None = None) -> Dataset:
    """Load a BEIR dataset already unpacked under `root/name`.

    `max_docs` subsamples the corpus but always keeps every judged document, so
    a smoke run stays a valid ranking problem rather than becoming an easy one.
    """
    d = Path(root) / name
    if not d.exists():
        raise FileNotFoundError(
            f"{d} not found. Download it:\n"
            f"  curl -O {BEIR_URL.format(name=name)} && unzip {name}.zip -d {root}")

    qrels_path = d / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        have = [p.stem for p in (d / "qrels").glob("*.tsv")]
        raise FileNotFoundError(f"no {split} qrels in {d}; have {have}")

    qrels: dict[str, dict[str, int]] = {}
    n_negative = [0]
    with qrels_path.open() as f:
        header = f.readline()
        if not header.lower().startswith("query"):
            f.seek(0)                      # some releases ship without a header
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            # TREC collections use -1 for "junk / not judged"; the standard
            # treatment is non-relevant, and silently keeping it negative
            # would corrupt every gain computation downstream
            if score < 0:
                n_negative[0] += 1
                score = 0
            qrels.setdefault(qid, {})[did] = score

    queries = {q["_id"]: q["text"] for q in _read_jsonl(d / "queries.jsonl")}
    queries = {k: v for k, v in queries.items() if k in qrels}

    judged = {did for rels in qrels.values() for did in rels}
    docs: dict[str, Doc] = {}
    for rec in _read_jsonl(d / "corpus.jsonl"):
        did = rec["_id"]
        if max_docs is not None and len(docs) >= max_docs and did not in judged:
            continue
        docs[did] = Doc(did, rec.get("title", "") or "", rec.get("text", "") or "")

    missing = judged - set(docs)
    return Dataset(name, split, docs, queries, qrels,
                   max_grade=AVAILABLE.get(name, (2, ""))[0],
                   meta={"root": str(d), "judged_docs": len(judged),
                         "judged_docs_missing_from_corpus": len(missing),
                         "negative_grades_clamped_to_zero": n_negative[0]})
