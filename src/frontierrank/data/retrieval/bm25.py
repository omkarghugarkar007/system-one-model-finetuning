"""BM25 as a sparse matrix. Own implementation, on purpose.

The first stage moves the number more than the reranker does -- RankZephyr
swings 10.7 nDCG@10 points on DL20 purely by changing what it reranks -- so the
first stage has to be pinned, inspectable and identical across every row of the
evaluation grid. A vendored BM25 whose k1, b and tokenizer are visible in this
file is worth more here than a faster black box.

Doc-side weights do not depend on the query, so the whole BM25 numerator is
precomputed once into a CSR matrix and a query becomes a column gather plus a
sum. That also hands us `term_positions`, which the evidence slicer needs to
find the best lexical window without a second pass over the text.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

__all__ = ["BM25Index", "tokenize", "STOPWORDS"]

_TOKEN = re.compile(r"[a-z0-9]+")

# a small, conventional list. Deliberately not aggressive: BM25's IDF already
# discounts frequent terms, and stripping too much hurts phrase-ish queries.
STOPWORDS = frozenset("""
a an and are as at be by for from has have he in is it its of on or that the
to was were will with this these those i you we they them his her our your
""".split())


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return [t for t in toks if t not in STOPWORDS] if drop_stopwords else toks


@dataclass
class BM25Index:
    """Okapi BM25 over a fixed corpus."""
    doc_ids: list
    k1: float = 0.9
    b: float = 0.4          # BEIR/Anserini defaults, not the textbook 1.2/0.75

    def __post_init__(self):
        self.vocab: dict[str, int] = {}
        self.matrix = None          # CSR: docs x terms, BM25 doc-side weights
        self.idf = None
        self.doc_len = None
        self._row_of = {d: i for i, d in enumerate(self.doc_ids)}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, doc_ids, texts, k1: float = 0.9, b: float = 0.4,
              drop_stopwords: bool = True) -> "BM25Index":
        from scipy import sparse

        self = cls(list(doc_ids), k1, b)
        self.drop_stopwords = drop_stopwords
        vocab: dict[str, int] = {}
        rows, cols, freqs = [], [], []
        lengths = np.zeros(len(self.doc_ids), dtype=np.float64)

        for i, text in enumerate(texts):
            counts = Counter(tokenize(text, drop_stopwords))
            lengths[i] = sum(counts.values())
            for term, f in counts.items():
                j = vocab.setdefault(term, len(vocab))
                rows.append(i)
                cols.append(j)
                freqs.append(f)

        n_docs, n_terms = len(self.doc_ids), len(vocab)
        tf = sparse.csr_matrix((np.asarray(freqs, dtype=np.float64), (rows, cols)),
                               shape=(n_docs, n_terms))
        df = np.asarray((tf > 0).sum(axis=0)).ravel()
        # Lucene/Anserini form: always positive, unlike the classic variant
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        avgdl = lengths.mean() if n_docs else 1.0
        denom_doc = self.k1 * (1.0 - self.b + self.b * lengths / max(avgdl, 1e-9))
        w = tf.tocoo()
        num = w.data * (self.k1 + 1.0)
        den = w.data + denom_doc[w.row]
        weighted = sparse.csr_matrix((num / den, (w.row, w.col)),
                                     shape=(n_docs, n_terms))

        self.vocab, self.matrix, self.idf = vocab, weighted.tocsc(), idf
        self.doc_len, self.avgdl = lengths, float(avgdl)
        return self

    # ----------------------------------------------------------------- search
    def score(self, query: str) -> np.ndarray:
        terms = tokenize(query, getattr(self, "drop_stopwords", True))
        scores = np.zeros(len(self.doc_ids), dtype=np.float64)
        for term, n in Counter(terms).items():
            j = self.vocab.get(term)
            if j is None:
                continue
            col = self.matrix[:, j]
            # repeated query terms count, matching Lucene's query-side handling
            scores[col.indices] += col.data * self.idf[j] * n
        return scores

    def search(self, query: str, k: int = 100) -> tuple[list, np.ndarray]:
        s = self.score(query)
        k = min(k, s.size)
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [self.doc_ids[i] for i in top], s[top]

    def query_terms_present(self, query: str) -> set[str]:
        return {t for t in tokenize(query, getattr(self, "drop_stopwords", True))
                if t in self.vocab}

    def __repr__(self) -> str:
        return (f"BM25Index({len(self.doc_ids):,} docs, {len(self.vocab):,} terms, "
                f"k1={self.k1}, b={self.b}, avgdl={getattr(self, 'avgdl', 0):.1f})")
