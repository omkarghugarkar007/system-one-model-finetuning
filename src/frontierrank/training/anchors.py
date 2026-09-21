"""Where pivot documents come from. A gap `plan.md` leaves open.

Part VII says the pivots are "fixed across the whole corpus, held out of
evaluation, and chosen to span the utility range". Those requirements conflict:
a pivot's *utility* is its relevance, relevance is a function of the query, and
a document fixed across the corpus cannot have a fixed relevance to every
query. A passage about hedgehogs is grade 0 for most queries and grade 3 for
one of them.

Three resolutions, and they are genuinely different experiments.

TEMPLATED (default). The *construction* is fixed corpus-wide, the text is
    rendered per query. A direct-answer template is grade 3 for any query by
    construction; a fixed off-topic passage is grade 0 for any query. This is
    the only option that delivers what the theory actually needs -- pivots whose
    utility is known *a priori*, with designed separation -- and it costs no
    judgements.

    It also fixes the problem F8 ran into. Grade-based pivots on trec-covid give
    the anchor regression three distinct x-values; templated pivots can be
    placed at any utilities you like, including a wide, evenly spaced range.

JUDGED. Sample the query's own judged documents as pivots and hold them out of
    that query's evaluation. Faithful to real relevance, but the pivots differ
    per query, so they do not establish one corpus-wide scale -- which is the
    entire purpose. Useful as a control.

TEACHER. Pivots graded by Jev, whose Score returns a *fractional* expected
    grade. Continuous utilities, real text, real relevance. Costs one teacher
    call per pivot set and is what production should use.

The templated anchors are an assumption with a cheap test, and the test is in
`scripts/run_anchor_probe.py`: if the model does not actually rank a
direct-answer template above an off-topic one, the templates are not pivots and
the whole approach is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.scoring import Anchor

__all__ = ["AnchorTemplate", "DEFAULT_TEMPLATES", "templated_anchors",
           "judged_anchors", "AnchorPool"]


@dataclass
class AnchorTemplate:
    """A pivot whose grade is known by construction.

    `render(query)` must produce text whose relevance to *that* query is fixed
    at `grade`, whatever the query is. `utility` is where the pivot sits on the
    global scale; it need not equal the grade, and spreading utilities wider
    than the grade scale is exactly what gives the anchor regression leverage.
    """
    name: str
    grade: int
    utility: float
    template: str          # "{query}" is substituted; no other fields

    def render(self, query: str) -> str:
        return self.template.format(query=query.strip().rstrip("?"))


# A fixed off-topic passage. Deliberately mundane, concrete and specific: a
# vague filler string risks reading as "generically about anything", which is
# the one thing a grade-0 pivot must not do.
_OFF_TOPIC = (
    "The municipal tram depot on River Street was rebuilt in 1974 and now "
    "houses a maintenance yard for the city's rolling stock, with two "
    "inspection pits and a small staff canteen open on weekday mornings.")

DEFAULT_TEMPLATES = [
    AnchorTemplate("off_topic", 0, -3.0, _OFF_TOPIC),
    AnchorTemplate(
        "topical_nonanswer", 1, -1.0,
        "This article surveys the general background and history of {query}. "
        "It describes why the subject has attracted attention and lists "
        "related areas of study, but it does not state any specific finding "
        "or give a direct answer."),
    AnchorTemplate(
        "partial", 2, 1.0,
        "Regarding {query}: the available evidence points in one direction, "
        "though the passage covers several other topics at greater length and "
        "the relevant statement appears only briefly, without detail."),
    AnchorTemplate(
        "direct_answer", 3, 3.0,
        "{query}. This passage answers that question directly and completely, "
        "stating the specific finding, the conditions under which it holds, "
        "and the evidence supporting it."),
]


def templated_anchors(query: str,
                      templates=DEFAULT_TEMPLATES) -> list[Anchor]:
    """Render the fixed template set for one query."""
    return [Anchor(text=t.render(query), utility=t.utility, grade=t.grade,
                   doc_id=f"__anchor__{t.name}") for t in templates]


def judged_anchors(ds, qid: str, n: int = 4,
                   utility_of_grade=None,
                   rng: np.random.Generator | None = None) -> list[Anchor]:
    """Pivots drawn from the query's own judged documents. The control.

    Returns fewer than `n` when the query lacks judged documents spanning the
    grade range, which is itself informative: a corpus that cannot supply
    spread pivots cannot test anchoring either.
    """
    rng = rng or np.random.default_rng(0)
    utility_of_grade = utility_of_grade or (lambda g: float(g) * 2.0 - 3.0)
    by_grade: dict[int, list[str]] = {}
    for did, g in ds.qrels.get(qid, {}).items():
        if did in ds.docs:
            by_grade.setdefault(int(g), []).append(did)
    out = []
    for g in sorted(by_grade):
        pick = by_grade[g][int(rng.integers(0, len(by_grade[g])))]
        out.append(Anchor(text=ds.docs[pick].full, utility=utility_of_grade(g),
                          grade=g, doc_id=pick))
    return out[:n]


@dataclass
class AnchorPool:
    """The anchor set a scorer uses, plus which documents it consumed.

    `held_out` must be excluded from every qrel and every candidate pool.
    Scoring a document that is also a pivot in the same slate would leak the
    label, and the leak would look like the mechanism working.
    """
    strategy: str = "templated"
    templates: list = field(default_factory=lambda: list(DEFAULT_TEMPLATES))
    held_out: set = field(default_factory=set)
    # only for strategy="teacher"
    teacher: object = None
    index: object = None
    signature_builder: object = None
    _cache: dict = field(default_factory=dict)

    def for_query(self, query: str, ds=None, qid: str = "",
                  n: int = 4, rng=None) -> list[Anchor]:
        if self.strategy == "templated":
            return templated_anchors(query, self.templates[:n])
        if self.strategy == "judged":
            anchors = judged_anchors(ds, qid, n, rng=rng)
            self.held_out.update(a.doc_id for a in anchors)
            return anchors
        if self.strategy == "teacher":
            # one teacher call per query, cached: the pivots for a query do not
            # change between slates or between epochs
            if qid in self._cache:
                return self._cache[qid]
            if self.teacher is None or self.index is None:
                raise ValueError("teacher strategy needs `teacher` and `index`")
            anchors = teacher_anchors(ds, self.index, self.teacher, qid, n,
                                      signature_builder=self.signature_builder)
            self.held_out.update(a.doc_id for a in anchors)
            self._cache[qid] = anchors
            return anchors
        raise ValueError(f"unknown anchor strategy {self.strategy!r}")

    def utilities(self, n: int = 4) -> np.ndarray:
        return np.array([t.utility for t in self.templates[:n]], dtype=float)

    def spread(self, n: int = 4) -> float:
        """Utility range the pivots span. The regression's leverage."""
        u = self.utilities(n)
        return float(np.ptp(u)) if u.size else 0.0


# ==========================================================================
# Teacher-graded pivots: real text, continuous utilities
# ==========================================================================
#
# `run_anchor_probe.py` on the base checkpoint found the templated pivots only
# weakly ordered (rank correlation 0.33, exact order 2.5%), and the diagnosis
# is not purely "the model is bad". The grade-3 template *claims* to answer the
# query without containing an answer, and a relevance model is right not to
# reward a self-referential claim. Synthetic text cannot carry real relevance
# at the top of the scale: a passage that answers a question has to contain the
# answer, and only the corpus has that.
#
# So the top of the pivot scale must be real text, which means its utility has
# to be measured rather than declared. Jev's Score returns a FRACTIONAL
# expected grade (2.85, not 3), which is strictly better than a qrel label for
# this purpose:
#
#   * continuous, so pivots can be placed anywhere on the scale rather than
#     landing on three integers -- the leverage problem F8 identified;
#   * on one rubric, so the scale is fixed corpus-wide even though the pivot
#     *documents* differ per query, which is what "one global scale" actually
#     requires;
#   * already paid for, since the escalation path calls the teacher anyway.
#
# One call per query, cached, at roughly $0.0002. For 250 training queries that
# is about five cents.

def teacher_anchors(ds, index, teacher, qid: str, n: int = 4,
                    pool_depth: int = 30, signature_builder=None,
                    tokens_per_candidate: int = 200,
                    rubric=None) -> list[Anchor]:
    """Pivots drawn from the corpus and graded by the teacher.

    Picks `n` documents whose teacher grades are as evenly spread as possible,
    because the affine fit's leverage is the spread of its pivots and clustered
    pivots give an ill-conditioned slope no regularisation repairs.

    Utilities are mapped from the 0-3 rubric onto roughly [-3, +3] so they sit
    in the same range as the templated set, keeping the two strategies
    comparable in an ablation.
    """
    from ..models.protocols import RUBRIC_4LEVEL

    query = ds.queries[qid]
    ranked, _ = index.search(query, pool_depth)
    if not ranked:
        return []
    if signature_builder is not None:
        texts = [signature_builder.build(query, ds.docs[d].title,
                                         ds.docs[d].text, tokens_per_candidate)
                 for d in ranked]
    else:
        texts = [ds.docs[d].full[: tokens_per_candidate * 4] for d in ranked]

    verdict = teacher.grade(query, texts, rubric or RUBRIC_4LEVEL)
    grades = np.asarray(verdict.expected_grade, dtype=float)

    # evenly spread over the observed grade range
    targets = np.linspace(grades.min(), grades.max(), n)
    picks, used = [], set()
    for t in targets:
        for j in np.argsort(np.abs(grades - t)):
            if int(j) not in used:
                used.add(int(j))
                picks.append(int(j))
                break

    return [Anchor(text=texts[j], utility=float(grades[j]) * 2.0 - 3.0,
                   grade=int(round(grades[j])), doc_id=ranked[j])
            for j in sorted(picks, key=lambda j: grades[j])]
