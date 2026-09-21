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

    def for_query(self, query: str, ds=None, qid: str = "",
                  n: int = 4, rng=None) -> list[Anchor]:
        if self.strategy == "templated":
            return templated_anchors(query, self.templates[:n])
        if self.strategy == "judged":
            anchors = judged_anchors(ds, qid, n, rng=rng)
            self.held_out.update(a.doc_id for a in anchors)
            return anchors
        raise ValueError(f"unknown anchor strategy {self.strategy!r}")

    def utilities(self, n: int = 4) -> np.ndarray:
        return np.array([t.utility for t in self.templates[:n]], dtype=float)

    def spread(self, n: int = 4) -> float:
        """Utility range the pivots span. The regression's leverage."""
        u = self.utilities(n)
        return float(np.ptp(u)) if u.size else 0.0
