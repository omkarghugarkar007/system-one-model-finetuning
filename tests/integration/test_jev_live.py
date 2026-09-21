"""Live Jev calls. Skipped without a key; costs fractions of a cent when run.

    pytest tests/integration -m network -q
"""
import os
import pathlib

import numpy as np
import pytest

from frontierrank.models.protocols import RUBRIC_4LEVEL
from frontierrank.models.teachers import CachedTeacher, JevTeacher

pytestmark = [pytest.mark.network, pytest.mark.slow]


def _key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = pathlib.Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    return ""


@pytest.fixture(scope="module")
def teacher():
    key = _key()
    if not key:
        pytest.skip("no OPENROUTER_API_KEY")
    return JevTeacher(provider="openrouter", api_key=key, max_candidates=6)


QUERY = "how long is the battery covered by the device warranty?"
CANDIDATES = [
    "Device warranty applies for 24 months from purchase, except batteries "
    "and consumables, which are limited to 12 months.",
    "Batteries are covered for twelve months under the standard warranty.",
    "The warranty covers the chassis and display for 24 months.",
    "Our returns policy allows refunds within 30 days of delivery.",
    "The Paris office is open Monday to Friday from 9am.",
]


def test_grades_are_ordered_by_relevance(teacher):
    v = teacher.grade(QUERY, CANDIDATES, RUBRIC_4LEVEL).validate()
    g = v.expected_grade
    assert v.level_probs.shape == (5, 4)
    # the two battery passages must outrank the two unrelated ones
    assert min(g[0], g[1]) > max(g[3], g[4]), f"grades were {g.round(2).tolist()}"
    assert v.cost_usd > 0 and v.input_tokens > 0
    assert v.meta["cost_reported"], "OpenRouter should report usage.cost"


def test_distribution_carries_uncertainty_where_the_item_is_uncertain(teacher):
    """Jev returns exactly 1.0 on unambiguous items -- that is correct, not a bug.

    What distillation needs is that the *contested* items carry graded mass, so
    the student learns where its own uncertainty should live. Asserting that no
    row is one-hot would be asserting that the teacher is never sure, which is
    not a property worth having.

    Consequence for `training.losses.distill_kl`: a teacher probability of
    exactly 0 is real, so the clamp there is load-bearing, not defensive.
    """
    v = teacher.grade(QUERY, CANDIDATES, RUBRIC_4LEVEL)
    ent = -(v.level_probs * np.log(v.level_probs.clip(1e-9))).sum(axis=1)
    assert ent.max() > 0.05, "no item carried any uncertainty at all"
    assert (v.level_probs == 0.0).any(), "exact zeros are expected; the KL clamp matters"


def test_reversing_candidate_order_preserves_the_grades(teacher):
    """Order robustness, on the teacher this time.

    The plan cites 24.7% top-1 change for Jev's Choice under reordering. A
    per-candidate rubric Score should be far more stable than that, because
    each question is about one named candidate rather than a competition.
    """
    fwd = teacher.grade(QUERY, CANDIDATES).expected_grade
    rev = teacher.grade(QUERY, CANDIDATES[::-1]).expected_grade[::-1]
    assert np.abs(fwd - rev).max() < 1.0, (
        f"grades moved by {np.abs(fwd - rev).round(2).tolist()} under reordering")


def test_cache_makes_the_second_call_free(teacher, tmp_path):
    cached = CachedTeacher(teacher, tmp_path, backend_tag="jev-openrouter")
    a = cached.grade(QUERY, CANDIDATES[:3])
    b = cached.grade(QUERY, CANDIDATES[:3])
    assert np.allclose(a.level_probs, b.level_probs)
    assert cached.stats() == {"hits": 1, "misses": 1, "hit_rate": 0.5,
                              "spent_usd": round(a.cost_usd, 6)}
