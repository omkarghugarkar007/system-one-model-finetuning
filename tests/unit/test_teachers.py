"""Teacher-layer invariants. No network: Jev is exercised in tests/integration."""
import json

import numpy as np
import pytest

from systemone.teachers import (
    CachedTeacher,
    InSessionTeacher,
    SimulatedTeacher,
    TeacherPending,
    request_digest,
)
from systemone.teachers.protocols import RUBRIC_4LEVEL, TeacherVerdict, expected_grade


def test_verdict_rejects_rows_that_do_not_sum_to_one():
    bad = TeacherVerdict(np.arange(2), np.array([[0.5, 0.2], [0.5, 0.5]]))
    with pytest.raises(ValueError, match="sum to 1"):
        bad.validate()


def test_expected_grade_is_the_distribution_mean():
    p = np.array([[0.01, 0.04, 0.21, 0.74]])
    assert expected_grade(p)[0] == pytest.approx(0.0 + 0.04 + 0.42 + 2.22)


def test_digest_is_order_sensitive_and_stable():
    a = request_digest("q", ["x", "y"], RUBRIC_4LEVEL, "t")
    assert a == request_digest("q", ["x", "y"], RUBRIC_4LEVEL, "t")
    assert a != request_digest("q", ["y", "x"], RUBRIC_4LEVEL, "t")
    assert a != request_digest("q", ["x", "y"], RUBRIC_4LEVEL, "other")


def test_simulated_teacher_peaks_on_the_true_grade():
    t = SimulatedTeacher({"q": np.array([0, 1, 2, 3])}, skill=1.0)
    v = t.grade("q", ["a", "b", "c", "d"]).validate()
    assert v.level_probs.argmax(axis=1).tolist() == [0, 1, 2, 3]


def test_low_skill_teacher_is_closer_to_uniform():
    good = SimulatedTeacher({"q": np.array([3.0])}, skill=1.0).grade("q", ["a"])
    poor = SimulatedTeacher({"q": np.array([3.0])}, skill=0.0).grade("q", ["a"])
    assert good.level_probs.max() > poor.level_probs.max()


def test_cache_replays_without_paying_twice(tmp_path):
    inner = SimulatedTeacher({"q": np.array([2, 1])}, skill=0.9)
    cached = CachedTeacher(inner, tmp_path, backend_tag="sim")
    first = cached.grade("q", ["a", "b"])
    second = cached.grade("q", ["a", "b"])
    assert inner.calls == 1, "second call must not reach the inner teacher"
    assert np.allclose(first.level_probs, second.level_probs)
    assert cached.stats()["hits"] == 1 and cached.stats()["misses"] == 1


def test_in_session_teacher_round_trip(tmp_path):
    t = InSessionTeacher(tmp_path)
    with pytest.raises(TeacherPending) as e:
        t.grade("what is the warranty?", ["12 months", "unrelated"])
    req = json.loads(e.value.path.read_text())
    assert list(req["candidates"]) == ["c0", "c1"]
    assert len(t.outstanding()) == 1

    digest = req["digest"]
    (tmp_path / f"{digest}.response.json").write_text(
        json.dumps({"grades": [3, 0], "grader": "test"}))
    v = t.grade("what is the warranty?", ["12 months", "unrelated"]).validate()
    assert v.level_probs.argmax(axis=1).tolist() == [3, 0]
    assert t.outstanding() == []


def test_hand_grades_are_not_distilled_as_certainties(tmp_path):
    t = InSessionTeacher(tmp_path, confidence=0.9)
    p = t._probs_from_grades([2], 4)
    assert p[0, 2] == pytest.approx(0.9)
    # a human writing "2" should leave more mass on 1 and 3 than on 0
    assert p[0, 1] > p[0, 0] and p[0, 3] > p[0, 0]
    assert p.sum() == pytest.approx(1.0)
