"""The cascade and the regret it is steered by. Both had bugs; both are pinned now."""
import numpy as np
import pytest

from systemone.reranking import Anchor, AnchoredScorer, Controller
from systemone.reranking.cascade import Cascade, fuse_teacher
from systemone.reranking.frontier import expected_regret
from systemone.teachers import SimulatedTeacher


# ------------------------------------------------------------------- regret
def test_regret_is_zero_only_when_the_order_is_certain():
    u = np.linspace(10, 0, 50)
    assert expected_regret(u, np.full(50, 1e-9), k=10) == pytest.approx(0.0, abs=1e-9)
    assert expected_regret(u, np.full(50, 1.0), k=10) > 0.0


def test_regret_is_not_identically_zero():
    """The bug this guards: scoring both rankings with the point estimate makes
    the shipped order optimal by construction, so regret is always 0 and the
    controller can never justify buying anything."""
    u = np.linspace(10, 0, 50)
    assert expected_regret(u, np.full(50, 0.5), k=10) > 1e-4


def test_regret_increases_with_uncertainty():
    u = np.linspace(10, 0, 50)
    vals = [expected_regret(u, np.full(50, s), k=10, rng=np.random.default_rng(0))
            for s in (0.1, 0.5, 1.0, 2.0, 4.0)]
    assert vals == sorted(vals)


def test_regret_is_bounded():
    u = np.linspace(10, 0, 50)
    assert 0.0 <= expected_regret(u, np.full(50, 50.0), k=10) <= 1.0


# --------------------------------------------------------------------- fuse
def test_fusion_touches_only_the_frontier():
    u = np.arange(10, dtype=float)
    s = np.full(10, 0.5)
    targets = np.array([3, 4])
    u2, s2 = fuse_teacher(u, s, targets, np.array([3.0, 0.0]))
    untouched = np.setdiff1d(np.arange(10), targets)
    assert np.allclose(u[untouched], u2[untouched])
    assert np.allclose(s[untouched], s2[untouched])
    assert not np.allclose(u[targets], u2[targets])


def test_fusion_narrows_sigma_but_does_not_claim_certainty():
    u = np.zeros(4)
    s = np.full(4, 0.9)
    _, s2 = fuse_teacher(u, s, np.array([1]), np.array([2.0]))
    assert s2[1] < s[1] and s2[1] > 0.0


def test_teacher_weight_zero_is_a_no_op_on_utility():
    u = np.arange(5, dtype=float)
    u2, _ = fuse_teacher(u, np.full(5, .5), np.array([2]), np.array([3.0]),
                         teacher_weight=0.0)
    assert np.allclose(u, u2)


# ------------------------------------------------------------------ cascade
def _fixture(seed=0, n=60):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1.5, n)
    sigs = [f"c{i}" for i in range(n)]
    anchors = [Anchor(f"a{j}", v) for j, v in enumerate(np.linspace(-2.2, 2.2, 4))]
    table = dict(zip(sigs, truth, strict=False)) | {a.text: a.utility for a in anchors}

    class M:
        def choice_logprobs(self, instr, options, state):
            z = np.array([table[o] for o in options]) + rng.normal(0, .3, len(options))
            return z - (z.max() + np.log(np.exp(z - z.max()).sum()))

    grades = np.clip((truth + 3) / 2, 0, 3)
    return (AnchoredScorer(M(), anchors, n_anchors=4), sigs,
            SimulatedTeacher({"q": grades}, skill=0.95))


def test_expensive_compute_stops_cheap_compute_escalates():
    scorer, sigs, teacher = _fixture()
    stop = Cascade(scorer, teacher, Controller(lambda_cost=1e5), k=10).run(
        "q", sigs, rng=np.random.default_rng(1))
    go = Cascade(scorer, teacher, Controller(lambda_cost=1.0), k=10).run(
        "q", sigs, rng=np.random.default_rng(1))
    assert not stop.escalated
    assert go.escalated


def test_escalation_actually_reduces_regret():
    scorer, sigs, teacher = _fixture()
    r = Cascade(scorer, teacher, Controller(lambda_cost=1.0), k=10).run(
        "q", sigs, rng=np.random.default_rng(1))
    assert r.regret_after < r.regret_before


def test_regret_does_not_move_when_nothing_is_bought():
    """Paired Monte Carlo draws. With independent draws this failed: regret
    appeared to RISE after an action that could only reduce it."""
    scorer, sigs, teacher = _fixture()
    r = Cascade(scorer, teacher, Controller(lambda_cost=1e9), k=10).run(
        "q", sigs, rng=np.random.default_rng(1))
    assert r.regret_after == pytest.approx(r.regret_before)


def test_escalation_yields_a_training_example():
    scorer, sigs, teacher = _fixture()
    r = Cascade(scorer, teacher, Controller(lambda_cost=1.0), k=10).run(
        "q", sigs, rng=np.random.default_rng(1))
    assert len(r.training_examples) == 1
    ex = r.training_examples[0]
    assert ex["teacher_level_probs"].shape[0] == r.teacher_targets.size
    assert ex["student_utility"].size == r.teacher_targets.size


def test_cascade_runs_without_a_teacher():
    scorer, sigs, _ = _fixture()
    r = Cascade(scorer, None, Controller(lambda_cost=1.0), k=10).run(
        "q", sigs, rng=np.random.default_rng(1))
    assert not r.escalated and r.cost_usd == 0.0


def test_break_even_lambda_locates_the_escalation_boundary():
    c = Controller()
    lam = c.break_even_lambda(0.03, action="teacher_rubric")
    below = Controller(lambda_cost=lam * 0.5)
    above = Controller(lambda_cost=lam * 2.0)
    scorer, sigs, teacher = _fixture()
    assert Cascade(scorer, teacher, below, k=10).run(
        "q", sigs, rng=np.random.default_rng(1)).escalated
    assert not Cascade(scorer, teacher, above, k=10).run(
        "q", sigs, rng=np.random.default_rng(1)).escalated
