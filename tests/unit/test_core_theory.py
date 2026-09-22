import numpy as np
import pytest

from systemone.eval import ndcg_at_k, recall_at_k
from systemone.reranking import (
    Anchor,
    AnchoredScorer,
    Controller,
    build_slates,
    compute_frontier,
    expected_regret,
    fit_slate,
    slates_per_query,
)


def test_shift_recovery_is_exact_without_noise():
    anchors = np.array([-2.0, 0.0, 2.0])
    logp = anchors - 7.3                      # a pure offset
    fit = fit_slate(logp, anchors, method="shift")
    assert fit.intercept == pytest.approx(7.3)
    assert fit.residual == pytest.approx(0.0, abs=1e-9)


def test_ridge_shrinks_slope_toward_one():
    anchors = np.linspace(-2, 2, 4)
    logp = anchors / 3.0 - 1.0                # slope 3 needed
    loose = fit_slate(logp, anchors, "ridge", lam=0.0).slope
    tight = fit_slate(logp, anchors, "ridge", lam=100.0).slope
    assert loose > tight > 1.0


def test_slate_budget_respected():
    plan = build_slates(100, 4, max_options=10, n_anchors=2)
    plan.validate()
    assert plan.n_passes == slates_per_query(100, 10, 2) == 13
    seen = sorted(i for s in plan.slates for i in s.candidate_ids)
    assert seen == list(range(100))


def test_anchors_cannot_eat_the_whole_slate():
    with pytest.raises(ValueError):
        build_slates(10, 4, max_options=2, n_anchors=2)


def test_frontier_excludes_the_obvious():
    # distinct, well-separated values -- exact ties would legitimately put every
    # tied candidate on the frontier, which is correct but not what we test here
    u = np.concatenate([np.linspace(30, 20, 4),      # clearly in
                        np.linspace(1.0, 0.9, 6),    # contested around rank 5
                        np.linspace(-20, -30, 90)])  # clearly out
    fr = compute_frontier(u, np.full(100, 0.05), k=5)
    assert fr.size <= 10                      # only the contested band
    assert 0 not in fr.members                # runaway leader excluded
    assert 99 not in fr.members               # hopeless candidate excluded
    assert set(range(4, 10)).issubset(set(fr.members.tolist()))


def test_regret_is_zero_when_certain():
    u = np.linspace(10, 0, 50)
    assert expected_regret(u, np.full(50, 1e-6), u, k=10) == pytest.approx(0.0, abs=1e-6)


def test_controller_stops_when_regret_is_tiny():
    u = np.linspace(10, 0, 50)
    s = np.full(50, 1e-6)
    fr = compute_frontier(u, s, k=10)
    assert Controller(lambda_cost=1000.0).decide(u, s, u, fr).action == "stop"


def test_ndcg_and_trec_threshold():
    rel = np.array([3, 0, 2, 1, 0])
    ideal = np.sort(rel)[::-1]
    assert ndcg_at_k(ideal, rel, 5) == pytest.approx(1.0)
    assert ndcg_at_k(rel, rel, 5) < 1.0           # that order is not ideal
    assert ndcg_at_k(np.sort(rel), rel, 5) < ndcg_at_k(ideal, rel, 5)
    # grade 1 is NOT relevant under trec_eval -l 2
    assert recall_at_k(rel, rel, 5, threshold=2) == pytest.approx(1.0)
    assert recall_at_k(np.array([1, 1, 0, 0, 0]), rel, 5, threshold=2) == 0.0


def test_scorer_end_to_end_is_globally_comparable():
    rng = np.random.default_rng(0)
    u = rng.normal(0, 1.5, 60)
    sigs = [f"c{i}" for i in range(60)]
    anchors = [Anchor(f"a{j}", v) for j, v in enumerate(np.linspace(-2.2, 2.2, 4))]
    table = dict(zip(sigs, u, strict=False)) | {a.text: a.utility for a in anchors}

    class M:
        def choice_logprobs(self, instr, options, state):
            z = np.array([table[o] for o in options]) + rng.normal(0, .2, len(options))
            return z - (z.max() + np.log(np.exp(z - z.max()).sum()))

    pool = AnchoredScorer(M(), anchors, n_anchors=4).score("the query", sigs)
    # calibrated utilities should live on the true scale, not a per-slate one
    assert np.corrcoef(pool.utility, u)[0, 1] > 0.9
    assert abs(pool.utility.mean() - u.mean()) < 0.6
