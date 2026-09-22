"""The generic objective and the reporting that keeps it honest."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from systemone import choice, score  # noqa: E402
from systemone.eval import report, slice_report  # noqa: E402
from systemone.train.losses import TypedDecisionLoss, typed_ce, typed_rps  # noqa: E402


def _batch():
    B, K = 4, 6
    logits = torch.randn(B, K)
    mask = torch.zeros(B, K, dtype=torch.bool)
    mask[:, :4] = True
    target = torch.zeros(B, K)
    target[torch.arange(B), torch.tensor([0, 1, 2, 3])] = 1.0
    ordinal = torch.tensor([False, False, True, True])
    return logits, target, mask, ordinal


def test_padded_slots_cannot_affect_the_loss():
    """Padded markers carry -1e4 logits. Any target mass left on them produces
    a loss in the hundreds that is pure padding artefact."""
    logits, target, mask, _ = _batch()
    a = typed_ce(logits, target, mask)
    moved = logits.clone()
    moved[:, 4:] = 50.0
    assert typed_ce(moved, target, mask) == pytest.approx(float(a), abs=1e-5)
    assert float(a) < 10.0


def test_rps_only_charges_ordinal_rows():
    logits, target, mask, ordinal = _batch()
    none_ordinal = torch.zeros_like(ordinal)
    assert typed_rps(logits, target, mask, none_ordinal) == pytest.approx(0.0)
    assert typed_rps(logits, target, mask, ordinal) > 0.0


def test_rps_punishes_being_off_by_three_more_than_off_by_one():
    """The whole reason to use an ordinal rule rather than cross-entropy."""
    mask = torch.ones(1, 4, dtype=torch.bool)
    ordinal = torch.tensor([True])
    target = torch.tensor([[0.0, 0.0, 0.0, 1.0]])      # truth is level 3
    near = torch.tensor([[-4.0, -4.0, 4.0, 0.0]])      # predicts 2
    far = torch.tensor([[4.0, -4.0, -4.0, 0.0]])       # predicts 0
    assert typed_rps(near, target, mask, ordinal) < typed_rps(far, target, mask, ordinal)


def test_composite_backprops():
    logits, target, mask, ordinal = _batch()
    logits = logits.requires_grad_(True)
    total, parts = TypedDecisionLoss()(logits=logits, target=target, mask=mask,
                                       ordinal=ordinal)
    total.backward()
    assert torch.isfinite(logits.grad).all()
    assert {"ce", "rps", "total"} <= set(parts)


# --------------------------------------------------------------------- eval
def _preds(n=200, acc_small=0.9, acc_big=0.4, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = 3 if i % 2 else 9
        ex = choice("s", "q", {f"o{j}": None for j in range(k)},
                    label=int(rng.integers(k)))
        hit = rng.random() < (acc_small if k == 3 else acc_big)
        p = np.full(k, 0.02)
        p[ex.label if hit else int(rng.integers(k))] = 1.0
        out.append({"example": ex, "probs": p / p.sum()})
    return out


def test_slice_report_exposes_a_bad_bucket_behind_a_good_aggregate():
    r = slice_report(_preds(), by="options")
    agg = r["__aggregate__"]
    assert agg["worst_slice_ece"] > agg["ece"]
    assert agg["hiding_ratio"] > 1.0


def test_report_compares_against_majority_not_chance():
    text = report(_preds())
    assert "majority-class baseline" in text
    assert "AGGREGATE" in text


def test_aurc_rewards_confidence_that_orders_errors():
    from systemone.eval import aurc
    conf = np.linspace(1.0, 0.0, 100)
    ordered = (np.arange(100) < 50).astype(float)      # confident ones are right
    shuffled = np.random.default_rng(0).permutation(ordered)
    assert aurc(conf, ordered) < aurc(conf, shuffled)


def test_ordinal_examples_are_reported_as_their_own_slice():
    preds = _preds(40) + [{"example": score("s", "r", ["a", "b", "c"], label=1),
                           "probs": np.array([0.1, 0.8, 0.1])} for _ in range(30)]
    r = slice_report(preds, by="type")
    assert {"choice", "score"} <= set(r)
