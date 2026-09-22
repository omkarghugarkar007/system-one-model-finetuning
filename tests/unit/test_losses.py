"""Loss invariants. The masking ones caught a real bug: see listnet_loss's docstring."""
import pytest

torch = pytest.importorskip("torch")

from systemone.train.losses import (CompositeRankingLoss, anchor_affine_loss,
                                          anchor_monotone_loss, coral_loss,
                                          listnet_loss, rps_loss)


def _slate():
    B, M = 2, 10
    rel = torch.tensor([[3., 2., 1., 0., 0., 0., -1, -1, -1, -1],
                        [2., 1., 0., 0., 1., 0., -1, -1, -1, -1]])
    anchor = torch.zeros(B, M, dtype=torch.bool)
    anchor[:, 6:] = True
    util = torch.full((B, M), float("nan"))
    util[:, 6:] = torch.tensor([-3., -1., 1., 3.])
    return rel, anchor, util, ~anchor


def test_listnet_does_not_explode_on_masked_positions():
    rel, anchor, _, mask = _slate()
    scores = torch.randn(2, 10)
    masked = listnet_loss(scores, rel.clamp_min(0), mask=mask)
    # the bug produced values in the hundreds purely from the -1e4 fill
    assert masked < 10.0, f"masking artefact is back: {masked}"
    assert torch.isfinite(masked)


def test_listnet_target_puts_no_mass_on_masked_positions():
    rel, anchor, _, mask = _slate()
    # a slate whose only difference is the score at a MASKED position
    a = torch.zeros(2, 10)
    b = a.clone()
    b[:, 7] = 50.0
    assert listnet_loss(a, rel.clamp_min(0), mask=mask) == pytest.approx(
        float(listnet_loss(b, rel.clamp_min(0), mask=mask)), abs=1e-5)


def test_coral_and_rps_ignore_ungraded_positions():
    rel, anchor, _, mask = _slate()
    ordi = torch.randn(2, 10, 3)
    lvl = torch.softmax(torch.randn(2, 10, 4), -1)
    for fn, args in ((coral_loss, (ordi, rel)), (rps_loss, (lvl, rel))):
        with_mask = fn(*args, mask=mask)
        without = fn(*args)
        assert with_mask != pytest.approx(float(without)), \
            "anchors must not be trained as grade-0 documents"


def test_anchor_affine_is_zero_for_a_perfect_ruler():
    u = torch.tensor([[-3., -1., 1., 3.]])
    m = torch.ones(1, 4, dtype=torch.bool)
    assert anchor_affine_loss(torch.tensor([[-1.5, -.5, .5, 1.5]]), u, m) \
        == pytest.approx(0.0, abs=1e-6)
    assert anchor_affine_loss(torch.tensor([[.5, -1.5, 1.5, -.5]]), u, m) > 0.5


def test_anchor_affine_is_invariant_to_slate_offset_and_scale():
    """The whole point: a pivot set is a ruler, and a ruler does not care
    where zero is. c_S shifts every logit in a slate equally, so the loss
    must not move."""
    u = torch.tensor([[-3., -1., 1., 3.]])
    m = torch.ones(1, 4, dtype=torch.bool)
    z = torch.tensor([[-1.3, -0.2, 0.7, 1.4]])
    base = anchor_affine_loss(z, u, m)
    assert anchor_affine_loss(z + 7.0, u, m) == pytest.approx(float(base), abs=1e-5)
    assert anchor_affine_loss(z * 3.0, u, m) == pytest.approx(float(base), abs=1e-4)


def test_anchor_monotone_penalises_only_inversions():
    u = torch.tensor([[-3., -1., 1., 3.]])
    m = torch.ones(1, 4, dtype=torch.bool)
    ok = torch.tensor([[-2., -1., 1., 2.]])
    assert anchor_monotone_loss(ok, u, m, margin=0.5) == pytest.approx(0.0)
    assert anchor_monotone_loss(ok.flip(-1), u, m, margin=0.5) > 0.0


def test_composite_backprops_and_stays_sane():
    rel, anchor, util, mask = _slate()
    sl = torch.randn(2, 10, requires_grad=True)
    total, parts = CompositeRankingLoss()(
        slate_logits=sl, rel=rel.clamp_min(0), mask=mask,
        ordinal_logits=torch.randn(2, 10, 3),
        level_probs=torch.softmax(torch.randn(2, 10, 4), -1),
        anchor_utility=util, is_anchor=anchor)
    total.backward()
    assert torch.isfinite(sl.grad).all()
    assert float(total) < 50.0, f"a term is dominating: {parts}"
    assert {"pair", "list", "ord", "cal", "anchor", "anchor_mono"} <= set(parts)
