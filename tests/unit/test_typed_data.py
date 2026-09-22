"""The typed-decision API: the contract everything else is built on."""
import json

import numpy as np
import pytest

from systemone import Question, TypedExample, choice, noul, score
from systemone.data import load_jsonl


def test_choice_accepts_a_label_by_name():
    ex = choice("s", "which?", {"a": "first", "b": "second"}, label="b")
    assert ex.label == 1
    assert np.allclose(ex.target, [0.0, 1.0])


def test_score_levels_are_ordered_and_index_is_the_score():
    ex = score("s", "rate", ["low", "mid", "high"], label=2)
    assert ex.question.ordinal
    assert ex.question.options == ["level 0: low", "level 1: mid", "level 2: high"]


def test_noul_is_two_options_with_true_second():
    ex = noul("s", "is it?", label=True)
    assert ex.question.n_options == 2
    assert np.allclose(ex.target, [0.0, 1.0]), "p[1] must be the noul value"


def test_soft_labels_are_normalised_and_preferred():
    ex = score("s", "rate", ["a", "b", "c", "d"], label=0,
               soft_label=[1.0, 4.0, 21.0, 74.0])
    assert ex.target.sum() == pytest.approx(1.0)
    assert ex.target.argmax() == 3, "the soft label wins over the hard one"


def test_an_example_needs_some_label():
    with pytest.raises(ValueError, match="needs a label"):
        TypedExample("s", Question("choice", "q", {"a": None, "b": None}))


def test_score_rejects_too_many_levels():
    with pytest.raises(ValueError, match="at most 10"):
        score("s", "rate", [f"l{i}" for i in range(11)], label=0)


def test_label_outside_the_option_range_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        choice("s", "q", {"a": None, "b": None}, label=5)


# ----------------------------------------------------- the permutation contract
def test_choice_options_are_shuffled_and_the_label_follows():
    ex = choice("s", "q", {f"o{i}": None for i in range(6)}, label=0)
    rng = np.random.default_rng(3)
    moved = ex.permuted(rng)
    assert moved.question.options[moved.label] == ex.question.options[ex.label], \
        "shuffling must carry the label with it"


def test_score_levels_are_NEVER_shuffled():
    """Order is the only structure that makes off-by-one cheaper than off-by-three."""
    ex = score("s", "rate", ["low", "mid", "high", "max"], label=1)
    for seed in range(8):
        moved = ex.permuted(np.random.default_rng(seed))
        assert moved.question.options == ex.question.options
        assert moved.label == ex.label


def test_permutation_carries_a_soft_label_too():
    ex = choice("s", "q", {f"o{i}": None for i in range(4)},
                soft_label=[0.1, 0.2, 0.3, 0.4])
    moved = ex.permuted(np.random.default_rng(1))
    assert sorted(moved.target) == pytest.approx(sorted(ex.target))


# ------------------------------------------------------------------- loading
def test_load_jsonl_round_trip(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"state": "a", "type": "choice", "instructions": "q",
         "criteria": {"x": None, "y": None}, "label": "y"},
        {"state": "b", "type": "score", "instructions": "r",
         "criteria": ["lo", "hi"], "label": 1},
        {"state": "c", "type": "noul", "instructions": "t", "label": True},
    ]))
    ex = load_jsonl(p)
    assert [e.question.type for e in ex] == ["choice", "score", "noul"]
    assert ex[0].label == 1 and ex[1].label == 1 and ex[2].label == 1
