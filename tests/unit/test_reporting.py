"""The Pareto frontier is the deliverable, so its arithmetic is load-bearing."""
import numpy as np
import pytest

from frontierrank.reporting import (Point, frontier_table, improvement_over,
                                    pareto_front)


def _grid():
    return [
        Point("cheap-bad", 0.05, 0.50, "a"),
        Point("cheap-good", 0.05, 0.60, "a"),      # dominates cheap-bad
        Point("mid", 0.50, 0.65, "b"),
        Point("dear-same", 2.00, 0.65, "c"),       # dominated by mid on price
        Point("dear-best", 2.50, 0.70, "c"),
    ]


def test_front_excludes_anything_beaten_on_both_axes():
    names = {p.name for p in pareto_front(_grid())}
    assert "cheap-bad" not in names, "same price, worse quality"
    assert "dear-same" not in names, "same quality, 4x the price"
    assert {"cheap-good", "mid", "dear-best"} == names


def test_front_is_sorted_by_cost_and_monotone_in_quality():
    front = pareto_front(_grid())
    costs = [p.cost_per_1k for p in front]
    quals = [p.quality for p in front]
    assert costs == sorted(costs)
    assert quals == sorted(quals), "paying more must buy more, or it is dominated"


def test_a_free_baseline_is_always_on_the_front():
    pts = _grid() + [Point("BM25", 0.0, 0.40, "baseline")]
    assert "BM25" in {p.name for p in pareto_front(pts)}


def test_identical_points_do_not_eliminate_each_other():
    a = Point("a", 1.0, 0.5)
    b = Point("b", 1.0, 0.5)
    assert len(pareto_front([a, b])) == 2


def test_cost_per_million_is_per_1k_times_1000():
    assert Point("x", 0.096, 0.7).cost_per_million == pytest.approx(96.0)


def test_improvement_reports_both_axes():
    pts = _grid()
    out = improvement_over(pts, "dear-best")
    # nothing on the front is better at or below 2.50, so the gain is zero
    assert out["quality_gain"] == pytest.approx(0.0)
    out = improvement_over(pts, "dear-same")
    assert out["cost_ratio"] == pytest.approx(4.0), "mid matches quality at 1/4 the price"
    assert out["cost_saving_from"] == "mid"


def test_improvement_rejects_an_unknown_baseline():
    with pytest.raises(KeyError):
        improvement_over(_grid(), "nope")


def test_table_stars_exactly_the_front():
    pts = _grid()
    table = frontier_table(pts)
    front = {p.name for p in pareto_front(pts)}
    for line in table.splitlines():
        for p in pts:
            if line.rstrip().endswith(tuple()) and f" {p.name} " in line:
                assert line.startswith(" *") == (p.name in front)
