"""The deliverable: a quality/cost frontier, not a leaderboard row.

Part X is explicit that FrontierRank is interesting only if it moves the
Pareto curve outward, and that reporting the highest nDCG@10 would be
answering a different question. So the primary artifact is a scatter of
dollars per million searches against nDCG@10, with each configuration a point
and the lambda_C sweep a connected curve.

This module deliberately computes the frontier rather than drawing a picture
of it. `pareto_front` is what makes the claim checkable: a configuration is
either on the non-dominated set or it is not, and that is arithmetic. The plot
is a convenience on top.

Costs come from `eval.cost`, which prices a configuration from measured
throughput rather than from a vendor's list price where the two disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Point", "pareto_front", "dominated_by", "frontier_table",
           "plot_frontier"]


@dataclass
class Point:
    """One configuration on the quality/cost plane."""
    name: str
    cost_per_1k: float          # dollars per 1,000 queries
    quality: float              # nDCG@10, or whatever metric is being swept
    family: str = ""            # groups a lambda_C sweep into one curve
    ci: tuple = (np.nan, np.nan)
    escalation_rate: float = np.nan
    latency_p50_ms: float = np.nan
    latency_p95_ms: float = np.nan
    meta: dict = field(default_factory=dict)

    @property
    def cost_per_million(self) -> float:
        return self.cost_per_1k * 1000.0


def dominated_by(a: Point, b: Point, tol: float = 1e-12) -> bool:
    """True when b is at least as good as a on both axes and strictly better on one."""
    cheaper = b.cost_per_1k <= a.cost_per_1k + tol
    better = b.quality >= a.quality - tol
    strict = (b.cost_per_1k < a.cost_per_1k - tol) or (b.quality > a.quality + tol)
    return cheaper and better and strict


def pareto_front(points) -> list[Point]:
    """The non-dominated set: cheaper is better, higher quality is better.

    Returned sorted by cost, which is the order the curve is drawn in and the
    order the claim reads in -- "at every price, this is the best quality
    available".
    """
    pts = list(points)
    front = [p for p in pts if not any(dominated_by(p, q) for q in pts if q is not p)]
    return sorted(front, key=lambda p: (p.cost_per_1k, -p.quality))


def frontier_table(points, front=None) -> str:
    pts = sorted(points, key=lambda p: p.cost_per_1k)
    front_names = {p.name for p in (front if front is not None else pareto_front(pts))}
    lines = [
        f"{'':>2} {'configuration':>34} {'$/1k':>9} {'$/1M':>10} {'nDCG@10':>9} "
        f"{'95% CI':>18} {'escal':>7} {'P95 ms':>8}",
        "-" * 104,
    ]
    for p in pts:
        ci = ("" if not np.isfinite(p.ci[0])
              else f"[{p.ci[0]:+.4f},{p.ci[1]:+.4f}]")
        esc = "" if not np.isfinite(p.escalation_rate) else f"{p.escalation_rate:.1%}"
        p95 = "" if not np.isfinite(p.latency_p95_ms) else f"{p.latency_p95_ms:.0f}"
        lines.append(
            f"{'*' if p.name in front_names else ' ':>2} {p.name:>34} "
            f"{p.cost_per_1k:>9.4f} {p.cost_per_million:>10.2f} {p.quality:>9.4f} "
            f"{ci:>18} {esc:>7} {p95:>8}")
    lines += ["", "* = on the Pareto frontier (nothing is both cheaper and better).",
              "",
              "The claim is the frontier, not the peak. A configuration with the",
              "highest nDCG@10 that is not starred has been beaten on price by",
              "something of equal quality, and reporting it as the result would be",
              "answering a question nobody asked."]
    return "\n".join(lines)


def improvement_over(points, baseline_name: str) -> dict:
    """At the baseline's price, how much more quality does the frontier give --
    and at the baseline's quality, how much cheaper?

    These two numbers are the honest summary of "we moved the frontier". A
    single nDCG delta at an unstated price is not.
    """
    pts = list(points)
    base = next((p for p in pts if p.name == baseline_name), None)
    if base is None:
        raise KeyError(f"no configuration named {baseline_name!r}")
    front = pareto_front(pts)

    at_price = [p for p in front if p.cost_per_1k <= base.cost_per_1k]
    at_quality = [p for p in front if p.quality >= base.quality]
    best_q = max(at_price, key=lambda p: p.quality, default=None)
    cheapest = min(at_quality, key=lambda p: p.cost_per_1k, default=None)
    return {
        "baseline": baseline_name,
        "baseline_cost_per_1k": base.cost_per_1k,
        "baseline_quality": base.quality,
        "quality_at_baseline_price": None if best_q is None else best_q.quality,
        "quality_gain": None if best_q is None else best_q.quality - base.quality,
        "quality_gain_from": None if best_q is None else best_q.name,
        "cost_at_baseline_quality": None if cheapest is None else cheapest.cost_per_1k,
        "cost_ratio": None if cheapest is None or cheapest.cost_per_1k <= 0
        else base.cost_per_1k / cheapest.cost_per_1k,
        "cost_saving_from": None if cheapest is None else cheapest.name,
    }


def plot_frontier(points, path: str = "runs/frontier.png",
                  baseline_names=(), title: str = "FrontierRank quality/cost frontier"):
    """Scatter with a log-scale cost axis; sweeps within a family are connected.

    Log scale is not stylistic: Part IV's cost table spans four orders of
    magnitude, from $0.05 to $256 per 1,000 queries, and a linear axis would
    collapse every self-hosted configuration onto the origin.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = list(points)
    front = pareto_front(pts)
    fig, ax = plt.subplots(figsize=(9, 6))

    families: dict[str, list] = {}
    for p in pts:
        families.setdefault(p.family or p.name, []).append(p)

    for fam, group in sorted(families.items()):
        group = sorted(group, key=lambda p: p.cost_per_million)
        x = [p.cost_per_million for p in group]
        y = [p.quality for p in group]
        if len(group) > 1:                      # a lambda_C sweep is a curve
            ax.plot(x, y, "-o", label=fam, linewidth=2, markersize=5, zorder=3)
        else:
            ax.scatter(x, y, s=70, label=fam, zorder=3)
            ax.annotate(group[0].name, (x[0], y[0]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)

    fx = [p.cost_per_million for p in front]
    fy = [p.quality for p in front]
    ax.step(fx, fy, where="post", color="0.4", linestyle="--", linewidth=1.2,
            zorder=2, label="Pareto frontier")

    for p in pts:
        if np.isfinite(p.ci[0]):
            ax.vlines(p.cost_per_million, p.quality + p.ci[0], p.quality + p.ci[1],
                      color="0.6", linewidth=1, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("$ per million searches (log scale)")
    ax.set_ylabel("nDCG@10")
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
