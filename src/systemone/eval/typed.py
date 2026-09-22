"""Evaluating a fine-tuned System One model. Accuracy is the least of it.

A System One model's product is *calibrated probabilities*, so accuracy alone
misses the property you fine-tuned for. Four things belong in every report:

  accuracy vs the MAJORITY-CLASS baseline, not vs chance. Laya's card compares
      its base checkpoint to both (0.362 against 0.318 random and 0.461
      majority) and only the second is informative -- a model below the
      majority baseline is worse than a constant.

  ECE PER SLICE, never only in aggregate. Laya's own eval reports an aggregate
      ECE of 0.030 concealing a task family at 0.438. `report()` prints the
      worst slice next to the aggregate and the ratio between them.

  Brier, which unlike ECE is a proper scoring rule, so it cannot be gamed by
      spreading probability mass around.

  risk-coverage (AURC), which answers the question an escalation policy
      actually asks: if I only act on my most confident answers, how wrong am I?

`report()` is the thing to call.
"""
from __future__ import annotations

import math

import numpy as np

from .metrics import ece

__all__ = ["accuracy", "brier", "aurc", "confidence_from_probs", "slice_report",
           "report"]


def accuracy(probs, labels) -> float:
    return float(np.mean([int(np.argmax(p) == y) for p, y in zip(probs, labels, strict=False)]))


def brier(probs, labels) -> float:
    """Multiclass Brier: mean squared error against the one-hot truth."""
    tot = 0.0
    for p, y in zip(probs, labels, strict=False):
        p = np.asarray(p, dtype=float)
        t = np.zeros_like(p)
        t[y] = 1.0
        tot += float(np.sum((p - t) ** 2))
    return tot / max(1, len(labels))


def confidence_from_probs(p, k: int | None = None) -> float:
    """1 - normalised entropy. The same definition the inference API returns."""
    p = np.asarray(p, dtype=float)
    k = k or p.size
    if k < 2:
        return 1.0
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return float(1 - ent / math.log(k))


def aurc(conf, correct) -> float:
    """Area under the risk-coverage curve, lower is better.

    This is the metric that matters if you plan to escalate: it measures
    whether the model's confidence ORDERS its errors, which is a weaker and
    more useful property than being calibrated.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return float("nan")
    order = np.argsort(-conf)
    err = 1.0 - correct[order]
    return float((np.cumsum(err) / np.arange(1, err.size + 1)).mean())


def _slice_key(ex, by: str) -> str:
    if by == "type":
        return ex.question.type
    if by == "options":
        k = ex.question.n_options
        return f"{ex.question.type}:{'2' if k <= 2 else '3-5' if k <= 5 else '6-10' if k <= 10 else '11+'}"
    return str(ex.meta.get(by, "unknown"))


def slice_report(predictions, by: str = "options", n_bins: int = 15) -> dict:
    """Metrics per slice plus the aggregate, and the ratio between them."""
    groups: dict[str, list] = {}
    for pred in predictions:
        ex = pred["example"]
        if ex.label is None:
            continue
        groups.setdefault(_slice_key(ex, by), []).append(pred)

    out: dict[str, dict] = {}
    all_conf, all_ok, all_p, all_y = [], [], [], []
    for key, rows in sorted(groups.items()):
        p = [r["probs"] for r in rows]
        y = [r["example"].label for r in rows]
        conf = np.array([float(np.max(q)) for q in p])
        ok = np.array([float(np.argmax(q) == t) for q, t in zip(p, y, strict=False)])
        counts = np.bincount(y, minlength=max(y) + 1)
        out[key] = {"n": len(rows), "accuracy": round(accuracy(p, y), 4),
                    "majority": round(float(counts.max() / len(y)), 4),
                    "ece": round(float(ece(conf, ok, n_bins)), 4),
                    "brier": round(brier(p, y), 4),
                    "aurc": round(aurc(conf, ok), 4),
                    "mean_conf": round(float(conf.mean()), 4)}
        all_conf.append(conf)
        all_ok.append(ok)
        all_p += list(p)
        all_y += list(y)

    if all_conf:
        conf = np.concatenate(all_conf)
        ok = np.concatenate(all_ok)
        counts = np.bincount(all_y, minlength=max(all_y) + 1)
        agg = {"n": int(conf.size), "accuracy": round(accuracy(all_p, all_y), 4),
               "majority": round(float(counts.max() / len(all_y)), 4),
               "ece": round(float(ece(conf, ok, n_bins)), 4),
               "brier": round(brier(all_p, all_y), 4),
               "aurc": round(aurc(conf, ok), 4),
               "mean_conf": round(float(conf.mean()), 4)}
        worst = max((v["ece"] for v in out.values()), default=0.0)
        agg["worst_slice_ece"] = round(worst, 4)
        # how much the aggregate is flattering you
        agg["hiding_ratio"] = round(worst / max(agg["ece"], 1e-9), 2)
        out["__aggregate__"] = agg
    return out


def report(predictions, by: str = "options") -> str:
    r = slice_report(predictions, by)
    agg = r.get("__aggregate__", {})
    lines = [f"{'slice':>16} {'n':>6} {'acc':>7} {'major':>7} {'ECE':>7} "
             f"{'Brier':>7} {'AURC':>7} {'conf':>7}", "-" * 68]
    for k, v in r.items():
        if k == "__aggregate__":
            continue
        lines.append(f"{k:>16} {v['n']:>6} {v['accuracy']:>7.4f} {v['majority']:>7.4f} "
                     f"{v['ece']:>7.4f} {v['brier']:>7.4f} {v['aurc']:>7.4f} "
                     f"{v['mean_conf']:>7.4f}")
    if agg:
        lines += ["-" * 68,
                  f"{'AGGREGATE':>16} {agg['n']:>6} {agg['accuracy']:>7.4f} "
                  f"{agg['majority']:>7.4f} {agg['ece']:>7.4f} {agg['brier']:>7.4f} "
                  f"{agg['aurc']:>7.4f} {agg['mean_conf']:>7.4f}", ""]
        verdict = ("BEATS" if agg["accuracy"] > agg["majority"] else "LOSES TO")
        lines.append(f"  accuracy {verdict} the majority-class baseline "
                     f"({agg['accuracy']:.4f} vs {agg['majority']:.4f})")
        lines.append(f"  worst slice ECE {agg['worst_slice_ece']:.4f} vs aggregate "
                     f"{agg['ece']:.4f}  ({agg['hiding_ratio']}x)")
        if agg["hiding_ratio"] > 2.0:
            lines.append("  -> the aggregate is hiding a badly calibrated slice. "
                         "Fit a temperature PER BUCKET, not globally.")
    return "\n".join(lines)
