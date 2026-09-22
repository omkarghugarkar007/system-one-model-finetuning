"""Temperature refit, per (type, option-count) bucket. Non-negotiable, and separate.

Laya ships over-confident and knows it: refitting moves its ECE from 0.466 to
0.081, and the shipped `temperature_by_options` map is per bucket precisely
because a 2-option noul and a 20-option choice need different scaling.

Two rules this module enforces, both learned from the vendor's own eval.

**Fit per bucket, not globally.** The shipped map ranges from T = 0.1006
(choice:11+) to T = 1.98 (noul:2). A single scalar cannot serve both, and
fitting one produces a model that is well calibrated on the majority bucket and
badly calibrated everywhere else.

**Report per slice, never in aggregate.** Laya's own eval shows an aggregate
ECE of 0.030 hiding a task family at 0.438. An aggregate ECE is not evidence of
calibration; it is evidence that most of your data is easy.

Fitted on held-out data, after training, with the backbone frozen. A
temperature fitted on the training set is a temperature fitted to memorised
answers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["fit_temperature", "TemperatureMap", "ece_by_bucket", "reliability_curve"]


def _nll(logits: np.ndarray, labels: np.ndarray, t: float) -> float:
    z = np.asarray(logits, dtype=float) / max(t, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-logp[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    lo: float = 0.02, hi: float = 10.0,
                    iters: int = 60) -> float:
    """Temperature minimising held-out NLL, by golden-section search.

    Search rather than gradient descent because the objective is
    one-dimensional and unimodal, and because a bracketed search cannot run
    away to a degenerate temperature the way an unconstrained optimiser can --
    which matters when a bucket has thirty examples in it.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if logits.ndim != 2 or logits.shape[0] != labels.size:
        raise ValueError(f"shape mismatch: {logits.shape} vs {labels.shape}")
    if labels.size == 0:
        return 1.0
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = _nll(logits, labels, c), _nll(logits, labels, d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _nll(logits, labels, c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _nll(logits, labels, d)
    return float((a + b) / 2.0)


@dataclass
class TemperatureMap:
    """Per-bucket temperatures, in the shape Laya's own config uses.

    `min_samples` guards the thing that goes wrong quietly: a bucket with a
    handful of examples fits a confident, wrong temperature, and because it is
    a rare bucket nobody notices. Under the threshold the bucket falls back to
    the global fit and says so in `report()`.
    """
    temperatures: dict = field(default_factory=dict)
    global_t: float = 1.0
    min_samples: int = 50
    counts: dict = field(default_factory=dict)
    fallbacks: list = field(default_factory=list)

    @staticmethod
    def bucket(qtype: str, k: int) -> str:
        size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
        return f"{qtype}:{size}"

    def get(self, qtype: str, k: int) -> float:
        return self.temperatures.get(self.bucket(qtype, k), self.global_t)

    def apply(self, logits: np.ndarray, qtype: str = "choice") -> np.ndarray:
        z = np.asarray(logits, dtype=float) / self.get(qtype, np.shape(logits)[-1])
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    @classmethod
    def fit(cls, records, min_samples: int = 50) -> "TemperatureMap":
        """`records` is an iterable of (logits, label, qtype)."""
        by: dict[str, list] = {}
        allz, ally = [], []
        for logits, label, qtype in records:
            z = np.asarray(logits, dtype=float)
            by.setdefault(cls.bucket(qtype, z.size), []).append((z, int(label)))
        m = cls(min_samples=min_samples)

        width = max((len(v[0][0]) for v in by.values()), default=2)
        for bucket, rows in by.items():
            k = max(len(z) for z, _ in rows)
            L = np.full((len(rows), k), -1e4)
            y = np.zeros(len(rows), dtype=int)
            for i, (z, lab) in enumerate(rows):
                L[i, :z.size] = z
                y[i] = lab
            m.counts[bucket] = len(rows)
            if len(rows) >= min_samples:
                m.temperatures[bucket] = fit_temperature(L, y)
            else:
                m.fallbacks.append(bucket)
            allz.append((L, y, k))
        if allz:
            kmax = max(k for _, _, k in allz)
            L = np.vstack([np.pad(z, ((0, 0), (0, kmax - z.shape[1])),
                                  constant_values=-1e4) for z, _, _ in allz])
            y = np.concatenate([y for _, y, _ in allz])
            m.global_t = fit_temperature(L, y)
        _ = width
        return m

    def report(self) -> str:
        lines = [f"{'bucket':>14} {'n':>7} {'T':>8} {'source':>10}", "-" * 42]
        for b in sorted(set(self.counts) | set(self.temperatures)):
            fitted = b in self.temperatures
            lines.append(f"{b:>14} {self.counts.get(b, 0):>7} "
                         f"{(self.temperatures.get(b, self.global_t)):>8.4f} "
                         f"{'fitted' if fitted else 'global':>10}")
        lines.append(f"{'(global)':>14} {sum(self.counts.values()):>7} "
                     f"{self.global_t:>8.4f} {'fitted':>10}")
        if self.fallbacks:
            lines.append(f"\nunder {self.min_samples} samples, fell back to global: "
                         + ", ".join(self.fallbacks))
        return "\n".join(lines)


def ece_by_bucket(records, temp: TemperatureMap | None = None,
                  n_bins: int = 15) -> dict:
    """ECE per bucket AND in aggregate, so the two can be compared.

    The comparison is the point. When aggregate ECE is far below the worst
    bucket, the aggregate is measuring the sample mix rather than the model.
    """
    from ..eval import ece

    by: dict[str, list] = {}
    for logits, label, qtype in records:
        z = np.asarray(logits, dtype=float)
        if temp is not None:
            p = temp.apply(z, qtype)
        else:
            e = np.exp(z - z.max())
            p = e / e.sum()
        by.setdefault(TemperatureMap.bucket(qtype, z.size), []).append(
            (float(p.max()), int(np.argmax(p) == label)))

    out = {}
    all_conf, all_ok = [], []
    for b, rows in by.items():
        conf = np.array([c for c, _ in rows])
        ok = np.array([o for _, o in rows], dtype=float)
        out[b] = {"n": len(rows), "ece": round(ece(conf, ok, n_bins), 4),
                  "acc": round(float(ok.mean()), 4),
                  "mean_conf": round(float(conf.mean()), 4)}
        all_conf.append(conf)
        all_ok.append(ok)
    if all_conf:
        c, o = np.concatenate(all_conf), np.concatenate(all_ok)
        out["__aggregate__"] = {"n": int(c.size), "ece": round(ece(c, o, n_bins), 4),
                                "acc": round(float(o.mean()), 4),
                                "mean_conf": round(float(c.mean()), 4)}
        worst = max((v["ece"] for k, v in out.items() if k != "__aggregate__"),
                    default=0.0)
        out["__aggregate__"]["worst_bucket_ece"] = round(worst, 4)
        out["__aggregate__"]["hiding_ratio"] = round(
            worst / max(out["__aggregate__"]["ece"], 1e-9), 2)
    return out


def reliability_curve(conf, correct, n_bins: int = 15):
    """(bin centre, mean confidence, empirical accuracy, count) per bin."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            rows.append(((lo + hi) / 2, float(conf[m].mean()),
                         float(correct[m].mean()), int(m.sum())))
    return rows
