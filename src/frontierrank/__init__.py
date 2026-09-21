"""FrontierRank -- reranking that spends compute in proportion to ranking regret.

Layers, in dependency order:

    core/         numpy-only theory: slates, anchor calibration, the regret
                  frontier, the EVI controller, and the token budget that
                  constrains all of it. No model, no network.
    models/       adapters behind two protocols. `models.laya` is the student,
                  `models.teachers` the escalation targets. Nothing outside
                  this package names a vendor.
    data/         corpora, qrels, first-stage retrieval, evidence signatures.
    training/     the composite objective and the fine-tuning loop.
    calibration/  temperature refit and conformal intervals.
    eval/         metrics with TREC conventions enforced, and statistics.
    experiments/  one module per phase gate in the plan.
"""
from . import core
from .core import (Anchor, AnchoredScorer, Controller, ScoredPool, SlateLayout,
                   build_slates, compute_frontier, expected_regret, fit_slate)
from .eval import ece, mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k
from .models.protocols import (RUBRIC_4LEVEL, ScorerProtocol, TeacherProtocol,
                               TeacherVerdict)

__version__ = "0.2.0"
__all__ = [
    "core", "Anchor", "AnchoredScorer", "ScoredPool", "SlateLayout",
    "build_slates", "compute_frontier", "expected_regret", "fit_slate",
    "Controller", "ndcg_at_k", "recall_at_k", "mrr_at_k", "ece",
    "paired_bootstrap", "ScorerProtocol", "TeacherProtocol", "TeacherVerdict",
    "RUBRIC_4LEVEL",
]
