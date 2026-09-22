"""Metrics. Report ECE per slice, never in aggregate -- see `slice_report`."""
from .metrics import ece, mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k
from .typed import accuracy, aurc, brier, confidence_from_probs, report, slice_report

__all__ = ["accuracy", "brier", "aurc", "confidence_from_probs", "ece",
           "slice_report", "report", "paired_bootstrap",
           "ndcg_at_k", "recall_at_k", "mrr_at_k"]
