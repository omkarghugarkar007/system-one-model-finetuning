"""Corpora, first-stage retrieval and the evidence slicer for the example."""
from .beir import AVAILABLE, Dataset, Doc, load_beir
from .bm25 import BM25Index, tokenize
from .signatures import make_signature_builder, signature_recall

__all__ = ["load_beir", "Dataset", "Doc", "AVAILABLE", "BM25Index", "tokenize",
           "make_signature_builder", "signature_recall"]
