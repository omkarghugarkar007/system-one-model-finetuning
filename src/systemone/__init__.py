"""systemone -- a recipe for fine-tuning System One models.

System One models (Laya, Jev) read a state and return *typed decisions with
calibrated probabilities* rather than text. They are fast, small and
self-hostable, and they are explicitly not useful out of the box: Laya's own
card puts its base checkpoint below the majority-class baseline and calls it
"a fast base to specialise, not a zero-shot decision engine".

So the interesting artifact is not the model, it is the recipe. This package is
that recipe, extracted from a reranking research project and generalised:

    systemone.data       typed examples -- choice / score / noul -- and the
                         token-budget diagnostics that decide whether your data
                         even reaches the model
    systemone.model      the model itself: a faithful sequence builder, the
                         decision head, an ordinal head the vendor does not
                         ship, and the token arithmetic
    systemone.train      composite objective and a trainer sized for a laptop
    systemone.calibrate  per-bucket temperature and conformal intervals
    systemone.eval       accuracy, ECE reported per slice, Brier, reliability
    systemone.teachers   distillation sources behind one protocol
    systemone.reranking  THE WORKED EXAMPLE -- the research this came from

Start at `RECIPE.md`. `FINDINGS.md` is what we measured, including the parts
that did not work.
"""
__version__ = "0.3.0"

from . import calibrate, data, eval, model, train  # noqa: F401
from .data.typed import Question, TypedDataset, TypedExample, choice, noul, score

__all__ = ["data", "model", "train", "calibrate", "eval",
           "TypedExample", "Question", "TypedDataset", "choice", "score", "noul",
           "__version__"]
