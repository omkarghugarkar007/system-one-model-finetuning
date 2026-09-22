"""Training: the composite objective and a trainer sized for a laptop."""
from .losses import (
                     CompositeRankingLoss,
                     coral_loss,
                     distill_kl,
                     lambda_ranknet_loss,
                     listnet_loss,
                     rps_loss,
)

__all__ = ["distill_kl", "coral_loss", "rps_loss", "listnet_loss",
           "lambda_ranknet_loss", "CompositeRankingLoss"]
