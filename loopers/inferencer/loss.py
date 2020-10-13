__all__ = ["LossInferencer"]

from dataclasses import dataclass

import utils
from datasets import BatchData
from losses import Loss
from .inferencer import Inferencer


@dataclass
class LossInferencer(Inferencer):
    loss: Loss = None

    def __post_init__(self):
        super().__post_init__()
        if self.loss is None:
            raise ValueError(f"must provide 'loss' field")

    def on_batch_ended(self, batch: BatchData, pred: BatchData, outputs
                       ) -> utils.TensorMap:
        stats = dict(super().on_batch_ended(batch, pred, outputs))
        loss, loss_stats = self.loss.compute(batch, outputs, self.global_step)
        stats.update({k: v.mean().detach()
                      for k, v in loss_stats.items()})
        stats["loss"] = loss
        return stats
