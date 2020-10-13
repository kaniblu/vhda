__all__ = ["VHDAInferencer"]

from dataclasses import dataclass

import torch

import utils
from datasets import BatchData
from .inferencer import Inferencer


@dataclass
class VHDAInferencer(Inferencer):
    sample_scale: float = 1.0
    dropout_scale: utils.Scheduler = utils.ConstantScheduler(1.0)

    def model_kwargs(self) -> dict:
        kwargs = super().model_kwargs()
        kwargs["sample_scale"] = self.sample_scale
        kwargs["dropout_scale"] = self.dropout_scale.get(self.global_step)
        return kwargs

    def on_batch_ended(self, batch: BatchData, pred: BatchData, outputs
                       ) -> utils.TensorMap:
        stats = dict(super().on_batch_ended(batch, pred, outputs))
        dropout_scale = self.dropout_scale.get(self.global_step)
        stats["dropout-scale"] = torch.tensor(dropout_scale).to(self.device)
        return stats
