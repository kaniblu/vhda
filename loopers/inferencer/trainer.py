__all__ = ["TrainInferencer"]

import itertools
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.optim as op

import utils
from datasets import BatchData
from .loss import LossInferencer


@dataclass
class TrainInferencer(LossInferencer):
    optimizer_cls: Callable[[Iterable[nn.Parameter]], op.Optimizer] = op.Adam
    l2norm: Optional[float] = None
    grad_clip: Optional[float] = None
    _optimizer: op.Optimizer = field(init=False, repr=False, default=None)

    def __post_init__(self):
        super(TrainInferencer, self).__post_init__()
        assert self.optimizer_cls is not None
        self._optimizer = self.optimizer_cls(
            p for p in self.model.parameters() if p.requires_grad
        )

    def on_batch_started(self, batch: BatchData) -> BatchData:
        batch = super().on_batch_started(batch)
        self.model.train()
        return batch

    def on_batch_ended(self, batch: BatchData, pred: BatchData, outputs
                       ) -> utils.TensorMap:
        stats = dict(super().on_batch_ended(batch, pred, outputs))
        if "loss" not in stats:
            raise KeyError(f"'loss' Tensor not returned by "
                           f"LossInferencer.on_batch_ended")
        if self.l2norm is not None:
            l2norm = sum(p.norm(2) for p in
                         self.model.parameters() if p.requires_grad)
            stats["l2norm"] = l2norm
            stats["loss"] += l2norm * self.l2norm
        loss = stats["loss"]
        self._optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            stats["grad-norm"] = torch.tensor(nn.utils.clip_grad_norm_(
                parameters=itertools.chain(*(d["params"] for d in
                                             self._optimizer.param_groups)),
                max_norm=self.grad_clip
            )).to(self.device)
        self._optimizer.step()
        return stats
