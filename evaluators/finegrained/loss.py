__all__ = ["LossEvaluator"]

from dataclasses import dataclass
from typing import Optional, List, Union, Sequence

import torch

import utils
from datasets import Dialog
from datasets import BatchData
from utils import TensorMap
from utils import Stacked1DTensor
from losses import Loss
from ..evaluator import FinegrainedEvaluator


@dataclass
class LossEvaluator(FinegrainedEvaluator):
    loss: Loss

    def reset(self):
        pass

    def update(self, batch: Union[Sequence[Dialog], BatchData],
               pred: Union[Sequence[Dialog], BatchData], outputs
               ) -> Optional[TensorMap]:
        return self.loss.compute(batch, outputs)[1]

    def get(self) -> TensorMap:
        return {}
