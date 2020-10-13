__all__ = ["Loss"]

from typing import Tuple

import torch

import utils
from datasets import BatchData


class Loss:

    def compute(self, batch: BatchData, outputs, step: int = None
                ) -> Tuple[torch.Tensor, utils.TensorMap]:
        raise NotImplementedError
