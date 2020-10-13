__all__ = ["DialogLengthEvaluator"]

from dataclasses import dataclass
from typing import Sequence, Optional, List

import torch

import utils
from utils import TensorMap
from ..evaluator import DialogEvaluator


@dataclass
class DialogLengthEvaluator(DialogEvaluator):
    _lens: List[int] = utils.private_field(default_factory=list)

    def reset(self):
        self._lens.clear()

    def update(self, samples: Sequence) -> Optional[TensorMap]:
        lens = [len(sample.output) for sample in samples]
        self._lens.extend(lens)
        return {"conv-len": torch.tensor(lens).float().mean()}

    def get(self) -> TensorMap:
        return {"conv-len": torch.tensor(self._lens).float().mean()}
