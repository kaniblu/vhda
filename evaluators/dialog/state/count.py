__all__ = ["StateCountEvaluator"]

from dataclasses import dataclass
from typing import Sequence, Optional

import torch

import utils
from datasets import VocabSet
from utils import TensorMap
from ...evaluator import DialogEvaluator


@dataclass
class StateCountEvaluator(DialogEvaluator):
    vocabs: VocabSet
    _values: dict = utils.private_field(default_factory=dict)

    def reset(self):
        self._values.clear()

    @property
    def speakers(self):
        return set(spkr for spkr in self.vocabs.speaker.f2i if spkr != "<unk>")

    def update(self, samples: Sequence) -> Optional[TensorMap]:
        for sample in samples:
            for turn in sample.output:
                if turn.speaker == "<unk>":
                    continue
                spkr = turn.speaker
                stats = {
                    "state-cnt": torch.tensor(len(turn.state)).float(),
                    f"state-cnt-{spkr}": torch.tensor(len(turn.state)).float()
                }
                for k, v in stats.items():
                    if k not in self._values:
                        self._values[k] = list()
                    self._values[k].append(v.item())
            stats = {"state-cnt-conv": torch.tensor(
                sum(1 for turn in sample.output for _ in turn.state)).float()}
            for k, v in stats.items():
                if k not in self._values:
                    self._values[k] = list()
                self._values[k].append(v.item())
        return

    def get(self) -> Optional[TensorMap]:
        return {k: torch.tensor(v).mean() for k, v in self._values.items()}
