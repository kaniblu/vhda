__all__ = ["DistinctStateEvaluator"]

from dataclasses import dataclass
from typing import Sequence, Optional

import torch

import utils
from utils import TensorMap
from datasets import VocabSet
from ...evaluator import DialogEvaluator


@dataclass
class DistinctStateEvaluator(DialogEvaluator):
    vocabs: VocabSet
    _values: dict = utils.private_field(default_factory=dict)

    def reset(self):
        self._values.clear()

    @property
    def speakers(self):
        return set(spkr for spkr in self.vocabs.speaker.f2i if spkr != "<unk>")

    @staticmethod
    def compute_distinct(tokens):
        if len(tokens) == 0:
            return torch.tensor(0.0)
        return torch.tensor(len(set(tokens)) / len(tokens))

    def compute(self, samples: Sequence, spkr=None):
        return {i: [self.compute_distinct(turn.text, i)
                    for sample in samples for turn in sample.output.turns
                    if spkr is None or turn.speaker == spkr]
                for i in self.ngrams}

    def update(self, samples: Sequence) -> Optional[TensorMap]:
        for sample in samples:
            asvs = [asv for turn in sample.output if turn.speaker != "<unk>"
                    for asv in turn.state]
            spkr_asvs = {spkr: [asv for turn in sample.output
                                if turn.speaker != "<unk>"
                                for asv in turn.state]
                         for spkr in self.speakers}
            stats = {"dist-a": self.compute_distinct(asvs)}
            stats.update({
                f"dist-a-{spkr}": self.compute_distinct(spkr_asvs[spkr])
                for spkr in self.speakers
            })
            for k, v in stats.items():
                if k not in self._values:
                    self._values[k] = list()
                self._values[k].append(v.item())
        return

    def get(self) -> Optional[TensorMap]:
        return {k: torch.tensor(v).mean() for k, v in self._values.items()}
