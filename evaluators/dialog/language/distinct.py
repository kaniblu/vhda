__all__ = ["DistinctEvaluator"]

from dataclasses import dataclass
from typing import Sequence, Optional, List, Dict

import torch
import numpy as np
from nltk import ngrams

import utils
from utils import TensorMap
from datasets import VocabSet
from ...evaluator import DialogEvaluator


@dataclass
class DistinctEvaluator(DialogEvaluator):
    vocabs: VocabSet
    ngrams: Sequence[int] = frozenset({1, 2})
    _values: Dict[int, List[float]] = utils.private_field(default_factory=dict)
    _values_spkr: Dict[str, Dict[int, List[float]]] = \
        utils.private_field(default_factory=dict)

    def reset(self):
        self._values.clear()
        self._values_spkr.clear()

    @staticmethod
    def compute_distinct(tokens, n):
        if len(tokens) == 0:
            return 0.0
        vocab = set(ngrams(tokens, n))
        return len(vocab) / len(tokens)

    def compute(self, samples: Sequence, spkr=None):
        return {i: [self.compute_distinct(turn.text, i)
                    for sample in samples for turn in sample.output.turns
                    if spkr is None or turn.speaker == spkr]
                for i in self.ngrams}

    def update(self, samples: Sequence) -> Optional[TensorMap]:
        res = self.compute(samples)
        for i, values in res.items():
            if i not in self._values:
                self._values[i] = list()
            self._values[i].extend(values)
        for spkr in self.vocabs.speaker.f2i:
            if spkr == "<unk>":
                continue
            if spkr not in self._values_spkr:
                self._values_spkr[spkr] = dict()
            res = self.compute(samples, spkr)
            for i, values in res.items():
                if i not in self._values_spkr[spkr]:
                    self._values_spkr[spkr][i] = list()
                self._values_spkr[spkr][i].extend(values)
        return

    def get(self) -> TensorMap:
        stats = {f"dist-{i}": torch.tensor(np.mean(vs))
                 for i, vs in self._values.items()}
        stats.update({f"dist-{i}-{spkr}": torch.tensor(np.mean(vs))
                      for spkr, values in self._values_spkr.items()
                      for i, vs in values.items()})
        return stats
