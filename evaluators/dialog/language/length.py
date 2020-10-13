__all__ = ["SentLengthEvaluator"]

from dataclasses import dataclass
from typing import Sequence, Optional, List

import torch
import numpy as np

import utils
from datasets import VocabSet
from utils import TensorMap
from ...evaluator import DialogEvaluator


@dataclass
class SentLengthEvaluator(DialogEvaluator):
    vocabs: VocabSet
    _lens: List[int] = utils.private_field(default_factory=list)
    _lens_spkr: dict = utils.private_field(default_factory=dict)

    def reset(self):
        self._lens.clear()
        self._lens_spkr.clear()

    def update(self, samples: Sequence) -> Optional[TensorMap]:
        lens = [len(turn.text.split()) for sample in samples
                for turn in sample.output.turns]
        self._lens.extend(lens)
        for spkr in self.vocabs.speaker.f2i:
            if spkr == "<unk>":
                continue
            if spkr not in self._lens_spkr:
                self._lens_spkr[spkr] = list()
            lens = [len(turn.text.split()) for sample in samples
                    for turn in sample.output.turns if turn.speaker == spkr]
            self._lens_spkr[spkr].extend(lens)
        return

    def get(self) -> TensorMap:
        stats = {"sent-len": torch.tensor(np.mean(self._lens))}
        for spkr in self.vocabs.speaker.f2i:
            if spkr == "<unk>":
                continue
            stats[f"sent-len-{spkr}"] = \
                torch.tensor(np.mean(self._lens_spkr[spkr]))
        return stats
