__all__ = ["StateNoveltyEvaluator"]

from dataclasses import dataclass
from typing import Sequence, Optional, Set

import torch

import utils
from datasets import DialogDataset
from utils import TensorMap
from ...evaluator import DialogEvaluator


@dataclass
class StateNoveltyEvaluator(DialogEvaluator):
    dataset: DialogDataset
    _turn_states: Set[frozenset] = utils.private_field(default_factory=set)
    _values: dict = utils.private_field(default_factory=dict)

    def __post_init__(self):
        self.prepare_turn_states()

    def prepare_turn_states(self):
        for dialog in self.dataset.data:
            for turn in dialog:
                state = frozenset(turn.state)
                self._turn_states.add(state)

    def reset(self):
        self._values.clear()

    @property
    def speakers(self):
        return set(spkr for spkr in self.vocabs.speaker.f2i if spkr != "<unk>")

    def update(self, samples: Sequence) -> Optional[TensorMap]:
        for sample in samples:
            stats = {"novel-a": torch.tensor([
                frozenset(turn.state) not in self._turn_states
                for turn in sample.output
            ]).float().mean()}
            for k, v in stats.items():
                if k not in self._values:
                    self._values[k] = list()
                self._values[k].append(v.item())
        return

    def get(self) -> Optional[TensorMap]:
        return {k: torch.tensor(v).mean() for k, v in self._values.items()}
