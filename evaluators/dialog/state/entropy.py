__all__ = ["StateEntropyEvaluator"]

import collections
from dataclasses import dataclass
from typing import Sequence, Optional

import torch

import utils
from utils import TensorMap
from datasets import ActSlotValue
from datasets import DialogDataset
from ...evaluator import DialogEvaluator


@dataclass
class StateEntropyEvaluator(DialogEvaluator):
    """(New!)"""
    dataset: DialogDataset
    _asv_prob: torch.Tensor = utils.private_field(default=None)
    _spkr_asv_prob: dict = utils.private_field(default_factory=dict)
    _values: dict = utils.private_field(default_factory=dict)

    def __post_init__(self):
        self.compute_distributions()

    @property
    def speakers(self):
        return set(spkr for spkr in self.vocabs.speaker.f2i if spkr != "<unk>")

    @property
    def vocabs(self):
        return self.dataset.processor.vocabs

    def compute_distributions(self):
        # +1 smoothing
        prob = torch.ones(len(self.vocabs.state.asv)).long()
        spkr_prob = {
            spkr: torch.ones(len(self.vocabs.speaker_state[spkr].asv)).long()
            for spkr in self.speakers
        }
        for dialog in self.dataset.data:
            for turn in dialog.turns:
                if turn.speaker == "<unk>":
                    continue
                spkr = turn.speaker
                for asv in turn.state:
                    asv_idx = self.vocabs.state.asv[asv]
                    spkr_asv_idx = self.vocabs.speaker_state[spkr].asv[asv]
                    spkr_prob[spkr][spkr_asv_idx] += 1
                    prob[asv_idx] += 1
        prob = prob.float() / prob.sum()
        spkr_prob = {spkr: p.float() / p.sum() for spkr, p in spkr_prob.items()}
        self._asv_prob, self._spkr_asv_prob = prob, spkr_prob

    def reset(self):
        self._values.clear()

    def compute_entropy(self, state: Sequence[ActSlotValue]) -> Optional[float]:
        if not state:
            return {
                "asv-ent": torch.tensor(0.0),
                "asv-ent-turn": torch.tensor(0.0)
            }
        asvs, counts = zip(*collections.Counter(state).items())
        asvs = [self.vocabs.state.asv[w] for w in asvs]
        asvs, counts = torch.tensor(asvs).long(), torch.tensor(counts).float()
        text_prob = counts.float() / counts.sum()
        ent = (text_prob * self._asv_prob[asvs]).sum()
        return {
            "asv-ent": ent,
            "asv-ent-turn": len(state) * ent
        }

    def compute_entropy_spkr(self, state: Sequence[ActSlotValue],
                             spkr: str) -> Optional[float]:
        if not state:
            return {
                "asv-ent": torch.tensor(0.0),
                "asv-ent-turn": torch.tensor(0.0)
            }
        asvs, counts = zip(*collections.Counter(state).items())
        asvs = [self.vocabs.speaker_state[spkr].asv[w] for w in asvs]
        asvs, counts = torch.tensor(asvs).long(), torch.tensor(counts).float()
        text_prob = counts.float() / counts.sum()
        ent = (text_prob * self._spkr_asv_prob[spkr][asvs]).sum()
        return {
            "asv-ent": ent,
            "asv-ent-turn": len(state) * ent
        }

    def update(self, samples: Sequence) -> Optional[TensorMap]:
        for sample in samples:
            for turn_gold, turn in zip(sample.input, sample.output):
                spkr = turn.speaker
                asvs = list(turn.state)
                stats = self.compute_entropy(asvs)
                if spkr != "<unk>":
                    stats.update({
                        f"{k}-{spkr}": v for k, v in
                        self.compute_entropy_spkr(asvs, spkr).items()
                    })
                for k, v in stats.items():
                    if k not in self._values:
                        self._values[k] = list()
                    self._values[k].append(v.item())
        return

    def get(self) -> Optional[TensorMap]:
        return {k: torch.tensor(v).mean() for k, v in self._values.items()}
