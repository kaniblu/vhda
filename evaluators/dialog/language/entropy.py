__all__ = ["WordEntropyEvaluator"]

import collections
from dataclasses import dataclass
from typing import Sequence, Optional

import torch

import utils
from utils import TensorMap
from datasets import DialogDataset
from ...evaluator import DialogEvaluator


@dataclass
class WordEntropyEvaluator(DialogEvaluator):
    """arXiv:1605.06069"""
    dataset: DialogDataset
    _prob: torch.Tensor = utils.private_field(default=None)
    _spkr_prob: dict = utils.private_field(default=None)
    _values: dict = utils.private_field(default_factory=dict)

    def __post_init__(self):
        self._prob, self._spkr_prob = self.compute_unigram_prob()

    @property
    def speakers(self):
        return set(spkr for spkr in self.vocabs.speaker.f2i if spkr != "<unk>")

    @property
    def vocabs(self):
        return self.dataset.processor.vocabs

    def compute_unigram_prob(self):
        prob = torch.ones(len(self.vocabs.word)).long()  # +1 smoothing
        spkr_prob = {spkr: torch.ones(len(self.vocabs.word)).long()
                     for spkr in self.speakers}
        for dialog in self.dataset.data:
            for turn in dialog.turns:
                if turn.speaker == "<unk>":
                    continue
                tokens = \
                    self.dataset.processor.sent_processor.process(turn.text)
                tokens = utils.lstrip(tokens, "<bos>")
                tokens = utils.rstrip(tokens, "<eos>")
                for word in tokens:
                    word_idx = self.vocabs.word[word]
                    spkr_prob[turn.speaker][word_idx] += 1
                    prob[word_idx] += 1
        prob = prob.float() / prob.sum()
        spkr_prob = {spkr: p.float() / p.sum() for spkr, p in spkr_prob.items()}
        return prob, spkr_prob

    def reset(self):
        self._values.clear()

    def compute_entropy(self, text: str) -> Optional[float]:
        tokens = self.dataset.processor.tokenize(text)
        if not tokens:
            return {
                "word-ent": torch.tensor(0.0),
                "word-ent-sent": torch.tensor(0.0)
            }
        words, counts = zip(*collections.Counter(tokens).items())
        words = [self.vocabs.word[w] for w in words]
        words, counts = torch.tensor(words).long(), torch.tensor(counts).float()
        text_prob = counts.float() / counts.sum()
        ent = (text_prob * self._prob[words]).sum()
        return {
            "word-ent": ent,
            "word-ent-sent": len(tokens) * ent
        }

    def compute_entropy_spkr(self, text: str, spkr) -> Optional[float]:
        tokens = self.dataset.processor.tokenize(text)
        if not tokens:
            return {
                "word-ent": torch.tensor(0.0),
                "word-ent-sent": torch.tensor(0.0)
            }
        words, counts = zip(*collections.Counter(tokens).items())
        words = [self.vocabs.word[w] for w in words]
        words, counts = torch.tensor(words).long(), torch.tensor(counts).float()
        text_prob = counts.float() / counts.sum()
        ent = (text_prob * self._spkr_prob[spkr][words]).sum()
        return {
            "word-ent": ent,
            "word-ent-sent": len(tokens) * ent
        }

    def update(self, samples: Sequence) -> Optional[TensorMap]:
        for sample in samples:
            for turn in sample.output:
                spkr = turn.speaker
                sent = turn.text
                stats = self.compute_entropy(sent)
                if spkr != "<unk>":
                    stats.update({
                        f"{k}-{spkr}": v for k, v in
                        self.compute_entropy_spkr(sent, spkr).items()
                    })
                for k, v in stats.items():
                    if k not in self._values:
                        self._values[k] = list()
                    self._values[k].append(v.item())
        return

    def get(self) -> Optional[TensorMap]:
        return {k: torch.tensor(v).mean() for k, v in self._values.items()}
