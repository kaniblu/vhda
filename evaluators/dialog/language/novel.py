"""Measures the novelty of languages"""
__all__ = ["LanguageNoveltyEvaluator"]

from dataclasses import dataclass
from typing import Sequence, Optional, List, Set, Tuple, Dict

import nltk
import torch

import utils
from datasets import DialogDataset
from utils import TensorMap
from ...evaluator import DialogEvaluator


@dataclass
class LanguageNoveltyEvaluator(DialogEvaluator):
    dataset: DialogDataset
    _sents: Set[str] = utils.private_field(default_factory=set)
    _bigrams: Set[Tuple[str, str]] = utils.private_field(default_factory=set)
    _trigrams: Set[Tuple[str, str, str]] = \
        utils.private_field(default_factory=set)
    _spkr_bigrams: Dict[str, Set[Tuple[str, str]]] = \
        utils.private_field(default_factory=dict)
    _spkr_trigrams: Dict[str, Set[Tuple[str, str, str]]] = \
        utils.private_field(default_factory=dict)
    _spkr_sents: Dict[str, Set[str]] = utils.private_field(default_factory=dict)
    _values: Dict[str, List[float]] = \
        utils.private_field(default_factory=dict)

    def __post_init__(self):
        self.prepare_ngrams()

    @property
    def speakers(self):
        return set(spkr for spkr in self.vocabs.speaker.f2i if spkr != "<unk>")

    @property
    def vocabs(self):
        return self.dataset.processor.vocabs

    def prepare_ngrams(self):
        for dialog in self.dataset.data:
            for turn in dialog.turns:
                spkr = turn.speaker
                if spkr == "<unk>":
                    continue
                if spkr not in self._spkr_bigrams:
                    self._spkr_bigrams[spkr] = set()
                    self._spkr_trigrams[spkr] = set()
                    self._spkr_sents[spkr] = set()
                tokens = \
                    self.dataset.processor.sent_processor.process(turn.text)
                tokens = utils.lstrip(tokens, "<bos>")
                tokens = utils.rstrip(tokens, "<eos>")
                for bigram in nltk.bigrams(tokens):
                    self._bigrams.add(tuple(bigram))
                    self._spkr_bigrams[spkr].add(tuple(bigram))
                for trigram in nltk.ngrams(tokens, 3):
                    self._trigrams.add(tuple(trigram))
                    self._spkr_trigrams[spkr].add(tuple(trigram))
                sent = " ".join(tokens)
                self._sents.add(sent)
                self._spkr_sents[spkr].add(sent)

    def reset(self):
        self._values.clear()

    def compute(self, text: str):
        stats = {
            "novel-2": torch.tensor(0.0),
            "novel-3": torch.tensor(0.0),
            "novel-utt": torch.tensor(0.0)
        }
        tokens = text.split()
        bigrams = list(map(tuple, nltk.bigrams(tokens)))
        trigrams = list(map(tuple, nltk.trigrams(tokens)))
        if bigrams:
            stats["novel-2"] = \
                (torch.tensor([w not in self._bigrams for w in bigrams])
                 .float().mean())
        if trigrams:
            stats["novel-3"] = \
                (torch.tensor([w not in self._trigrams for w in trigrams])
                 .float().mean())
        if text:
            stats["novel-utt"] = torch.tensor(text not in self._sents).float()
        return stats

    def compute_spkr(self, text: str, spkr: str):
        stats = {
            "novel-2": torch.tensor(0.0),
            "novel-3": torch.tensor(0.0),
            "novel-utt": torch.tensor(0.0)
        }
        tokens = text.split()
        bigrams = list(map(tuple, nltk.bigrams(tokens)))
        trigrams = list(map(tuple, nltk.trigrams(tokens)))
        if bigrams:
            stats["novel-2"] = \
                (torch.tensor([w not in self._spkr_bigrams[spkr]
                               for w in bigrams])
                 .float().mean())
        if trigrams:
            stats["novel-3"] = \
                (torch.tensor([w not in self._spkr_trigrams[spkr]
                               for w in trigrams])
                 .float().mean())
        if text:
            stats["novel-utt"] = \
                torch.tensor(text not in self._spkr_sents[spkr]).float()
        return stats

    def update(self, samples: Sequence) -> Optional[TensorMap]:
        for sample in samples:
            for turn in sample.output:
                spkr = turn.speaker
                stats = self.compute(turn.text)
                if spkr != "<unk>":
                    stats.update({f"{k}-{spkr}": v for k, v in
                                  self.compute_spkr(turn.text, spkr).items()})
                for k, v in stats.items():
                    if k not in self._values:
                        self._values[k] = list()
                    self._values[k].append(v.item())
        return

    def get(self) -> Optional[TensorMap]:
        return {k: torch.tensor(v).mean() for k, v in self._values.items()}
