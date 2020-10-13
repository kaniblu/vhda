__all__ = ["EmbeddingEvaluator"]

import logging
from dataclasses import dataclass
from typing import Set, Sequence, Optional

import torch

import utils
from utils import TensorMap
from embeds import Embeddings
from ...evaluator import DialogEvaluator


def cosine_sim(a, b, dim=0):
    return a.mul(b).sum(dim) / a.pow(2).sum(dim).mul(b.pow(2).sum(dim)).sqrt()


@dataclass
class EmbeddingEvaluator(DialogEvaluator):
    vocab: utils.Vocabulary
    embeds: Embeddings
    _emb: torch.Tensor = utils.private_field(default=None)
    _cache: Set[int] = utils.private_field(default_factory=set)
    _stats: dict = utils.private_field(default_factory=dict)
    _seen: int = utils.private_field(default=0)
    _logger: logging.Logger = utils.private_field(default=None)

    def __post_init__(self):
        self._emb = torch.zeros(len(self.vocab), self.embeds.dim,
                                requires_grad=False)
        self._logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def compare_mean(pred: torch.Tensor, gold: torch.Tensor):
        return cosine_sim(pred.mean(0), gold.mean(0))

    @staticmethod
    def compare_extrema(pred: torch.Tensor, gold: torch.Tensor):
        def extrema(x):
            return x.gather(0, x.abs().max(0)[1].unsqueeze(0)).squeeze(0)

        return cosine_sim(extrema(pred), extrema(gold))

    @staticmethod
    def compare_greedy(pred: torch.Tensor, gold: torch.Tensor):
        dim = pred.size(-1)
        return cosine_sim(
            (pred.unsqueeze(1).expand(-1, gold.size(0), -1)
             .contiguous().view(-1, dim)),
            (gold.unsqueeze(0).expand(pred.size(0), -1, -1)
             .contiguous().view(-1, dim)),
            dim=1
        ).view(pred.size(0), gold.size(0)).max(0)[0].mean()

    def to_embeddings(self, words: torch.Tensor):
        self._emb = self._emb
        new_idx = set(e.item() for e in words.unique()) - self._cache
        for i in new_idx:
            w = self.vocab.i2f[i]
            if w not in self.embeds:
                self._logger.debug(f"word ({w}) not found in the embeddings")
            else:
                self._emb[i, :] = torch.tensor(self.embeds[w])
        if new_idx:
            self._cache.update(new_idx)
        exists = torch.tensor([self.vocab.i2f[w.item()] in self.embeds
                               for w in words]).bool()
        return self._emb[words.masked_select(exists)]

    def _eval(self, p, g, compare_fn) -> Optional[torch.Tensor]:
        pe = self.to_embeddings(p)
        ge = self.to_embeddings(g)
        if not len(ge):
            self._logger.debug(
                f"entire utterance omitted as no matching "
                f"embeddings found for the gold sentence"
            )
            return
        if not len(pe):
            self._logger.debug(
                f"no matching embeddings found for the pred sentence; "
                f"giving a zero score for the similarity score"
            )
            return torch.tensor(0.0)
        res = compare_fn(pe, ge)
        if (res != res).any():
            self._logger.debug(f"NaN detected for pred = {pu} and gold = {gu}")
            return
        return res

    def reset(self):
        self._seen = 0
        self._stats.clear()

    def update(self, samples: Sequence) -> Optional[TensorMap]:
        pred = [torch.tensor([self.vocab[w] for turn in sample.output.turns
                              for w in turn.text.split()]).long()
                for sample in samples]
        batch = [torch.tensor([self.vocab[w] for turn in sample.input.turns
                               for w in turn.text.split()]).long()
                 for sample in samples]
        for p, g in zip(pred, batch):
            stats = {
                "emb-mean": self._eval(p, g, self.compare_mean),
                "emb-extrema": self._eval(p, g, self.compare_extrema),
                "emb-greedy": self._eval(p, g, self.compare_greedy)
            }
            for k, v in stats.items():
                if v is None:
                    continue
                if k not in self._stats:
                    self._stats[k] = list()
                self._stats[k].append(v.item())
        return

    def get(self) -> TensorMap:
        return {k: torch.tensor(v).mean() for k, v in self._stats.items()}
