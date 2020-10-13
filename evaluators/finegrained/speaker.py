__all__ = ["SpeakerEvaluator"]

from dataclasses import dataclass
from typing import Optional, List

import torch

import utils
from datasets import BatchData
from utils import TensorMap
from utils import Stacked1DTensor
from ..evaluator import FinegrainedEvaluator


@dataclass
class SpeakerEvaluator(FinegrainedEvaluator):
    vocab: utils.Vocabulary
    detail: bool = False
    return_update: bool = False
    _pred: List[torch.Tensor] = utils.private_field(default_factory=list)
    _gold: List[torch.Tensor] = utils.private_field(default_factory=list)
    _unk_idx: int = utils.private_field(default=None)

    def __post_init__(self):
        self._unk_idx = self.vocab["<unk>"]

    @property
    def requires_lexical_form(self):
        return False

    def compute_rec_prec_cls(self, idx, pred, gold) -> utils.TensorMap:
        if idx not in self.vocab:
            raise ValueError(f"not a valid speaker idx: {idx}")
        spkr = self.vocab[idx]
        seq_mask = utils.mask(pred.lens, pred.size(1))
        crt, prec_mask, rec_mask = \
            pred.value == gold.value, pred.value == idx, gold.value == idx
        valid_mask = (gold.value != self._unk_idx)
        prec_mask = prec_mask & seq_mask & valid_mask
        rec_mask = rec_mask & seq_mask & valid_mask
        return {
            f"prec-turn-{spkr}": ((crt & prec_mask).sum().float() /
                                  prec_mask.sum()),
            f"prec{spkr}": (crt | ~prec_mask).all(1).float().mean(),
            f"rec-turn-{spkr}": (crt & rec_mask).sum().float() / rec_mask.sum(),
            f"rec{spkr}": (crt | ~rec_mask).all(1).float().mean()
        }

    def compute_accuracy(self, pred: Stacked1DTensor, gold: Stacked1DTensor
                         ) -> utils.TensorMap:
        crt = pred.value == gold.value
        mask = utils.mask(pred.lens, crt.size(1))
        mask = mask & (gold.value != self._unk_idx)
        ret = {
            "acc-turn": (crt & mask).sum().float() / mask.sum(),
            "acc": (crt | ~mask).all(1).float().mean()
        }
        if self.detail:
            for idx, spkr in self.vocab.i2f.items():
                if spkr == "<unk>":
                    continue
                ret.update(self.compute_rec_prec_cls(idx, pred, gold))
        return ret

    def reset(self):
        self._pred, self._gold = list(), list()

    def update(self, batch: BatchData, pred: BatchData, outputs
               ) -> Optional[TensorMap]:
        self._pred.extend(list(pred.speaker.to(torch.device("cpu"))))
        self._gold.extend(list(batch.speaker.to(torch.device("cpu"))))
        if not self.return_update:
            return
        return {f"spkr-{k}": v for k, v in
                self.compute_accuracy(pred.speaker, batch.speaker).items()}

    def get(self) -> TensorMap:
        return {f"spkr-{k}": v for k, v in
                self.compute_accuracy(utils.pad_stack(self._pred),
                                      utils.pad_stack(self._gold)).items()}
