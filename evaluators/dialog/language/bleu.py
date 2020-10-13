__all__ = ["BLEUEvaluator"]

import bisect
import logging
import collections
from dataclasses import dataclass
from typing import Sequence, Optional, List

import torch
import numpy as np
import nltk.translate.bleu_score as bl

import utils
from datasets import VocabSet
from utils import TensorMap
from ...evaluator import DialogEvaluator


@dataclass
class BLEUEvaluator(DialogEvaluator):
    vocabs: VocabSet
    _ref: List[Sequence[Sequence[str]]] = \
        utils.private_field(default_factory=list)
    _hyp: List[Sequence[Sequence[str]]] = \
        utils.private_field(default_factory=list)
    _hyp_spkr: dict = utils.private_field(default_factory=dict)
    _ref_spkr: dict = utils.private_field(default_factory=dict)
    _logger: logging.Logger = utils.private_field(default=None)

    def __post_init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def reset(self):
        self._ref.clear()
        self._hyp.clear()
        self._ref_spkr.clear()
        self._hyp_spkr.clear()

    @staticmethod
    def closest_rlen(ref_lens, hyp_len):
        idx = bisect.bisect_left(ref_lens, hyp_len)
        if idx == 0:
            return ref_lens[0]
        elif idx == len(ref_lens):
            return ref_lens[idx - 1]
        else:
            len1, len2 = ref_lens[idx - 1], ref_lens[idx]
            if hyp_len - len1 <= len2 - hyp_len:
                return len1
            else:
                return len2

    def try_compute(self, *args, **kwargs) -> float:
        try:
            return bl.sentence_bleu(*args, **kwargs)
        except Exception as e:
            self._logger.debug(f"error in bleu: {e}")
            return 0.0

    def compute_bleu(self, ref: Sequence[Sequence[str]],
                     hyp: Sequence[Sequence[str]]):
        r_lens = list(sorted(set(map(len, ref))))
        chencherry = bl.SmoothingFunction()
        methods = (
            ("smooth0", chencherry.method0),
            ("smooth1", chencherry.method1),
            ("smooth2", chencherry.method2),
            ("smooth3", chencherry.method3),
            ("smooth4", chencherry.method4),
            ("smooth5", chencherry.method5),
            ("smooth6", chencherry.method6),
            ("smooth7", chencherry.method7)
        )
        if not hyp:
            return {name: 0.0 for name, _ in methods}
        stats = {
            name: np.mean([
                (self.try_compute(ref, h, smoothing_function=method) *
                 bl.brevity_penalty(self.closest_rlen(r_lens, len(h)), len(h)))
                for h in hyp
            ]) for name, method in methods
        }
        return stats

    def compute_bleu_corpus(self, refs: Sequence[Sequence[Sequence[str]]],
                            hyps: Sequence[Sequence[Sequence[str]]]):
        assert len(refs) == len(hyps)
        stats = collections.defaultdict(float)
        for ref, hyp in zip(refs, hyps):
            for k, v in self.compute_bleu(ref, hyp).items():
                stats[k] += v
        return {k: v / len(refs) for k, v in stats.items()}

    def update(self, samples: Sequence) -> Optional[TensorMap]:
        ref = [[turn.text.split() for turn in sample.input.turns]
               for sample in samples]
        hyp = [[turn.text.split() for turn in sample.output.turns]
               for sample in samples]
        self._ref.extend(ref)
        self._hyp.extend(hyp)
        for spkr in self.vocabs.speaker.f2i:
            if spkr == "<unk>":
                continue
            if spkr not in self._ref_spkr:
                self._ref_spkr[spkr] = list()
                self._hyp_spkr[spkr] = list()
            ref = [[turn.text.split() for turn in sample.input.turns
                    if turn.speaker == spkr]
                   for sample in samples]
            hyp = [[turn.text.split() for turn in sample.output.turns
                    if turn.speaker == spkr]
                   for sample in samples]
            self._ref_spkr[spkr].extend(ref)
            self._hyp_spkr[spkr].extend(hyp)

    def get(self) -> TensorMap:
        stats = {f"bleu-{k}": torch.tensor(v) for k, v in
                 self.compute_bleu_corpus(self._ref, self._hyp).items()}
        for spkr in self.vocabs.speaker.f2i:
            if spkr == "<unk>":
                continue
            stats.update({
                f"bleu-{k}-{spkr}": torch.tensor(v) for k, v in
                self.compute_bleu_corpus(self._ref_spkr[spkr],
                                         self._hyp_spkr[spkr]).items()
            })
        return stats
