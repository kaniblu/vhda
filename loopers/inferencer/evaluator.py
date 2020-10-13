__all__ = ["EvaluatingInferencer"]

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.utils.data as td

import utils
from datasets import BatchData
from .inferencer import Inferencer
from evaluators import FinegrainedEvaluator


@dataclass
class EvaluatingInferencer(Inferencer):
    evaluators: Sequence[FinegrainedEvaluator] = tuple()
    _requires_lexical_form: bool = utils.private_field(default=False)

    def __post_init__(self):
        super().__post_init__()
        self._requires_lexical_form = any(e.requires_lexical_form
                                          for e in self.evaluators)

    def on_run_started(self, dataloader: td.DataLoader) -> td.DataLoader:
        dataloader = super().on_run_started(dataloader)
        for evaluator in self.evaluators:
            evaluator.reset()
        return dataloader

    def on_batch_ended(self, batch: BatchData, pred: BatchData, outputs
                       ) -> utils.TensorMap:
        stats = dict(super().on_batch_ended(batch, pred, outputs))
        batch_lex, pred_lex = None, None
        if self._requires_lexical_form:
            batch_lex = list(map(self.processor.lexicalize_global, batch))
            pred_lex = list(map(self.processor.lexicalize_global, pred))
        with torch.no_grad():
            for evaluator in self.evaluators:
                if evaluator.requires_lexical_form:
                    eval_stats = evaluator.update(batch_lex, pred_lex, outputs)
                else:
                    eval_stats = evaluator.update(batch, pred, outputs)
                stats.update(eval_stats or dict())
        return stats

    def on_run_ended(self, stats: utils.TensorMap) -> utils.TensorMap:
        stats = dict(super().on_run_ended(stats))
        with torch.no_grad():
            for evaluator in self.evaluators:
                stats.update(evaluator.get() or dict())
        return stats
