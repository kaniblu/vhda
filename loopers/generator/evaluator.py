__all__ = ["EvaluatingGenerator"]

from dataclasses import dataclass
from typing import Sequence, Tuple

from utils import TensorMap
from evaluators import DialogEvaluator
from .generator import Sample
from .generator import Generator


@dataclass
class EvaluatingGenerator(Generator):
    evaluators: Sequence[DialogEvaluator] = tuple()

    def on_run_started(self):
        super().on_run_started()
        for evaluator in self.evaluators:
            evaluator.reset()

    def on_batch_ended(self, samples: Sequence[Sample]) -> TensorMap:
        stats = dict(super().on_batch_ended(samples))
        for evaluator in self.evaluators:
            stats.update(evaluator.update(samples) or dict())
        return stats

    def on_run_ended(self, samples: Sequence[Sample], stats: TensorMap
                     ) -> Tuple[Sequence[Sample], TensorMap]:
        samples, stats = super().on_run_ended(samples, stats)
        stats = dict(stats)
        for evaluator in self.evaluators:
            stats.update(evaluator.get() or dict())
        return samples, stats
