__all__ = ["Evaluator", "FinegrainedEvaluator", "DialogEvaluator"]

from typing import Optional, Union, Sequence

from datasets import Dialog
from datasets import BatchData
from utils import TensorMap


class Evaluator:

    def reset(self):
        """Reset accumulation"""
        raise NotImplementedError

    def get(self) -> Optional[TensorMap]:
        """Evaluate results using accumulated data"""
        raise NotImplementedError


class FinegrainedEvaluator(Evaluator):

    def reset(self):
        """Reset accumulation"""
        raise NotImplementedError

    def get(self) -> Optional[TensorMap]:
        """Evaluate results using accumulated data"""
        raise NotImplementedError

    @property
    def requires_lexical_form(self):
        return False

    def update(self, batch: Union[Sequence[Dialog], BatchData],
               pred: Union[Sequence[Dialog], BatchData], outputs
               ) -> Optional[TensorMap]:
        """Updated with some mini-batch"""
        raise NotImplementedError


class DialogEvaluator(Evaluator):

    def reset(self):
        """Reset accumulation"""
        raise NotImplementedError

    def get(self) -> Optional[TensorMap]:
        """Evaluate results using accumulated data"""
        raise NotImplementedError

    def update(self, samples: Sequence) -> Optional[TensorMap]:
        """Updated with some mini-batch.

        Arguments:
            samples (Sequence[Sample]): Sample has the following signature:

                @dataclass
                class Sample:
                    input: Dialog
                    output: Dialog
                    log_prob: float

        """
        raise NotImplementedError
