__all__ = ["ValidatingGenerator"]

from dataclasses import dataclass
from typing import Callable

from .generator import Sample
from .generator import Generator


@dataclass
class ValidatingGenerator(Generator):
    validator: Callable[[Sample], bool] = None

    def __post_init__(self):
        super().__post_init__()
        assert self.validator is not None

    def validate_sample(self, sample: Sample):
        return super().validate_sample(sample) and self.validator(sample)
