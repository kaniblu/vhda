__all__ = ["LogGenerator"]

import random
import logging
from dataclasses import dataclass, field
from typing import Sequence, Optional, Union, Set

import tqdm

import utils
from utils import TensorMap
from utils import StatsFormatter
from utils import DialogFormatter
from .generator import Sample
from .generator import Generator


@dataclass
class LogGenerator(Generator):
    progress_stat: Optional[str] = None
    display_stats: Optional[Union[Set[str], Sequence[str]]] = None
    stats_formatter: StatsFormatter = field(default_factory=StatsFormatter)
    dialog_formatter: DialogFormatter = field(default_factory=DialogFormatter)
    run_end_report: bool = True
    report_every: Optional[int] = None
    _tqdm: tqdm.tqdm = utils.private_field(default=None)
    _last_report: int = utils.private_field(default=0)
    _logger: logging.Logger = utils.private_field(default=None)

    def __post_init__(self):
        super(LogGenerator, self).__post_init__()
        self._logger = logging.getLogger(self.__class__.__name__)

    def on_run_started(self):
        super().on_run_started()
        self._tqdm = tqdm.tqdm(
            total=self._num_instances,
            dynamic_ncols=True,
            desc=self.__class__.__name__
        )

    def sample(self, samples: Sequence[Sample]):
        sample = random.choice(samples)
        msg = self.dialog_formatter.format(sample.input, sample.output)
        self._logger.info(f"sample input:\n{msg}")

    def log_stats(self, tag: str, stats: utils.TensorMap):
        if self.display_stats is not None:
            display_stats = {k: v for k, v in stats.items() if
                             k in self.display_stats}
        else:
            display_stats = stats
        self._logger.info(f"global step {self.global_step:,d} {tag}:\n"
                          f"{self.stats_formatter.format(display_stats)}")

    def on_batch_ended(self, samples: Sequence[Sample]) -> TensorMap:
        stats = dict(super().on_batch_ended(samples))
        self._tqdm.update(len(samples))
        if self.progress_stat is not None and self.progress_stat in stats:
            self._tqdm.set_postfix(
                {self.progress_stat: stats[self.progress_stat].item()})
        if (self.report_every is not None and
                (self._last_report is None or
                 (self.global_step - self._last_report) >= self.report_every)):
            self.sample(samples)
            self.log_stats("stats", stats)
            self._last_report = self.global_step
        return stats

    def on_run_ended(self, samples: Sequence[Sample], stats: TensorMap):
        samples, stats = super().on_run_ended(samples, stats)
        if self.run_end_report:
            self.log_stats("final-summary", stats)
        self._tqdm.close()
        return samples, stats
