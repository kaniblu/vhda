__all__ = ["LogInferencer"]

import random
from dataclasses import dataclass, field
from typing import Optional, Set, Union, Sequence, ClassVar

import tqdm
import torch.utils.tensorboard
import torch.utils.data as td

import utils
from datasets import Dialog
from datasets import BatchData
from utils import StatsFormatter
from utils import DialogFormatter
from utils import DialogTableFormatter
from .inferencer import Inferencer


@dataclass
class LogInferencer(Inferencer):
    progress_stat: Optional[str] = None
    display_stats: Optional[Union[Set[str], Sequence[str]]] = None
    writer: Optional[torch.utils.tensorboard.SummaryWriter] = None
    stats_formatter: StatsFormatter = field(default_factory=StatsFormatter)
    dialog_formatter: DialogFormatter = \
        field(default_factory=DialogTableFormatter)
    report_every: Optional[int] = None
    run_end_report: bool = True
    _tqdm: tqdm.tqdm = utils.private_field(default=None)
    _last_report: int = utils.private_field(default=0)
    _dialog_md_formatter: ClassVar[utils.DialogMarkdownFormatter] = \
        utils.DialogMarkdownFormatter()

    def __post_init__(self):
        super().__post_init__()
        if self.display_stats is not None:
            self.display_stats = set(self.display_stats)

    def on_run_started(self, dataloader: td.DataLoader) -> td.DataLoader:
        dataloader = super(LogInferencer, self).on_run_started(dataloader)
        self._tqdm = tqdm.tqdm(
            total=len(dataloader.dataset),
            dynamic_ncols=True,
            desc=self.__class__.__name__,
        )
        return dataloader

    def on_batch_started(self, batch: BatchData) -> BatchData:
        batch = super().on_batch_started(batch)
        self._tqdm.update(batch.batch_size)
        return batch

    def on_batch_ended(self, batch: BatchData, pred: BatchData, outputs
                       ) -> utils.TensorMap:
        stats = super().on_batch_ended(batch, pred, outputs)
        if self.progress_stat is not None and self.progress_stat in stats:
            self._tqdm.set_postfix(
                {self.progress_stat: stats[self.progress_stat].item()})
        if self.report_every is not None and \
                (self.global_step - self._last_report) >= self.report_every:
            idx = random.randint(0, batch.batch_size - 1)
            batch_sample = batch.raw[idx]
            pred_sample = self.processor.lexicalize_global(pred[idx])
            self.log_dialog("input-sample", batch_sample)
            self.log_dialog("pred-sample", pred_sample)
            self.log_stats("summary", stats)
            self._last_report = self.global_step
        return stats

    def log_dialog(self, tag: str, dialog: Dialog):
        if self.writer is not None:
            self.writer.add_text(
                tag=tag,
                text_string=self._dialog_md_formatter.format(dialog),
                global_step=self.global_step
            )
        self._logger.info(
            f"global step {self.global_step:,d} {tag}:\n"
            f"{self.dialog_formatter.format(dialog)}"
        )

    def log_stats(self, tag: str, stats: utils.TensorMap,
                  prefix=None, postfix=None):
        def wrap_key(key):
            if prefix is not None:
                key = f"{prefix}-{key}"
            if postfix is not None:
                key = f"{key}-{postfix}"
            return key

        if self.writer is not None:
            for k, v in stats.items():
                self.writer.add_scalar(wrap_key(k), v.item(), self.global_step)
        if self.display_stats is not None:
            display_stats = {k: v for k, v in stats.items() if
                             k in self.display_stats}
        else:
            display_stats = stats
        display_stats = {wrap_key(k): v for k, v in display_stats.items()}
        self._logger.info(f"global step {self.global_step:,d} {tag}:\n"
                          f"{self.stats_formatter.format(display_stats)}")

    def on_run_ended(self, stats: utils.TensorMap) -> utils.TensorMap:
        stats = super().on_run_ended(stats)
        if self.run_end_report:
            self.log_stats("run-end-summary", stats, postfix="run")
        self._tqdm.close()
        return stats
