__all__ = ["Inferencer"]

import logging
import collections
from dataclasses import dataclass, field

import torch
import torch.utils.data as td

import utils
from datasets import DialogProcessor
from datasets import BatchData
from models import AbstractTDA


@dataclass
class Inferencer:
    model: AbstractTDA
    processor: DialogProcessor
    device: torch.device = torch.device("cpu")
    global_step: int = 0
    asv_tensor: utils.Stacked1DTensor = None
    _logger: logging.Logger = utils.private_field(default=None)

    def __post_init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        if self.asv_tensor is None:
            self.asv_tensor = self.processor.tensorize_state_vocab("goal_state")
        self.asv_tensor = self.asv_tensor.to(self.device)

    def on_run_started(self, dataloader: td.DataLoader) -> td.DataLoader:
        return dataloader

    def on_run_ended(self, stats: utils.TensorMap) -> utils.TensorMap:
        return stats

    def on_batch_started(self, batch: BatchData) -> BatchData:
        return batch

    def on_batch_ended(self, batch: BatchData, pred: BatchData, outputs
                       ) -> utils.TensorMap:
        return {}

    def model_kwargs(self) -> dict:
        return {}

    @staticmethod
    def predict(batch: BatchData, outputs) -> BatchData:
        def predict_state(state_logit, conv_lens):
            batch_size, max_conv_len, num_asv = state_logit.size()
            mask = ((state_logit == float("-inf")) |
                    (state_logit == float("inf")))
            pred = (torch.sigmoid(state_logit.masked_fill(mask, 0))
                    .masked_fill(mask, 0)) > 0.5
            pred = utils.to_sparse(pred.view(-1, num_asv))
            return utils.DoublyStacked1DTensor(
                value=pred.value.view(batch_size, max_conv_len, -1),
                lens=conv_lens,
                lens1=pred.lens.view(batch_size, max_conv_len)
            )

        logit, post, prior = outputs
        return BatchData(
            sent=utils.DoublyStacked1DTensor(
                value=torch.cat([batch.sent.value[..., :1],
                                 logit["sent"].max(-1)[1][..., :-1]], 2),
                lens=batch.sent.lens,
                lens1=batch.sent.lens1
            ),
            speaker=utils.Stacked1DTensor(
                value=logit["speaker"].max(-1)[1],
                lens=batch.conv_lens
            ),
            goal=predict_state(logit["goal"], batch.conv_lens),
            state=predict_state(logit["state"], batch.conv_lens),
            raw=batch.raw
        )

    def prepare_batch(self, batch: BatchData) -> dict:
        return {
            "conv_lens": batch.conv_lens,
            "sent": batch.sent.value,
            "sent_lens": batch.sent.lens1,
            "speaker": batch.speaker.value,
            "goal": batch.goal.value,
            "goal_lens": batch.goal.lens1,
            "state": batch.state.value,
            "state_lens": batch.state.lens1,
            "asv": self.asv_tensor.value,
            "asv_lens": self.asv_tensor.lens
        }

    def __call__(self, dataloader):
        dataloader = self.on_run_started(dataloader)
        cum_stats = collections.defaultdict(float)
        total_steps = 0
        for batch in dataloader:
            batch = batch.to(self.device)
            total_steps += batch.batch_size
            self.global_step += batch.batch_size
            batch = self.on_batch_started(batch)
            self.model.inference()
            outputs = self.model(self.prepare_batch(batch),
                                 **self.model_kwargs())
            pred = self.predict(batch, outputs)
            stats = self.on_batch_ended(batch, pred, outputs)
            for k, v in stats.items():
                cum_stats[k] += v.detach() * batch.batch_size
        cum_stats = {k: v / total_steps for k, v in cum_stats.items()}
        return self.on_run_ended(cum_stats)
