__all__ = ["Sample", "Generator"]

import random
import itertools
import collections
from dataclasses import dataclass
from typing import Tuple, Sequence, Mapping, Optional

import torch

import utils
from utils import TensorMap
from datasets import Dialog
from datasets import DialogProcessor
from datasets import create_dataloader
from datasets import BatchData
from datasets import DialogDataset
from models import AbstractTDA


@dataclass
class Sample:
    input: Dialog
    output: Dialog
    log_prob: float


@dataclass
class Generator:
    model: AbstractTDA
    processor: DialogProcessor
    batch_size: int = 32
    device: torch.device = torch.device("cpu")
    global_step: int = 0
    asv_tensor: utils.Stacked1DTensor = None
    _num_instances: int = utils.private_field(default=None)

    def __post_init__(self):
        if self.asv_tensor is None:
            self.asv_tensor = self.processor.tensorize_state_vocab("goal_state")
        self.asv_tensor = self.asv_tensor.to(self.device)

    def on_run_started(self):
        return

    def on_run_ended(self, samples: Sequence[Sample], stats: TensorMap
                     ) -> Tuple[Sequence[Sample], TensorMap]:
        return samples, stats

    def on_batch_started(self, batch: BatchData) -> BatchData:
        return batch

    def validate_sample(self, sample: Sample):
        return True

    def on_batch_ended(self, samples: Sequence[Sample]) -> TensorMap:
        return dict()

    def generate_kwargs(self) -> dict:
        return dict()

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

    def __call__(self, data: Optional[Sequence[Dialog]] = None,
                 num_instances: Optional[int] = None
                 ) -> Tuple[Sequence[Sample], TensorMap]:
        if data is None and num_instances is None:
            raise ValueError(f"must provide a data source or "
                             f"number of instances.")
        dataloader = None
        if data is not None:
            dataloader = create_dataloader(
                dataset=DialogDataset(
                    data=data,
                    processor=self.processor
                ),
                batch_size=self.batch_size,
                drop_last=False,
                shuffle=False
            )
            if num_instances is None:
                num_instances = len(data)
        self._num_instances = num_instances
        self.on_run_started()
        dataloader = (itertools.repeat(None) if dataloader is None
                      else itertools.cycle(dataloader))
        cum_stats = collections.defaultdict(float)
        samples = []
        for batch in dataloader:
            self.model.eval()
            if batch is None:
                batch_size = min(self.batch_size, num_instances - len(samples))
                self.model.genconv_prior()
                with torch.no_grad():
                    pred, info = self.model(
                        torch.tensor(batch_size).to(self.device),
                        **self.generate_kwargs()
                    )
            else:
                batch = batch.to(self.device)
                batch_size = batch.batch_size
                self.global_step += batch_size
                batch = self.on_batch_started(batch)
                self.model.genconv_post()
                with torch.no_grad():
                    pred, info = self.model(self.prepare_batch(batch),
                                            **self.generate_kwargs())
            batch_samples = list(filter(self.validate_sample, (
                Sample(*args) for args in
                zip(map(self.processor.lexicalize_global, batch),
                    map(self.processor.lexicalize_global, pred),
                    info["logprob"])
            )))
            num_res = max(0, len(samples) + len(batch_samples) - num_instances)
            if num_res > 0:
                batch_samples = random.sample(batch_samples,
                                              num_instances - len(samples))
            batch_size = len(batch_samples)
            self.global_step += batch_size
            stats = self.on_batch_ended(batch_samples)
            samples.extend(batch_samples)
            for k, v in stats.items():
                cum_stats[k] += v * batch_size
            if len(samples) >= num_instances:
                break
        assert len(samples) == num_instances
        cum_stats = {k: v / len(samples) for k, v in cum_stats.items()}
        return self.on_run_ended(samples, cum_stats)
