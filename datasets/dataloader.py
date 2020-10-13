__all__ = ["TurnData", "ConvData", "BatchData",
           "PaddingCollator", "create_dataloader"]

from dataclasses import dataclass
from typing import Sequence, Optional

import torch
import torch.utils.data as td

import utils
from utils import Stacked1DTensor
from utils import DoublyStacked1DTensor
from .common import Dialog
from .collator import PaddingCollator


@dataclass
class TurnData:
    sent: torch.Tensor  # [sent_len] LongTensor
    speaker: torch.Tensor  # [] LongTensor
    goal: torch.Tensor  # [goal_len] LongTensor
    state: torch.Tensor  # [state_len] LongTensor


@dataclass
class ConvData:
    sent: Stacked1DTensor
    speaker: torch.Tensor  # [num_turns] LongTensor
    goal: Stacked1DTensor
    state: Stacked1DTensor
    raw: Optional[Dialog] = None

    def __post_init__(self):
        # sanity check
        # num turns check
        assert all(t.size(0) == self.sent.size(0) for t in self.tensors), \
            f"conversation length mismatch"

    @property
    def max_sent_len(self):
        return self.sent.lens.max().item()

    @property
    def max_goal_len(self):
        return self.goal.lens.max().item()

    @property
    def max_state_len(self):
        return self.state.lens.max().item()

    @property
    def length(self):
        return self.sent.size(0)

    @property
    def conv_len(self):
        return self.length

    def __len__(self):
        return self.length

    @property
    def tensors(self):
        return self.sent, self.speaker, self.goal, self.state

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        if isinstance(item, (slice, Sequence)):
            return ConvData(*(t[item] for t in self.tensors))
        elif isinstance(item, int):
            if not (0 <= item < self.length):
                raise IndexError(f"not a valid turn index: "
                                 f"!(0 <= {item} < {self.length}")
            return TurnData(*(t[item] for t in self.tensors))
        else:
            raise TypeError(f"unsupported index type: {type(item)}")


@dataclass
class BatchData:
    sent: DoublyStacked1DTensor
    speaker: Stacked1DTensor
    goal: DoublyStacked1DTensor
    state: DoublyStacked1DTensor
    raw: Optional[Sequence[Dialog]] = None

    def __post_init__(self):
        # sanity check
        # batch size and conv_lens check
        assert all(utils.compare_tensors(t.lens, self.sent.lens)
                   for t in (self.sent, self.speaker, self.goal, self.state)), \
            f"not all tensors have the same batch size and conv. length"
        if self.raw is not None:
            assert len(self.sent) == len(self.raw)

    @property
    def tensors(self):
        return self.sent, self.speaker, self.goal, self.state

    def to_dict(self):
        return dict(
            sent=self.sent,
            speaker=self.speaker,
            goal=self.goal,
            state=self.state,
            raw=self.raw
        )

    @property
    def conv_lens(self):
        return self.sent.lens

    @property
    def batch_size(self):
        return self.sent.size(0)

    def __len__(self):
        return self.batch_size

    @property
    def max_conv_len(self):
        return self.conv_lens.max().item()

    @property
    def max_sent_len(self):
        return self.sent.lens1.max().item()

    @property
    def max_state_len(self):
        return self.state.lens1.max().item()

    @property
    def max_goal_len(self):
        return self.goal.lens1.max().item()

    def to(self, device: torch.device):
        return BatchData(
            sent=self.sent.to(device),
            speaker=self.speaker.to(device),
            state=self.state.to(device),
            goal=self.goal.to(device),
            raw=self.raw
        )

    def __iter__(self):
        for i in range(self.batch_size):
            yield self[i]

    def clip(self, n):
        """Clips conversations in either direction depending on the sign of n"""
        if n == 0:
            return self
        if abs(n) >= self.max_conv_len:
            raise ValueError(f"clip size must be within the max conv len: "
                             f"{n} >= {self.max_conv_len}")
        max_conv_len = self.conv_lens.max().item()
        if n > 0:
            return BatchData(*(t.narrow(1, n, max_conv_len - n)
                               for t in self.tensors), raw=self.raw)
        else:
            end = max_conv_len + n
            return BatchData(*(t.narrow(1, 0, end) for t in self.tensors),
                             raw=self.raw)

    def __getitem__(self, item):
        if isinstance(item, str):
            raise TypeError(f"string type index not supported: '{item}'")
        if isinstance(item, (slice, Sequence, torch.Tensor)):
            return BatchData(*(t[item] for t in self.tensors), raw=self.raw)
        elif isinstance(item, int):
            if not (0 <= item < self.batch_size):
                raise IndexError(f"not a valid batch index: "
                                 f"!(0 <= {item} < {self.batch_size}")
            return ConvData(*(t[item] for t in self.tensors),
                            raw=self.raw[item] if self.raw else None)
        else:
            raise TypeError(f"unsupported index type: {type(item)}")


def create_dataloader(dataset, **dl_kwargs) -> td.DataLoader:
    collator = PaddingCollator(frozenset(("sent", "speaker", "goal", "states")))

    def collate(batch):
        return BatchData(**collator(batch))

    return td.DataLoader(
        dataset=dataset,
        collate_fn=collate,
        **dl_kwargs
    )
