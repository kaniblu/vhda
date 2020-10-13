__all__ = ["DialogDataset"]

from dataclasses import dataclass
from typing import Sequence

import torch.utils.data as td

from .processor import DialogProcessor
from .common import Dialog
from .common import DialogState


@dataclass
class DialogDataset(td.Dataset):
    data: Sequence[Dialog]
    processor: DialogProcessor
    state_index: str = "global"

    def __post_init__(self):
        if not self.processor.is_initialized:
            raise ValueError(f"dialog processor must be initialized")
        if self.state_index not in {"global", "local"}:
            raise ValueError(f"unsupported state index mode: "
                             f"{self.state_index}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.state_index == "global":
            turn_tensorizer = self.processor.tensorize_turn_global_state
        elif self.state_index == "local":
            turn_tensorizer = self.processor.tensorize_turn
        else:
            raise ValueError(f"unsupported method: {self.state_index}")
        data = self.processor.tensorize(
            self.data[item],
            turn_tensorizer=turn_tensorizer
        )
        data["raw"] = self.data[item]
        return data

    def get_ontology(self, speaker=None) -> DialogState:
        state = DialogState()
        for dialog in self.data:
            if speaker is None:
                turns = dialog
            else:
                turns = dialog.filter(speaker)
            for turn in turns:
                state = state | turn.state
        return state
