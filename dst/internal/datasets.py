__all__ = ["DSTTurn", "DSTBatchData", "DSTTestData", "DSTTestBatchData",
           "DSTDialogProcessor", "DSTDialogDataset", "create_dataloader"]

import bisect
from dataclasses import dataclass
from typing import Sequence, Mapping

import nltk
import numpy as np
import torch
import torch.utils.data as td

import utils
from utils import Stacked1DTensor
from utils import DoublyStacked1DTensor
from datasets import ActSlotValue
from datasets import DSTUserTurn
from datasets import DSTWizardTurn
from datasets import DSTDialog
from datasets import DialogProcessor
from datasets import PaddingCollator


@dataclass(frozen=True)
class DSTTurn:
    wizard: DSTWizardTurn
    user: DSTUserTurn


@dataclass
class DSTData:
    sent: torch.Tensor
    system_acts: torch.Tensor
    belief_state: torch.Tensor
    slot: torch.Tensor
    raw: DSTTurn

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"unsupported type: {type(other)}")
        return (utils.compare_tensors(self.sent, other.sent) and
                utils.compare_tensors(self.system_acts, other.system_acts) and
                utils.compare_tensors(self.belief_state, other.belief_state) and
                utils.compare_tensors(self.slot, other.slot) and
                self.raw == other.raw)

    def to_dict(self):
        return {
            "sent": self.sent,
            "system_acts": self.system_acts,
            "belief_state": self.belief_state,
            "slot": self.slot,
            "raw": self.raw
        }


@dataclass
class DSTTestData(DSTData):
    asr: Stacked1DTensor
    asr_score: torch.Tensor

    def __eq__(self, other):
        return (super().__eq__(other) and
                self.asr == other.asr and
                utils.compare_tensors(self.asr_score, other.asr_score))

    def to_dict(self):
        return utils.merge_dict(super().to_dict(), {
            "asr": self.asr,
            "asr_score": self.asr_score
        })


@dataclass(frozen=True)
class DSTBatchData:
    sent: Stacked1DTensor
    system_acts: Stacked1DTensor
    belief_state: Stacked1DTensor
    slot: Stacked1DTensor
    raw: Sequence[DSTTurn]

    @property
    def batch_size(self):
        return self.sent.size(0)

    @staticmethod
    def from_dict(data: dict):
        return DSTBatchData(**data)

    def to_dict(self):
        return {
            "sent": self.sent,
            "system_acts": self.system_acts,
            "belief_state": self.belief_state,
            "slot": self.slot,
            "raw": self.raw
        }

    def to(self, device: torch.device):
        return DSTBatchData(**{k: v.to(device) if k != "raw" else v
                               for k, v in self.to_dict().items()})

    def narrow(self, start, length):
        end = start + length
        return DSTBatchData(
            sent=self.sent.narrow(0, start, length),
            system_acts=self.system_acts.narrow(0, start, length),
            belief_state=self.belief_state.narrow(0, start, length),
            slot=self.slot.narrow(0, start, length),
            raw=self.raw[start:end]
        )

    def __getitem__(self, item):
        if isinstance(item, str):
            raise TypeError(f"string type index not supported: '{item}'")
        if isinstance(item, (slice, Sequence, torch.Tensor)):
            return DSTBatchData(
                sent=self.sent[item],
                system_acts=self.system_acts[item],
                belief_state=self.belief_state[item],
                slot=self.slot[item],
                raw=[
                    self.raw[idx] for idx in
                    (item.tolist() if isinstance(item, torch.Tensor) else item)
                ]
            )
        elif isinstance(item, int):
            return DSTData(
                sent=self.sent[item],
                system_acts=self.system_acts[item],
                belief_state=self.belief_state[item],
                slot=self.slot[item],
                raw=self.raw[item]
            )
        else:
            raise TypeError(f"unsupported index type: {type(item)}")

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


@dataclass(frozen=True)
class DSTTestBatchData(DSTBatchData):
    asr: DoublyStacked1DTensor
    asr_score: Stacked1DTensor

    @staticmethod
    def from_dict(data: dict):
        return DSTTestBatchData(**data)

    def to_dict(self):
        return utils.merge_dict(super().to_dict(), {
            "asr": self.asr,
            "asr_score": self.asr_score
        })

    def to(self, device: torch.device):
        return DSTTestBatchData(**{k: v.to(device) if k != "raw" else v
                                   for k, v in self.to_dict().items()})

    def narrow(self, start, length):
        end = start + length
        return DSTTestBatchData(
            sent=self.sent.narrow(0, start, length),
            system_acts=self.system_acts.narrow(0, start, length),
            belief_state=self.belief_state.narrow(0, start, length),
            slot=self.slot.narrow(0, start, length),
            raw=self.raw[start:end],
            asr=self.asr.narrow(0, start, length),
            asr_score=self.asr_score.narrow(0, start, length)
        )

    def __getitem__(self, item):
        if isinstance(item, str):
            raise TypeError(f"string type index not supported: '{item}'")
        if isinstance(item, (slice, Sequence, torch.Tensor)):
            return DSTTestBatchData(
                sent=self.sent[item],
                system_acts=self.system_acts[item],
                belief_state=self.belief_state[item],
                slot=self.slot[item],
                raw=self.raw[item],
                asr=self.asr[item],
                asr_score=self.asr_score[item]
            )
        elif isinstance(item, int):
            return DSTTestData(
                sent=self.sent[item],
                system_acts=self.system_acts[item],
                belief_state=self.belief_state[item],
                slot=self.slot[item],
                raw=self.raw[item],
                asr=self.asr[item],
                asr_score=self.asr_score[item]
            )
        else:
            raise TypeError(f"unsupported index type: {type(item)}")


@dataclass
class DSTDialogProcessor(DialogProcessor):

    def tensorize_turn_label_asv(self, asv: ActSlotValue):
        if asv == self.asv_pad:
            return self.tensorize_processed_tokens(("<pad>",))
        if asv.act == "inform":
            slot, value = asv.slot, asv.value,
        elif asv.act == "request":
            slot, value = "request", asv.value
        else:
            raise RuntimeError(f"unexpected act: {asv.act}")
        tokens = (list(nltk.casual_tokenize(slot)) + ["="] +
                  list(nltk.casual_tokenize(value)) + ["<eos>"])
        return self.tensorize_processed_tokens(tokens)

    def tensorize_state_dict(self, tensor=None, speaker=None, tensorizer=None):
        if tensor is None:
            tensor = self.tensorize_state_vocab("state", speaker, tensorizer)
        ont = dict()
        if speaker is None:
            vocab = self.vocabs.state.asv
        else:
            vocab = self.vocabs.speaker_state[speaker].asv
        for idx, asv in vocab.i2f.items():
            if asv == self.asv_pad:
                continue
            key = (asv.act, asv.slot)
            if key not in ont:
                ont[key] = []
            ont[key].append((torch.tensor(idx), tensor[idx]))
        return {k: (torch.stack([x[0] for x in v]),
                    utils.pad_stack([x[1] for x in v]))
                for k, v in ont.items()}

    def tensorize_dst_turn(self, turn: DSTTurn):
        system_acts, turn_state = turn.wizard.state, turn.user.state
        return {
            "sent": self.tensorize_sent(turn.user.text),
            "system_acts": self.tensorize_state(system_acts, True, "wizard"),
            "belief_state": self.tensorize_state(turn_state, speaker="user"),
            "slot": self.tensorize_slot(turn_state, speaker="user"),
            "raw": turn
        }

    def tensorize_asr(self, asr: Mapping[str, float]):
        sents, scores = zip(*asr.items())
        return (utils.pad_stack(list(map(self.tensorize_sent, sents))),
                torch.tensor(scores))

    def tensorize_dst_test_turn(self, turn: DSTTurn):
        data = self.tensorize_dst_turn(turn)
        data["asr"], data["asr_score"] = self.tensorize_asr(turn.user.asr)
        return data


@dataclass
class DSTDialogDataset:
    dialogs: Sequence[DSTDialog]
    processor: DSTDialogProcessor
    _num_turns: Sequence[int] = utils.private_field()
    _cum_turns: Sequence[int] = utils.private_field()

    def __post_init__(self):
        self._num_turns = [len(dialog.dst_turns) for dialog in self.dialogs]
        self._cum_turns = [0] + np.cumsum(self._num_turns).tolist()

    def __len__(self):
        return self._cum_turns[-1]

    def __getitem__(self, item):
        idx = bisect.bisect_right(self._cum_turns, item) - 1
        if idx >= len(self.dialogs):
            raise ValueError(f"must be less than dataset size: "
                             f"{item} >= {len(self)}")
        turn_idx = item - self._cum_turns[idx]
        dialog = self.dialogs[idx]
        return self.processor.tensorize_dst_turn(dialog.dst_turns[turn_idx])


def create_dataloader(dataset, **dl_kwargs) -> td.DataLoader:
    collator = PaddingCollator(frozenset(("sent", "system_acts",
                                          "belief_state", "slot")))
    return td.DataLoader(
        dataset=dataset,
        collate_fn=lambda batch: DSTBatchData.from_dict(collator(batch)),
        **dl_kwargs
    )
