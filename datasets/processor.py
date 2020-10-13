__all__ = ["SentProcessor", "DialogProcessor"]

import random
import warnings
from dataclasses import dataclass
from typing import Sequence, Optional, ClassVar

import nltk
import torch

import utils
from utils import VocabularyFactory
from datasets import Turn
from datasets import Dialog
from .common import DialogState
from .common import ActSlotValue
from .vocabs import VocabSet
from .vocabs import VocabSetFactory
from .collator import PaddingCollator
from .dataloader import TurnData
from .dataloader import ConvData
from .tokenizer import tokenize
from .vocabs import StateVocabSetFactory


@dataclass
class SentProcessor:
    bos: bool = False
    eos: bool = False
    lowercase: bool = False
    tokenizer: str = "space"
    max_len: Optional[int] = None

    def __post_init__(self):
        if self.max_len is not None and self.max_len < 3:
            raise ValueError(f"max sentence length must be "
                             f"larger than or equal to 3: {self.max_len}")

    def tokenize(self, sent: str) -> Sequence[str]:
        return tokenize(sent, self.tokenizer)

    def process_tokens(self, tokens: Sequence[str]) -> Sequence[str]:
        tokens = [token.lower() for token in tokens]
        if self.max_len is not None:
            effective_max_len = self.max_len
            if self.bos:
                effective_max_len -= 1
            if self.eos:
                effective_max_len -= 1
            tokens = tokens[:effective_max_len]
        return ((["<bos>"] if self.bos else []) + tokens +
                (["<eos>"] if self.eos else []))

    def process(self, sent: str) -> Sequence[str]:
        return self.process_tokens(self.tokenize(sent))


@dataclass
class DialogProcessor:
    sent_processor: SentProcessor
    vocabs: Optional[VocabSet] = None
    boc: bool = False
    eoc: bool = False
    state_order: str = "lexicographical"
    max_len: Optional[int] = None
    asv_pad: ClassVar[ActSlotValue] = ActSlotValue("<pad>", "<pad>", "<pad>")

    def __post_init__(self):
        if self.state_order not in \
                {"lexicographical", "randomized", "unordered"}:
            raise ValueError(f"unsupported state order: {self.state_order}")

    def tokenize(self, text: str):
        return self.sent_processor.tokenize(text)

    @property
    def is_initialized(self):
        return self.vocabs is not None

    def sort_asv(self, asvs: Sequence[ActSlotValue]) -> Sequence[ActSlotValue]:
        asvs = list(asvs)
        if self.state_order == "lexicographical":
            asvs.sort()
        elif self.state_order == "randomized":
            random.shuffle(asvs)
        elif self.state_order == "unordered":
            pass
        else:
            raise ValueError(f"unsupported order: {self.state_order}")
        return asvs

    def prepare_vocabs(self, data: Sequence[Dialog], max_word_size=None):
        if self.is_initialized:
            warnings.warn("reinitializing dialog processor")
        vf = VocabSetFactory(
            tokenizer=self.tokenize,
            word=VocabularyFactory(
                max_size=max_word_size,
                reserved=(["<unk>", "(", "=", ")", "<pad>", "<eos>"] +
                          (["<bos>"] if self.sent_processor.bos else []) +
                          (["<boc>"] if self.boc else []) +
                          (["<eoc>"] if self.eoc else []))
            ),
            speaker=VocabularyFactory(reserved=["<unk>"]),
            goal_cls=lambda: StateVocabSetFactory(
                asv=VocabularyFactory(reserved=[self.asv_pad]),
            ),
            state_cls=lambda: StateVocabSetFactory(
                asv=VocabularyFactory(reserved=[self.asv_pad]),
            ),
            goal_state_cls=lambda: StateVocabSetFactory(
                asv=VocabularyFactory(reserved=[self.asv_pad])
            )
        )
        vf.update_turns(turn for dialog in data for turn in dialog)
        self.vocabs = vf.get_vocabs()

    def tensorize_processed_tokens(self, tokens: Sequence[str]):
        if not self.is_initialized:
            raise RuntimeError(f"vocabulary unset")
        return torch.LongTensor([
            self.vocabs.word[w] if w in self.vocabs.word
            else self.vocabs.word["<unk>"] for w in tokens
        ])

    def tensorize_tokens(self, tokens: Sequence[str]):
        return self.tensorize_processed_tokens(
            self.sent_processor.process_tokens(tokens))

    def tensorize_sent(self, sent: str):
        return self.tensorize_processed_tokens(
            self.sent_processor.process(sent))

    def tensorize_asv(self, asv: ActSlotValue):
        tokens = (list(self.tokenize(asv.act)) + ["("] +
                  list(self.tokenize(asv.slot)) + ["="] +
                  list(self.tokenize(asv.value)) + [")", "<eos>"])
        return self.tensorize_processed_tokens(tokens)

    def tensorize_state_vocab(self, mode="state", speaker=None,
                              tensorizer=None) -> utils.Stacked1DTensor:
        if mode not in {"state", "goal", "goal_state"}:
            raise ValueError(f"unsupported mode: {mode}")
        if not self.is_initialized:
            raise RuntimeError(f"vocabulary unset")
        if speaker is None:
            vocab = getattr(self.vocabs, mode).asv
        else:
            vocab = getattr(self.vocabs, f"speaker_{mode}")[speaker].asv
        state = sorted(vocab.i2f.items(), key=lambda x: x[0])
        tensors = list(map(tensorizer or self.tensorize_asv,
                           (v for k, v in state)))
        return utils.pad_stack(tensors)

    def tensorize_goal(self, state: DialogState, pad=False, speaker=None):
        asvs = list(self.sort_asv(list(state)))
        if pad:
            asvs.append(self.asv_pad)
        if speaker is None:
            vocab = self.vocabs.goal.asv
        else:
            vocab = self.vocabs.speaker_goal[speaker].asv
        return torch.LongTensor([vocab.f2i[asv] for asv in asvs])

    def tensorize_state(self, state: DialogState, pad=False, speaker=None):
        asvs = list(self.sort_asv(list(state)))
        if pad:
            asvs.append(self.asv_pad)
        if speaker is None:
            vocab = self.vocabs.state.asv
        else:
            vocab = self.vocabs.speaker_state[speaker].asv
        return torch.LongTensor([vocab.f2i[asv] for asv in asvs])

    def tensorize_slot(self, state: DialogState, speaker=None):
        if speaker is None:
            vocab = self.vocabs.state.slot
        else:
            vocab = self.vocabs.speaker_state[speaker].slot
        return torch.LongTensor([vocab.f2i[asv.slot] for asv in state])

    def tensorize_act(self, state: DialogState, speaker=None):
        if speaker is None:
            vocab = self.vocabs.state.act
        else:
            vocab = self.vocabs.speaker_state[speaker].act
        return torch.LongTensor([vocab.f2i[asv.act] for asv in state])

    def tensorize_turn(self, turn: Turn):
        return {
            "sent": (self.tensorize_tokens([turn.text])
                     if self.is_boc(turn) or self.is_eoc(turn) else
                     self.tensorize_sent(turn.text)),
            "speaker": torch.tensor(self.vocabs.speaker[turn.speaker]),
            "state": self.tensorize_state(turn.state, turn.speaker),
            "goal": self.tensorize_goal(turn.goal, turn.speaker),
        }

    def tensorize_turn_global_state(self, turn: Turn):
        """Tensorizes turns using global (goal + state) ASV vocabulary."""
        asv_vocab = self.vocabs.goal_state.asv

        def tensorize_state(state: DialogState):
            return torch.LongTensor([asv_vocab.f2i[asv]
                                     for asv in list(state) + [self.asv_pad]])

        return {
            "sent": (self.tensorize_tokens([turn.text])
                     if self.is_boc(turn) or self.is_eoc(turn) else
                     self.tensorize_sent(turn.text)),
            "speaker": torch.tensor(self.vocabs.speaker[turn.speaker]),
            "state": tensorize_state(turn.state),
            "goal": tensorize_state(turn.goal),
        }

    def tensorize(self, dialog: Dialog, turn_tensorizer=None):
        turn_tensorizer = turn_tensorizer or self.tensorize_turn
        turns = list(dialog.turns)
        if self.max_len is not None and len(turns) > self.max_len:
            turns = turns[:self.max_len]
        if self.boc:
            turns = [Turn("<boc>", "<unk>")] + turns
        if self.eoc:
            turns = turns + [Turn("<eoc>", "<unk>")]
        collator = PaddingCollator(frozenset(("sent", "goal", "state", "slot")))
        return collator(list(map(turn_tensorizer, turns)))

    def lexicalize_sent(self, tokens: torch.Tensor) -> str:
        if not self.is_initialized:
            raise RuntimeError(f"vocabulary unset")
        tokens = [self.vocabs.word.i2f.get(x.item(), "<unk>") for x in tokens]
        if self.sent_processor.bos:
            tokens = utils.lstrip(tokens, "<bos>")
        if self.sent_processor.eos:
            tokens = utils.rstrip(tokens, "<eos>")
        return " ".join(tokens)

    def lexicalize_goal(self, goal: torch.Tensor, speaker=None
                        ) -> DialogState:
        if speaker is None:
            vocab = self.vocabs.goal.asv
        else:
            vocab = self.vocabs.speaker_goal[speaker].asv
        ret = DialogState()
        for idx in goal.tolist():
            asv = vocab.i2f[idx]
            if asv == self.asv_pad:
                continue
            ret.add(asv)
        return ret

    def lexicalize_state_global(self, state: torch.Tensor) -> DialogState:
        vocab = self.vocabs.goal_state.asv
        ret = DialogState()
        for idx in state.tolist():
            asv = vocab.i2f[idx]
            if asv == self.asv_pad:
                continue
            ret.add(asv)
        return ret

    def lexicalize_state(self, state: torch.Tensor, speaker=None
                         ) -> DialogState:
        if speaker is None:
            vocab = self.vocabs.state.asv
        else:
            vocab = self.vocabs.speaker_state[speaker].asv
        ret = DialogState()
        for idx in state.tolist():
            asv = vocab.i2f[idx]
            if asv == self.asv_pad:
                continue
            ret.add(asv)
        return ret

    def lexicalize_turn(self, data: TurnData, speaker_vocab=False) -> Turn:
        speaker = self.vocabs.speaker[data.speaker.item()]
        return Turn(
            text=self.lexicalize_sent(data.sent),
            speaker=speaker,
            goal=self.lexicalize_goal(data.goal,
                                      speaker if speaker_vocab else None),
            state=self.lexicalize_state(data.state,
                                        speaker if speaker_vocab else None)
        )

    def lexicalize_turn_global(self, data: TurnData) -> Turn:
        speaker = self.vocabs.speaker[data.speaker.item()]
        return Turn(
            text=self.lexicalize_sent(data.sent),
            speaker=speaker,
            goal=self.lexicalize_state_global(data.goal),
            state=self.lexicalize_state_global(data.state)
        )

    @staticmethod
    def is_boc(turn):
        return "<boc>" in turn.text

    @staticmethod
    def is_eoc(turn):
        return "<eoc>" in turn.text

    def lexicalize(self, data: ConvData) -> Dialog:
        turns = list(map(self.lexicalize_turn, data))
        if self.boc:
            turns = utils.lstrip(turns, self.is_boc)
        if self.eoc:
            turns = utils.rstrip(turns, self.is_eoc)
        return Dialog(turns)

    def lexicalize_global(self, data: ConvData) -> Dialog:
        turns = list(map(self.lexicalize_turn_global, data))
        if self.boc:
            turns = utils.lstrip(turns, self.is_boc)
        if self.eoc:
            turns = utils.rstrip(turns, self.is_eoc)
        return Dialog(turns)

    def save(self, path):
        utils.save_pickle(self, path)

    @staticmethod
    def from_pickle(path):
        return utils.load_pickle(path)
