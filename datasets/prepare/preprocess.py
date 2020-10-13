__all__ = ["DialogPreprocessor"]

import string
from dataclasses import dataclass
from typing import Optional, Mapping, ClassVar

import utils
from ..common import Turn
from ..common import Dialog
from ..common import DialogState
from ..common import ActSlotValue
from ..tokenizer import tokenize


@dataclass
class DialogPreprocessor:
    lowercase: bool = True
    replace_number: Optional[str] = None
    tokenizer: str = "corenlp"
    special_chars: Optional[str] = ",.?'"
    _charset: set = utils.private_field(default_factory=set)
    _replace_token: ClassVar[Mapping[str, str]] = {
        "&": "and"
    }

    def __post_init__(self):
        self._charset.update((list(self.special_chars)
                              if self.special_chars is not None else []) +
                             list(string.ascii_letters))

    def filter_word(self, word: str) -> Optional[str]:
        if not word:
            return
        if word.isdigit():
            if self.replace_number is not None:
                word = self.replace_number
            return word
        if self.lowercase:
            word = word.lower()
        word = self._replace_token.get(word, word)
        if any(c not in self._charset for c in word):
            return
        return word

    def preprocess_sent(self, sent: str) -> str:
        tokens = tokenize(sent, self.tokenizer)
        tokens = filter(None, map(self.filter_word, tokens))
        return " ".join(tokens)

    def preprocess_state(self, state: DialogState) -> DialogState:
        def preprocess_token(token):
            if self.lowercase:
                token = token.lower()
            return token.strip()

        new_state = DialogState()
        for asv in state:
            new_state.add(ActSlotValue(*map(preprocess_token, asv.values)))
        return new_state

    def preprocess_turn(self, turn: Turn) -> Turn:
        return Turn(
            text=self.preprocess_sent(turn.text),
            speaker=turn.speaker,
            goal=self.preprocess_state(turn.goal),
            state=self.preprocess_state(turn.state),
            asr={self.preprocess_sent(sent): score
                 for sent, score in turn.asr.items()},
            meta=turn.meta
        )

    def preprocess(self, dialog: Dialog) -> Dialog:
        return Dialog(
            turns=list(map(self.preprocess_turn, dialog.turns)),
            meta=dialog.meta
        )
