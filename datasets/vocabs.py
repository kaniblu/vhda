__all__ = ["VocabSet", "StateVocabSet", "VocabSetFactory",
           "StateVocabSetFactory"]

from dataclasses import dataclass
from typing import Iterable, Callable, Mapping, Dict, Sequence

import utils
from utils import VocabularyFactory
from .common import ActSlotValue
from .common import Turn


@dataclass
class StateVocabSet:
    """Equivalent to Ontology in literature."""
    asv: utils.Vocabulary
    act: utils.Vocabulary
    slot: utils.Vocabulary
    value: utils.Vocabulary
    act_slot: utils.Vocabulary


@dataclass
class VocabSet:
    word: utils.Vocabulary
    speaker: utils.Vocabulary
    goal: StateVocabSet
    state: StateVocabSet
    goal_state: StateVocabSet  # combination of goal and state
    # goal vocabulary set (ontology) per speaker
    speaker_goal: Mapping[str, StateVocabSet]
    # state vocabulary set (ontology) per speaker
    speaker_state: Mapping[str, StateVocabSet]
    # goal and state vocabulary set (ontology) per speaker
    speaker_goal_state: Mapping[str, StateVocabSet]

    @property
    def num_speakers(self):
        return len(self.speaker)


@dataclass
class StateVocabSetFactory:
    # element type: ActSlotValue
    asv: VocabularyFactory = VocabularyFactory()
    # element type: str
    act: VocabularyFactory = VocabularyFactory()
    # element type: str
    slot: VocabularyFactory = VocabularyFactory()
    # element type: str
    value: VocabularyFactory = VocabularyFactory()
    # element type: Tuple[str, str]
    act_slot: VocabularyFactory = VocabularyFactory()

    def add(self, asv: ActSlotValue):
        self.asv.update((asv,))
        self.act.update((asv.act,))
        self.slot.update((asv.slot,))
        self.value.update((asv.value,))
        self.act_slot.update(((asv.act, asv.slot),))

    def update(self, asvs: Iterable[ActSlotValue]):
        for asv in asvs:
            self.add(asv)

    def get_vocabs(self) -> StateVocabSet:
        return StateVocabSet(
            asv=self.asv.get_vocab(),
            act=self.act.get_vocab(),
            slot=self.slot.get_vocab(),
            value=self.value.get_vocab(),
            act_slot=self.act_slot.get_vocab()
        )


@dataclass
class VocabSetFactory:
    """A helper class for managing vocabularies related to task-oriented
    dialogues. Note that dialogue acts (act-slot-value triples) are organized
    by the speakers.
    """
    tokenizer: Callable[[str], Sequence[str]] = None
    word: VocabularyFactory = VocabularyFactory()
    speaker: VocabularyFactory = VocabularyFactory()
    goal_cls: Callable[[], StateVocabSetFactory] = StateVocabSetFactory
    state_cls: Callable[[], StateVocabSetFactory] = StateVocabSetFactory
    goal_state_cls: Callable[[], StateVocabSetFactory] = StateVocabSetFactory
    _goal: StateVocabSetFactory = utils.private_field(default=None)
    _state: StateVocabSetFactory = utils.private_field(default=None)
    _goal_state: StateVocabSetFactory = utils.private_field(default=None)
    _goal_factories: Dict[str, StateVocabSetFactory] = \
        utils.private_field(default_factory=dict)
    _state_factories: Dict[str, StateVocabSetFactory] = \
        utils.private_field(default_factory=dict)
    _goal_state_factories: Dict[str, StateVocabSetFactory] = \
        utils.private_field(default_factory=dict)

    def __post_init__(self):
        self._goal = self.goal_cls()
        self._state = self.state_cls()
        self._goal_state = self.goal_state_cls()

        def tokenize(s):
            return s.split()

        self.tokenizer = self.tokenizer or tokenize

    def update_turn(self, turn: Turn):
        tokenizer = self.tokenizer
        self.word.update(tokenizer(turn.text))
        for sent in turn.asr:
            self.word.update(tokenizer(sent))
        for asv in turn.state:
            self.word.update(tokenizer(asv.act))
            self.word.update(tokenizer(asv.slot))
            self.word.update(tokenizer(asv.value))
        for asv in turn.goal:
            self.word.update(tokenizer(asv.act))
            self.word.update(tokenizer(asv.slot))
            self.word.update(tokenizer(asv.value))
        self.speaker.update((turn.speaker,))
        self._goal.update(turn.goal)
        self._state.update(turn.state)
        self._goal_state.update(turn.goal)
        self._goal_state.update(turn.state)
        if turn.speaker not in self._state_factories:
            self._state_factories[turn.speaker] = self.state_cls()
        self._state_factories[turn.speaker].update(turn.state)
        if turn.speaker not in self._goal_factories:
            self._goal_factories[turn.speaker] = self.goal_cls()
        self._goal_factories[turn.speaker].update(turn.goal)
        if turn.speaker not in self._goal_state_factories:
            self._goal_state_factories[turn.speaker] = self.goal_state_cls()
        self._goal_state_factories[turn.speaker].update(turn.goal)
        self._goal_state_factories[turn.speaker].update(turn.state)

    def update_turns(self, turns: Iterable[Turn]):
        for turn in turns:
            self.update_turn(turn)

    def update_words(self, words: Iterable[str]):
        self.word.update(words)

    def get_vocabs(self) -> VocabSet:
        return VocabSet(
            word=self.word.get_vocab(),
            speaker=self.speaker.get_vocab(),
            goal=self._goal.get_vocabs(),
            state=self._state.get_vocabs(),
            goal_state=self._goal_state.get_vocabs(),
            speaker_goal={spkr: factory.get_vocabs()
                          for spkr, factory in self._goal_factories.items()},
            speaker_state={spkr: factory.get_vocabs()
                           for spkr, factory in self._state_factories.items()},
            speaker_goal_state={s: f.get_vocabs()
                                for s, f in self._goal_state_factories.items()}
        )
