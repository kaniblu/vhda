__all__ = ["SlotValue", "ActSlotValue", "DialogState", "DialogAct", "DSTTurn",
           "Turn", "DSTUserTurn", "DSTWizardTurn", "Dialog", "DSTDialog"]

import re
import copy
import logging
import warnings
from dataclasses import dataclass, field
from typing import Sequence, ClassVar, Mapping, Set, Dict, Iterable

import utils


@dataclass(order=True, frozen=True)
class SlotValue:
    slot: str
    value: str
    _parse_regex: ClassVar[re.Pattern] = re.compile(r"^([^=]+)=([^=]+)$")

    @classmethod
    def parse_from(cls, s):
        match = cls._parse_regex.match(s)
        if not match:
            raise ValueError(f"invalid slot-value string: '{s}'")
        return SlotValue(match.group(1), match.group(2))

    def __str__(self):
        return f"{self.slot}={self.value}"


@dataclass(order=True, frozen=True)
class ActSlotValue:
    act: str
    slot: str
    value: str
    _regex: ClassVar[re.Pattern] = re.compile(r"^(.+)\((.*)\)$")

    @property
    def values(self):
        return self.act, self.slot, self.value

    @property
    def sv(self):
        return SlotValue(self.slot, self.value)

    @classmethod
    def parse_from(cls, s):
        match = cls._regex.match(s)
        if not match:
            raise ValueError(f"invalid act-slot-value string: '{s}'")
        return ActSlotValue(
            act=match.group(1).strip(),
            slot=match.group(2).strip(),
            value=match.group(3).strip()
        )

    def to_dict(self):
        return {
            "act": self.act,
            "slot": self.slot,
            "value": self.value
        }

    def __str__(self):
        return f"{self.act}({self.slot}={self.value})"

    def clone(self, **kwargs):
        self_kwargs = self.to_dict()
        self_kwargs.update(kwargs)
        return ActSlotValue(**self_kwargs)


@dataclass(order=True, frozen=True)
class ActSlotValuePairs:
    act: str
    sv_pairs: Set[SlotValue]
    _regex_all: ClassVar[re.Pattern] = \
        re.compile(r"^(.+)\(((([^=]+)=([^=]+))(\s*,\s*([^=]+)=([^=]+))*)?\)$")
    _regex_svpair: ClassVar[re.Pattern] = re.compile(r"([^=,]+)=([^=,]+)")

    @classmethod
    def parse_from(cls, s):
        match = cls._regex_all.match(s)
        if not match:
            raise ValueError(f"invalid ASV string: '{s}'")
        sv_pairs = match.group(2)
        matches = cls._regex_svpair.findall(sv_pairs)
        return ActSlotValuePairs(
            act=match.group(1),
            sv_pairs={SlotValue(m[0].strip(), m[1].strip())
                      for m in matches}
        )

    def clone(self):
        return ActSlotValuePairs(self.act, set(self.sv_pairs))

    def __iter__(self):
        """Iterates through ActSlotValue(s)"""
        for sv in self.sv_pairs:
            yield ActSlotValue(self.act, sv.slot, sv.value)

    def __str__(self):
        return f"{self.act}({', '.join(map(str, self.sv_pairs))})"


@dataclass
class DialogAct:
    """slot -> value set"""
    data: Dict[str, Set[str]] = field(default_factory=dict)

    @classmethod
    def parse_from(cls, s):
        pass

    @property
    def slots(self):
        return set(self.data)

    def __iter__(self):
        for s, vs in self.data.items():
            for v in vs:
                yield SlotValue(s, v)

    def __eq__(self, other):
        if not isinstance(other, DialogAct):
            raise TypeError(f"unsupported operand type: {other}")
        return self.data == other.data

    def __len__(self):
        return sum(map(len, self.data.values()))

    def __bool__(self):
        return len(self.data) > 0

    def __contains__(self, item):
        return item in self.slots

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def get(self, key, default=frozenset()):
        return self.data.get(key, default)

    def to_json(self):
        return {s: list(vs) for s, vs in self.data.items()}

    @classmethod
    def from_json(cls, data):
        return cls({s: set(vs) for s, vs in data.items()})

    def add(self, sv: SlotValue):
        if sv.slot not in self.data:
            self.data[sv.slot] = set()
        self.data[sv.slot].add(sv.value)

    def remove(self, sv: SlotValue, silent=False):
        if sv.slot not in self.data:
            if not silent:
                raise KeyError(f"'{sv.slot}' does not exist")
            else:
                return
        if sv.value not in self.data[sv.slot]:
            if not silent:
                raise KeyError(f"{sv} does not exist")
            else:
                return
        self.data[sv.slot].remove(sv.value)

    def update(self, svs: Iterable[SlotValue]):
        for sv in svs:
            self.add(sv)

    def __or__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"unsupported addition operand: {type(other)}")
        act = copy.deepcopy(self)
        for sv in other:
            act.add(sv)
        return act


@dataclass
class DialogState:
    """act -> [ slot -> value set ]"""
    data: Dict[str, DialogAct] = field(default_factory=dict)

    def __post_init__(self):
        self.data = {k: v for k, v in self.data.items() if v}

    @classmethod
    def parse_from(cls, s):
        if not s.strip():
            return DialogState()
        tokens = [t.strip() for t in s.split("|")]
        asv_pairs = set(map(ActSlotValuePairs.parse_from, tokens))
        state = DialogState()
        for asp in asv_pairs:
            for asv in asp:
                state.add(asv)
        return state

    @property
    def acts(self):
        return set(self.data)

    def __eq__(self, other):
        if not isinstance(other, DialogState):
            raise TypeError(f"unsupported operand type: {other}")
        return self.data == other.data

    def __len__(self):
        return sum(map(len, self.data.values()))

    def __bool__(self):
        return len(self.data) > 0

    def __contains__(self, item):
        return item in self.acts

    def __iter__(self):
        """Iterates through ActSlotValue(s)"""
        for act, da in self.data.items():
            for sv in da:
                yield ActSlotValue(act, sv.slot, sv.value)

    def __getitem__(self, item):
        return self.data[item]

    def get(self, item, default=DialogAct()):
        return self.data.get(item, default)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def to_json(self):
        return {act: da.to_json() for act, da in self.data.items()}

    @classmethod
    def from_json(cls, data):
        return cls({act: DialogAct.from_json(da) for act, da in data.items()})

    @classmethod
    def from_asvs(cls, data: Sequence[ActSlotValue]):
        state = DialogState()
        for asv in data:
            state.add(asv)
        return state

    def to_cam(self):
        return "|".join(
            f"{act}({', '.join(f'{sv.slot}={sv.value}' for sv in sorted(da))})"
            for act, da in sorted(self.data.items()))

    def add(self, asv: ActSlotValue):
        if asv.act not in self.data:
            self.data[asv.act] = DialogAct()
        self.data[asv.act].add(asv.sv)

    def remove(self, asv: ActSlotValue, silent=False):
        if asv.act not in self.data:
            if not silent:
                raise KeyError(f"'{asv.act}' does not exist")
            else:
                return
        self.data[asv.act].remove(asv.sv, silent)

    def update(self, asvs: Iterable[ActSlotValue]):
        for asv in asvs:
            self.add(asv)

    def clone(self):
        return copy.deepcopy(self)

    def __str__(self):
        return self.to_cam()

    def __or__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"unsupported addition operand: {type(other)}")
        state = self.clone()
        for asv in other:
            state.add(asv)
        return state


class Turn:

    def __init__(self,
                 text: str, speaker: str,
                 goal: DialogState = None,
                 state: DialogState = None,
                 asr: Mapping[str, float] = None,
                 meta: Mapping = None):
        self._text = text
        self._speaker = speaker
        self._goal = goal or DialogState()
        self._state = state or DialogState()
        self._asr = asr or dict()
        self._meta = meta or dict()

    @property
    def text(self):
        return self._text

    @property
    def speaker(self):
        return self._speaker

    @property
    def goal(self):
        return self._goal

    @property
    def state(self):
        return self._state

    @property
    def asr(self):
        if not self._asr:
            return {self.text: 1.0}
        return self._asr

    @property
    def meta(self):
        return self._meta

    def clone(self, **kwargs):
        clone_kwargs = {
            "text": self.text,
            "speaker": self._speaker,
            "goal": self._goal,
            "state": self._state,
            "asr": self._asr,
            "meta": self._meta
        }
        clone_kwargs.update(kwargs)
        return Turn(**clone_kwargs)

    @classmethod
    def from_json(cls, data):
        return Turn(
            text=data["text"],
            speaker=data["speaker"],
            goal=DialogState.from_json(data.get("goal")),
            state=DialogState.from_json(data.get("state")),
            asr=data.get("asr"),
            meta=data.get("meta"),
        )

    def to_json(self) -> dict:
        return dict(
            text=self.text,
            speaker=self.speaker,
            goal=self.goal.to_json(),
            state=self.state.to_json(),
            asr=self.asr,
            meta=self.meta
        )

    def __eq__(self, other):
        if not isinstance(other, Turn):
            raise TypeError(f"unsupported type: {type(other)}")
        return (self.text == other.text and
                self.speaker == other.speaker and
                self.state == other.state and
                self.goal == other.goal and
                self.asr == other.asr)

    def __add__(self, other):
        if not isinstance(other, Turn):
            raise TypeError(f"unsupported operand type: {type(other)}")
        return Turn(
            text=self.text + " " + other.text,
            speaker=self.speaker,
            goal=self.goal | other.goal,
            state=self.state | other.state,
            asr={k: self.asr.get(k, 0) + other.asr.get(k, 0)
                 for k in set(self.asr.keys()) | set(other.asr.keys())},
            meta=utils.merge_dict(self.meta, other.meta)
        )


class DSTUserTurn(Turn):
    """Represents a user turn in a DST-style dialogue."""

    def __init__(self, text: str,
                 goal: Mapping[str, str] = None,
                 inform: Mapping[str, str] = None,
                 request: Set[str] = None,
                 asr: Mapping[str, float] = None,
                 meta: Mapping = None):
        super().__init__(
            text=text,
            speaker="user",
            goal=None,
            state=None,
            asr=asr,
            meta=meta
        )
        self._goal = goal or dict()
        self._inform = inform or dict()
        self._request = request or set()

    @property
    def inform(self):
        return self._inform

    @property
    def request(self):
        return self._request

    @property
    def goal(self):
        goal = DialogState()
        goal.update(ActSlotValue("inform", slot, value)
                    for slot, value in self._goal.items())
        return goal

    @property
    def state(self):
        state = DialogState()
        for slot, value in self._inform.items():
            state.add(ActSlotValue("inform", slot, value))
        for slot in self._request:
            state.add(ActSlotValue("request", "slot", slot))
        return state

    def validate(self):
        """Checks the integrity of this turn as a DST-style user turn.
        A valid user turn must have
            - its inform-act slots as a subset of the goal slots
            - its inform-act slot values match the goal slot values.
        """
        inform_slots = set(self._inform.keys())
        goal_slots = set(self._goal.keys())
        if not inform_slots.issubset(goal_slots):
            raise RuntimeError(f"inform slot is not a subset of goal slot: "
                               f"{inform_slots} not in {goal_slots}")
        for slot, value in self._inform.items():
            if value != self._goal[slot]:
                raise RuntimeError(f"inform slot value is not equal the "
                                   f"equivalent slot value in goal: "
                                   f"{value} != {self._goal[slot]}")

    @classmethod
    def from_turn(cls, turn: Turn):
        def extract_inform(state: DialogState):
            inform = dict()
            for sv in state.get("inform"):
                inform[sv.slot] = sv.value
            return inform

        if turn.speaker != "user":
            raise RuntimeError(f"turn cannot be converted: not a user's turn")
        return cls(
            text=turn.text,
            goal=extract_inform(turn.goal),
            inform=extract_inform(turn.state),
            request=set(turn.state.get("request").get("slot")),
            asr=turn.asr,
            meta=turn.meta
        )

    def clone(self, **kwargs):
        clone_kwargs = {
            "text": self.text,
            "goal": self._goal,
            "inform": self._inform,
            "request": self._request,
            "asr": self._asr,
            "meta": self._meta
        }
        clone_kwargs.update(kwargs)
        return DSTUserTurn(**clone_kwargs)


class DSTWizardTurn(Turn):
    """Represents a user turn in a DST-style dialogue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, speaker="wizard", **kwargs)

    @classmethod
    def from_turn(cls, turn: Turn):
        if turn.speaker != "wizard":
            raise RuntimeError(f"turn cannot be converted: not a wizard's turn")
        return cls(
            text=turn.text,
            goal=turn.goal,
            state=turn.state,
            asr=turn.asr,
            meta=turn.meta
        )

    def clone(self, **kwargs):
        clone_kwargs = {
            "text": self.text,
            "goal": self.goal,
            "state": self.state,
            "asr": self.asr,
            "meta": self.meta
        }
        clone_kwargs.update(kwargs)
        return DSTWizardTurn(**clone_kwargs)


@dataclass
class DSTTurn:
    wizard: DSTWizardTurn
    user: DSTUserTurn

    def resolve_this(self):
        """DSTC2: 'this' slot resolution.
        reference: https://github.com/jeremyfix/dstc/blob/ee8e0e90db5ea8917c178e8800e0f02ce953b4e6/baseline_yarbus.py#L92
        """

        def _resolve_this():
            cands = set()
            for wiz_asv in self.wizard.state:
                if wiz_asv.act == "request":
                    cands.add(wiz_asv.value)
                elif wiz_asv.act == "explicit confirm":
                    cands.add(wiz_asv.slot)
                elif wiz_asv.act == "select":
                    cands.add(wiz_asv.slot)
            if len(cands) == 1:
                return next(iter(cands))

        inform = dict()
        for asv in self.user.state:
            if asv.act == "inform" and asv.slot == "this":
                slot = _resolve_this()
                if slot is not None:
                    inform[slot] = asv.value
            elif asv.act == "inform":
                inform[asv.slot] = asv.value
        return DSTTurn(
            wizard=self.wizard,
            user=self.user.clone(inform=inform)
        )


class Dialog:

    def __init__(self, turns: Sequence[Turn], meta: Mapping = None):
        self._turns = turns
        self._meta = meta or dict()

        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def turns(self):
        return self._turns

    @property
    def meta(self):
        return self._meta

    def __len__(self):
        return len(self.turns)

    @classmethod
    def from_json(cls, data):
        return cls(
            turns=[Turn.from_json(turn) for turn in data["turns"]],
            meta=data.get("meta", {})
        )

    def to_json(self) -> dict:
        return dict(turns=[t.to_json() for t in self.turns], meta=self.meta)

    def __eq__(self, other):
        if not isinstance(other, Dialog):
            return False
        return len(self) == len(other) and all(x == y for x, y in
                                               zip(self.turns, other.turns))

    def __str__(self):
        return utils.DialogTableFormatter().format(self)

    def __iter__(self):
        return iter(self.turns)

    def __bool__(self):
        return len(self.turns) > 0

    def __repr__(self):
        return f"Dialog({len(self.turns)} turns)"

    def filter(self, speaker):
        for turn in self:
            if turn.speaker == speaker:
                yield turn


class DSTDialog(Dialog):

    def __init__(self, turns: Sequence[DSTTurn], meta: Mapping = None):
        super().__init__(
            turns=[],
            meta=meta
        )
        self._dst_turns = turns

    @property
    def dst_turns(self):
        return self._dst_turns

    @property
    def turns(self):
        return [turn for dst_turn in self.dst_turns
                for turn in (dst_turn.wizard, dst_turn.user)]

    @classmethod
    def from_dialog(cls, dialog: Dialog, force=False):
        """Creates DSTDialog from the given dialog object. Assumes that the
        dialog fits the conditions of a DSTDialog. If not, the behavior is
        unexpected.
        To force the dialog into a DSTDialog, use `force` option.

        The `force` option will try to prune and clean the dialogue as little
        but necessary as possible without breaking the conditions.
        """

        def clean(dialog: Dialog) -> Dialog:
            if len(dialog) == 0:
                warnings.warn("dialogue contains no turns (0 length)")
                return dialog
            speaker_set = {"user", "wizard"}
            turns = list(dialog.turns)
            # remove extra speakers
            turns = [turn for turn in turns if turn.speaker in speaker_set]
            # ensure the first and last turns are by the wizard and the user
            if turns[0].speaker != "wizard":
                turns.insert(0, DSTWizardTurn(text=""))
            if turns[-1].speaker != "user":
                turns = turns[:-1]
            # ensure the wizard and the user are alternating
            turns = list(
                utils.bucket(turns, lambda a, b: a.speaker == b.speaker))
            for i in range(len(turns)):
                turn = turns[i]
                if len(turn) == 1:
                    turns[i] = turn[0]
                    continue
                cum_turn = None
                for t in turn:
                    if cum_turn is None:
                        cum_turn = t
                    else:
                        cum_turn = cum_turn + t
                turns[i] = cum_turn
            return Dialog(turns, dialog.meta)

        if force:
            dialog = clean(dialog)
        return DSTDialog(
            turns=[DSTTurn(DSTWizardTurn.from_turn(w), DSTUserTurn.from_turn(u))
                   for w, u in zip(dialog.turns[::2], dialog.turns[1::2])],
            meta=dialog.meta
        )

    def compute_user_goals(self):
        """User goals are computed by accumulating turn-specific inform-type
        dialogue states of user turns.

        Goals are assumed to be dictionary structures, hence the latest slot
        value will override any previous mentions of the same slot.
        If multiple slots are mentioned in the same turn, then a random
        slot-value will be selected with a warning.
        """
        cum_goal = dict()
        for turn in self.dst_turns:
            for slot, value in turn.user.inform.items():
                cum_goal[slot] = value
            turn.user = DSTUserTurn(
                text=turn.user.text,
                goal=copy.copy(cum_goal),
                inform=turn.user.inform,
                request=turn.user.request,
                asr=turn.user.asr,
                meta=turn.user.meta
            )

    def validate(self):
        """Validate whether the dialogue is a task-oriented dialogue suitable
        for dialogue state tracking. A run-time error is raised if else.

        A valid DST-friendly dialogue must
            - be a conversation between the user and the wizard
            - be initiated by the wizard and terminated by the user
            - contain alternating turns between the user and the wizard
            - each user inform state must a subset of each user goal
        """
        for turn in self.dst_turns:
            turn.user.validate()
        # raise NotImplementedError
        # extra_speakers = (set(turn.speaker for turn in self.turns) -
        #                   {"user", "wizard"})
        # if extra_speakers:
        #     raise RuntimeError(f"contains speakers other than user and wizard: "
        #                        f"{extra_speakers}")
        # if self.turns[0].speaker != "wizard":
        #     raise RuntimeError(f"must be initiated by the wizard: "
        #                        f"{self.turns[0].speaker}")
        # if len(self.turns) % 2 != 0:
        #     raise RuntimeError(f"must contain even number of turns: "
        #                        f"{len(self.turns)}")
        # spkr_even = set(self.turns[i].speaker for i in range(len(self), 2))
        # spkr_odd = set(self.turns[i].speaker for i in range(1, len(self), 2))
        # if len(spkr_even) != 1 or spkr_even != "wizard":
        #     raise RuntimeError(f"unexpected speaker in "
        #                        f"even-indexed turns: {spkr_even}")
        # if len(spkr_odd) != 1 or spkr_odd != "user":
        #     raise RuntimeError(f"unexpected speaker in "
        #                        f"even-indexed turns: {spkr_even}")
        # for turn in self.turns:
        #     if turn.speaker != "user":
        #         continue
        #     if turn.state.acts != {"inform"}:
        #         raise RuntimeError(f"user dialogue state must contain user "
        #                            f"goals only: {turn}")
