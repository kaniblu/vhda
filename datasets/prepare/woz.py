__all__ = ["WoZAdapter"]

import re
import pathlib
from dataclasses import dataclass
from typing import (Sequence, Iterable, Tuple, TypeVar,
                    Mapping, ClassVar, List, Optional)

import utils
from .adapter import DataAdapter
from ..common import Dialog
from ..common import DSTDialog
from ..common import DSTUserTurn
from ..common import DSTWizardTurn
from ..common import Turn
from ..common import DialogState
from ..common import ActSlotValue

K = TypeVar("K")
V = TypeVar("V")


def reduce(kvps: Iterable[Tuple[K, V]]) -> Mapping[K, Sequence[V]]:
    ret = dict()
    for k, v in kvps:
        if k not in ret:
            ret[k] = list()
        ret[k].append(v)
    return ret


@dataclass
class WoZAdapter(DataAdapter):
    """WoZ2.0 Dataset Adapter"""
    delexicalize_location: bool = True
    delexicalize_number: bool = True
    delexicalize_name: bool = True
    delexicalize_postcode: bool = True
    delexicalize_price: bool = True
    delexicalize_time: bool = True
    normalize_slot_values: bool = True
    price_regex: ClassVar[re.Pattern] = re.compile(r"\$\d+(\.\d*)?")
    time_regex: ClassVar[re.Pattern] = \
        re.compile(r"\d{2}:\d{2}\s*([aA][mM]|[pP][mM])?")
    postcode_regex: ClassVar[re.Pattern] = re.compile(
        r"[cC]\s*\.?\??\s*[bB]\s*\.?\??\s*(\d\s*[.,-]*\s*)*"
        r"[a-zA-Z]\s*\.?\s*[a-zA-Z]"
    )
    slot_norm_map: ClassVar[Mapping[str, str]] = {
        "centre": "center",
        "areas": "area",
        "phone number": "number",
        "dontcare": "dont care"
    }
    word_map: ClassVar[Mapping[str, str]] = {
        "restuarant": "restaurant",
        "leesure": "leisure",
        "neewmarket": "newmarket",
        "ren": "fen",
        "g4": "g 4",
        "dilton": "ditton",
        "informaion": "information",
        "c?b?1,7d": "code",
        "!": ".",
        "expinsive": "expensive",
        "chery": "cherry",
        "moderatre": "moderate",
        "locdated": "located",
        "differnt": "different",
        "ritish": "british",
        "resataurnts": "restaurants",
        "mathcing": "matching",
        "recomments": "recommends",
        "moderatley": "moderately",
        "adress": "address",
        "nd": "and",
        "&": "and",
        "unfortuntately": "unfortunately",
        "resaturant": "restaurant",
        "wouold": "would",
        "anothe": "another",
        "thier": "their",
        "vietmanese": "vietnamese",
        "margherta": "margherita",
        "nados": "nandos",
        "prefferred": "preferred",
        "malasian": "malaysian",
        "elese": "else",
        "resaurants": "restaurants",
        "issome": "is some",
        "postode": "postcode",
        "den": "fen",
        "tnahh": "thanh",
        "aimcan": "i can",
        "entre": "entree",
        "mediterrean": "mediterranean",
        "inexpesnsive": "inexpensive",
        "chesteron": "chesterton",
        "vince": "vinci",
        "pizzera": "pizzeria",
        "pizzaria": "pizzeria",
        "barbeque": "barbecue",
        "cheapeast": "cheapest",
        "bankok": "bangkok",
        "anarolia": "anatolia",
        "portugese": "portuguese",
        "addres": "address",
        "centr": "center",
        "tanh": "thanh",
        "ther": "there",
        "postr": "post",
        "therea": "there",
        "restaruants": "restaurants",
        "anythig": "anything",
        "bihn": "binh",
        "kitchan": "kitchen",
        "margharitas": "margherita",
        "davinci": "da vinci",
        "thrus": "throughs",
        "tandoon": "tandoori",
        "chicquito": "chiquito",
        "reataurants": "restaurants",
        "abou": "about",
        "hmmm": "hmm",
        "hunington": "huntington",
        "afriad": "afraid",
        "drivethrus": "drive throughs",
        "resturaunt": "restaurant",
        "prefferably": "preferably",
        "saintjjohns": "saint johns",
        "restuarnt": "restaurant",
        "loated": "located",
        "sanit": "saint",
        "woluld": "would",
        "chineses": "chinese",
        "quisine": "cuisine",
        "mising": "missing",
        "servies": "serves",
        "marghertia": "margherita",
        "grafitti": "graffiti",
        "pelase": "please",
        "surem": "sure",
        "is01223": "is 01223",
        "looing": "looking",
        "thankyou": "thank you",
        "alimentun": "alimentum",
        "placeo": "place",
        "epensive": "expensive",
        "magherita": "margherita",
        "moderatly": "moderately",
        "wagamam": "wagamama",
        "mmmm": "mm",
        "resteraunts": "restaurants",
        "eriteran": "eritrean",
        "addresss": "address",
        "seve": "serve",
        "michaellhouse": "michaelhouse",
        "crusine": "cuisine",
        "ex23": "postal",
        "8qu": "code",
        "margharita": "margherita",
        "that'it": "that's it",
        "iat": "at",
        "sita": "and",
    }
    places: ClassVar[List[str]] = [utils.tokenize(line.lower()) for line in [
        "doubletree by hilton cambridge",
        "doubletree by hilton hotel cambridge",
        "doubletree",
        "hilton hotel",
        "Thompsons Lane",
        "Fen Ditton",
        "Bridge Street",
        "Cambridge Leisure Park",
        "Clifton Way",
        "cherry hilton",
        "Cherry HInton road",
        "Cherry HInton",
        "Regent Street",
        "Milton Road",
        "Chesterton",
        "Hills Road",
        "sturton street",
        "sturton",
        "barnweel street",
        "barnweel",
        "barnwheel street",
        "barnwheel",
        "northampton road",
        "northampton street",
        "High Street",
        "Green Street City",
        "green street",
        "city centre",
        "Cambridge Retail Park",
        "Huntingdon Road",
        "huntington road",
        "huntington",
        "huntingdon",
        "Finders Corner",
        "Newmarket Road",
        "King Street",
        "Cherry Hinton Rd.",
        "Mill Road",
        "Trumpington Street",
        "Little Rose",
        "Histon Road",
        "Cambridge Lodge Hotel",
        "Lensfield Road",
        "Millers Yard",
        "Newmarket",
        "Saint Andrews Street",
        "Saint Andrew Street",
        "Saint Andrews",
        "Saint Andrew",
        "Victoria Avenue",
        "Magdalene Street",
        "magdalene",
        "Newnham Road",
        "Newnham",
        "Bridge Street",
        "kings parade",
        "Quayside Off",
        "Castle Street",
        "St . Michael's Church",
        "St . Michael Church",
        "St Michael Church",
        "st michael's church",
        "Trinity Street",
        "Corn Exchange Street",
        "Victoria Road",
        "mill lane",
        "granta place",
        "grafton hotel",
        "grafton",
        "de vere hotel",
        "de vere",
        "university arms",
        "university",
        "crowne plaza hotel",
        "crowne plaza",
        "downing street",
        "homerton street",
        "barnwell road",
        "hotel felix",
        "whitehouse lane",
        "whitehouse",
        "rose crescenta",
        "rose crescent",
        "cambridge",
        "histon",
        "victoria",
        "milton",
    ]]
    names: ClassVar[List[str]] = [utils.tokenize(line.lower()) for line in [
        "The Missing Sock",
        "Ali Baba",
        "Anatolia",
        "Restaurant Two Two",
        "Caffe Uno",
        "Cambridge Lodge Restaurant",
        "Chiquito",
        "Cocum",
        "Cote",
        "Curry Prince",
        "da vinci pizzeria",
        "da vinci pizza",
        "da vinci",
        "Shiraz",
        "Royal Standard",
        "River Bar Steakhouse and grill",
        "River Bar Steakhouse",
        "Kymmoy",
        "La Margherita",
        "margherita",
        "La Raza",
        "La Tasca",
        "Charlie Chan",
        "Golden House",
        "Little Seoul",
        "Loch Fyne",
        "Meghna",
        "Michaelhouse Cafe",
        "Michaelhouse",
        "Nandos City Centre",
        "Nandos",
        "Nirala",
        "Royal Spice",
        "Alimentum",
        "Thanh Binh",
        "Thank Binh",
        "Backstreet Bistro",
        "Gardenia",
        "Golden Wok",
        "gold wok",
        "Gourmet Burger Kitchen",
        "Grafton Hotel",
        "Hakka",
        "Lucky Star",
        "Lan Hong House",
        "La Mimosa",
        "J Restaurant",
        "Loch Fyne",
        "Little Seoul",
        "Meze Bar Restaurant",
        "Meze",
        "Meze Bar",
        "Varisty restaurant",
        "Yippee Noodle Bar",
        "restaurant one seven",
        "Yu Garden",
        "Charlie Chan",
        "taj tandoori",
        "wagamama",
        "Grafton hotel restaurant",
        "cow pizza kitchen and bar",
        "zizzi cambridge",
        "zizzi",
        "prezzo",
        "Frankie and Bennys",
        "Midsummer House Restaurant",
        "Saigon City",
        "La Tasca",
        "Pipasha Restaurant",
        "Pipasha",
        "Sitar Tandoori",
        "sital tandoori",
        "sital",
        "Sitar",
        "Saint johns chop house",
        "Saint johns",
        "Saint John's chop house",
        "darrys cookhouse and wine shop",
        "darrys cookhouse",
        "darry's cookhouse and wine shop",
        "darry's cookhouse",
        "Rajmahal",
        "beni hana",
        "de luca cucina and bar",
        "de luca cucina",
        "Galleria",
        "Cambridges Lodge",
        "Eraina",
        "bedouin",
        "efes",
        "pizza hut",
        "Bloomsbury",
        "Chiquito Restaurant Bar",
        "Stazione Restaurant and Coffee Bar",
        "Stazione Restaurant",
        "Charliie Chan",
        "Rice House",
        "Dojo Noodle Bar",
        "jinling noodle bar",
        "jinling",
        "Peking",
        "Chop House",
        "riverside brasserie",
        "cotto",
        "city stop restaurant",
        "ciquito restaurant bar",
        "ciquito",
        "bangkok city",
        "sala thong",
        "sala thing",
        "tandoori",
        "gandhi",
        "kohinoor",
        "mahal",
        "la razam",
        "graffiti",
        "fitzbillies",
        "maghna",
    ]]

    def normalize_token(self, token: str) -> Optional[str]:
        pass

    @staticmethod
    def remove_duplicate_items(tokens, item):
        return [e[0] for e in
                utils.bucket(tokens, lambda x, y: x == item and y == item)]

    @staticmethod
    def remove_duplicate_items_multi(tokens, needle, cand: set):
        return [needle if needle in e else e[0] for e in
                utils.bucket(tokens, lambda x, y: x in cand and y in cand)]

    @staticmethod
    def merge_loc_and_numeric(tokens):
        return WoZAdapter.remove_duplicate_items_multi(
            tokens, "loc", {"loc", "numeric", ",", "-",
                            "road", "g", "st", "street", "avenue", "rd"}
        )

    @staticmethod
    def merge_phone_number(tokens):
        return WoZAdapter.remove_duplicate_items_multi(
            tokens, "numeric", {"numeric", "-"}
        )

    def normalize_sent(self, text, tokenizer=utils.tokenize):
        text_original = text
        if self.delexicalize_postcode:
            text = self.postcode_regex.sub("some code", text)
        if self.delexicalize_price:
            text = self.price_regex.sub("some price", text)
        if self.delexicalize_time:
            text = self.time_regex.sub("some time", text)
        text = (text.lower().replace(".", " . ").replace("-", " - ")
                .replace("(", "").replace(")", "").replace("#", "")
                .replace(">", "").replace(",", " , ").replace("/", "?"))
        tokens = tokenizer(text)
        tokens = [self.word_map.get(token, token) for token in tokens]
        tokens = " ".join(tokens).split()
        if self.delexicalize_number:
            tokens = ["numeric" if tok.isdigit() else tok for tok in tokens]
            tokens = self.merge_phone_number(tokens)
        text = " ".join(self.remove_duplicate_items(tokens, "numeric"))
        if self.delexicalize_location:
            for loc in self.places:
                loc = " ".join(loc)
                text = re.sub(rf"(^|\s){loc}(?=($|\s))", r"\1loc", text)
        text = " ".join(self.merge_loc_and_numeric(text.split()))
        if self.delexicalize_name:
            for name in self.names:
                name = " ".join(name)
                text = re.sub(rf"(^|\s){name}(?=($|\s))", r"\1someplace", text)
        tokens = text.split()
        tokens = self.remove_duplicate_items(tokens, ".")
        tokens = self.remove_duplicate_items(tokens, ",")
        tokens = self.remove_duplicate_items(tokens, "?")
        tokens = self.remove_duplicate_items(tokens, "_")
        tokens = ["something" if token == "_" else token for token in tokens]
        tokens = ["," if token == "-" else token for token in tokens]
        tokens = ["is" if token == ":" else token for token in tokens]
        tokens = ["." if token == ";" else token for token in tokens]
        ret = " ".join(tokens)
        return ret

    def norm_sv(self, sv: str):
        sv = sv.lower().strip()
        if self.normalize_slot_values:
            return self.slot_norm_map.get(sv, sv)
        return sv

    def load_turn_label(self, data) -> DialogState:
        state = DialogState()
        for slot, value in data:
            if slot == "request":
                act = "request"
                slot = "slot"
            else:
                act = "inform"
            slot, value = self.norm_sv(slot), self.norm_sv(value)
            state.add(ActSlotValue(act, slot, value))
        return state

    def load_belief_state(self, data) -> DialogState:
        state = DialogState()
        for d in data:
            act = d["act"]
            for sv in d["slots"]:
                slot, value = map(self.norm_sv, sv)
                state.add(ActSlotValue(act, slot, value))
        return state

    def load_system_acts(self, data) -> DialogState:
        state = DialogState()
        for d in data:
            if isinstance(d, str):
                act = "request"
                slot = "slot"
                value = d
            elif isinstance(d, list):
                assert len(d) == 2
                act = "inform"
                slot, value = d
            else:
                raise TypeError(f"unsupported system act type: {d}")
            slot, value = self.norm_sv(slot), self.norm_sv(value)
            state.add(ActSlotValue(act, slot, value))
        return state

    def load_json(self, path: pathlib.Path) -> Sequence[Dialog]:
        data = utils.load_yaml(path)
        ret = []
        for dialog in data:
            turns = []
            for turn in dialog["dialogue"]:
                if turn["system_transcript"]:
                    turns.append(Turn(
                        text=self.normalize_sent(turn["system_transcript"]),
                        speaker="wizard",
                        state=self.load_system_acts(turn["system_acts"]),
                        meta={
                            "transcript": turn["system_transcript"]
                        }
                    ))
                # We discard belief state because we can infer the goal
                # by accumulating turn-specific goal labels.
                # The following line has been disabled.
                # goal = self.load_belief_state(turn["belief_state"])
                transcript = (max(turn["asr"], key=lambda x: x[1])[0]
                              if "asr" in turn else turn["transcript"])
                turns.append(Turn(
                    text=self.normalize_sent(transcript),
                    speaker="user",
                    state=self.load_turn_label(turn["turn_label"]),
                    meta={
                        "transcript": transcript
                    }
                ))
            dst_dialog = DSTDialog.from_dialog(Dialog(turns, meta={
                "dialogue_idx": dialog["dialogue_idx"]
            }), force=True)
            dst_dialog.compute_user_goals()
            ret.append(dst_dialog)
        return ret

    def load(self, path: pathlib.Path, split: str = None
             ) -> Mapping[str, Sequence[Dialog]]:
        """Load WoZ2.0 Dataset from the root directory. The root directory
        must contain `train.json`, `dev.json`, and `test.json` files"""
        return {split: self.load_json(path.joinpath(f"{split}.json"))
                for split in (("train", "dev", "test")
                              if split is None else [split])}

    @staticmethod
    def serialize_turn_label(state: DialogState) -> list:
        ret = []
        if "inform" in state:
            for sv in state["inform"]:
                ret.append([sv.slot, sv.value])
        if "request" in state:
            for sv in state["request"]:
                ret.append(["request", sv.value])
        return ret

    @staticmethod
    def serialize_system_acts(state: DialogState) -> list:
        ret = []
        if "inform" in state:
            for sv in state["inform"]:
                ret.append([sv.slot, sv.value])
        if "request" in state:
            for sv in state["request"]:
                ret.append(sv.value)
        return ret

    @staticmethod
    def serialize_turn(wizard: DSTWizardTurn, user: DSTUserTurn) -> dict:
        return {
            "system_transcript": wizard.text,
            "system_acts": WoZAdapter.serialize_system_acts(wizard.state),
            "asr": [[user.text, 1.0]],
            "transcript": user.text,
            "turn_label": WoZAdapter.serialize_turn_label(user.state),
        }

    @staticmethod
    def serialize_bs(data: dict) -> dict:
        state = DialogState()
        for k, v in data.items():
            if k == "request":
                asv = ActSlotValue("request", "slot", v)
            else:
                asv = ActSlotValue("inform", k, v)
            state.add(asv)
        return WoZAdapter.serialize_semantics(state)

    @staticmethod
    def serialize_dialog(dialog: DSTDialog, idx: int = None) -> dict:
        turns = []
        belief_state = dict()
        for turn_idx, turn in enumerate(dialog.dst_turns):
            wizard, user = turn.wizard, turn.user
            turn = WoZAdapter.serialize_turn(wizard, user)
            turn["turn_idx"] = turn_idx
            for asv in user.state:
                if not (asv.act == "inform" and asv.slot != "request"):
                    continue
                belief_state[asv.slot] = asv.value
            turn["belief_state"] = WoZAdapter.serialize_bs(belief_state)
            turns.append(turn)
        return {
            "dialogue": turns,
            "dialogue_idx": dialog.meta.get("dialogue_idx", idx) or idx
        }

    def save_json(self, data: Sequence[Dialog], path: pathlib.Path):
        json = []
        for idx, dialog in enumerate(data):
            if not isinstance(dialog, DSTDialog):
                dialog = DSTDialog.from_dialog(dialog)
            json.append(self.serialize_dialog(dialog, idx))
        utils.save_json(json, path)

    def save_imp(self, dat: Mapping[str, Sequence[Dialog]], path: pathlib.Path):
        shell = utils.ShellUtils()
        shell.mkdir(path, True)
        for split in ("train", "dev", "test"):
            self.save_json(dat[split], path.joinpath(f"{split}.json"))
