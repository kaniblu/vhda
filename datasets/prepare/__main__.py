import logging
import logging.config
import pathlib
import random
import itertools
from dataclasses import dataclass, field
from typing import ClassVar, Mapping, Sequence, Iterable, Set

import yaap
import torch
from tqdm import tqdm

import utils
from . import woz
from .preprocess import DialogPreprocessor
from ..common import Dialog
from ..common import Turn

SPLIT_MAP = {
    "train": "train",
    "valid": "dev",
    "val": "dev",
    "dev": "dev",
    "test": "test"
}


def split_data(data, ratios):
    ratios = torch.tensor(list(ratios))
    ratios = (ratios / ratios.sum()).cumsum(0)
    indices = [int(round(len(data) * r)) for r in ratios.tolist()]
    randidx = torch.randperm(len(data)).tolist()
    return tuple([data[randidx[k]] for k in range(i, j)]
                 for i, j in zip([0] + indices, indices))


@dataclass
class DataPreparer:
    preprocessor: DialogPreprocessor
    woz_adapter: woz.WoZAdapter = field(default_factory=woz.WoZAdapter)
    logger: logging.Logger = utils.private_field(default=None)
    shell: utils.ShellUtils = \
        utils.private_field(default_factory=utils.ShellUtils)
    split_map: ClassVar[Mapping[str, str]] = {
        "train": "train",
        "valid": "dev",
        "val": "dev",
        "dev": "dev",
        "test": "test"
    }

    def __post_init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def extract_asv_ontology(turns: Iterable[Turn]) -> Mapping[str, Set]:
        asv_set, act_set, slot_set, value_set = set(), set(), set(), set()
        for turn in turns:
            for asv in turn.state:
                asv_set.add(str(asv))
                act_set.add(asv.act)
                slot_set.add(asv.slot)
                value_set.add(asv.value)
        return {
            "asv": asv_set,
            "act": act_set,
            "slot": slot_set,
            "value": value_set
        }

    def save_asv_ontology(self, dialogs: Sequence[Dialog], path: pathlib.Path):
        self.logger.info(f"saving asv ontology at {path}...")
        turns = [turn for dialog in dialogs for turn in dialog.turns]
        speakers = {turn.speaker for turn in turns}
        shell = utils.ShellUtils()
        for speaker in speakers:
            speaker_dir = path.joinpath(speaker)
            shell.mkdir(speaker_dir, silent=True)
            speaker_turns = [turn for turn in turns if turn.speaker == speaker]
            for key, ont in self.extract_asv_ontology(speaker_turns).items():
                file_path = speaker_dir.joinpath(f"{key}.txt")
                utils.save_lines(list(sorted(ont)), file_path)
        all_dir = path.joinpath("all")
        shell.mkdir(all_dir, silent=True)
        for key, ont in self.extract_asv_ontology(turns).items():
            file_path = all_dir.joinpath(f"{key}.txt")
            utils.save_lines(list(sorted(ont)), file_path)

    def save_asv_ontologies(self, data, path: pathlib.Path):
        self.logger.info(f"saving asv ontologies at {path}...")
        for split, dialogs in data.items():
            split_dir = path.joinpath(split)
            self.save_asv_ontology(dialogs, split_dir)
        self.save_asv_ontology(list(itertools.chain(*data.values())),
                               path.joinpath("all"))

    def save_utt_ontology(self, dialogs: Sequence[Dialog], path: pathlib.Path):
        self.logger.info(f"saving utt ontology at {path}...")
        turns = [turn for dialog in dialogs for turn in dialog.turns]
        speakers = {turn.speaker for turn in turns}
        shell = utils.ShellUtils()
        shell.mkdir(path, silent=True)
        for speaker in speakers:
            speaker_path = path.joinpath(f"{speaker}.txt")
            speaker_turns = [turn for turn in turns if turn.speaker == speaker]
            speaker_utts = {turn.text for turn in speaker_turns}
            utils.save_lines(list(sorted(speaker_utts)), speaker_path)
        all_path = path.joinpath("all.txt")
        utts = {turn.text for turn in turns}
        utils.save_lines(list(sorted(utts)), all_path)

    def save_utt_ontologies(self, data, path: pathlib.Path):
        self.logger.info(f"saving utt ontologies at {path}...")
        for split, dialogs in data.items():
            split_dir = path.joinpath(split)
            self.save_utt_ontology(dialogs, split_dir)
        self.save_utt_ontology(list(itertools.chain(*data.values())),
                               path.joinpath("all"))

    def save_asr_ontology(self, dialogs: Sequence[Dialog], path: pathlib.Path):
        self.logger.info(f"saving asr ontology at {path}...")
        turns = [turn for dialog in dialogs for turn in dialog.turns]
        speakers = {turn.speaker for turn in turns}
        shell = utils.ShellUtils()
        shell.mkdir(path, silent=True)
        for speaker in speakers:
            speaker_path = path.joinpath(f"{speaker}.txt")
            speaker_turns = [turn for turn in turns if turn.speaker == speaker]
            speaker_utts = {asr for turn in speaker_turns for asr in turn.asr}
            utils.save_lines(list(sorted(speaker_utts)), speaker_path)
        all_path = path.joinpath("all.txt")
        utts = {asr for turn in turns for asr in turn.asr}
        utils.save_lines(list(sorted(utts)), all_path)

    def save_asr_ontologies(self, data, path: pathlib.Path):
        self.logger.info(f"saving asr ontologies at {path}...")
        for split, dialogs in data.items():
            split_dir = path.joinpath(split)
            self.save_asr_ontology(dialogs, split_dir)
        self.save_asr_ontology(list(itertools.chain(*data.values())),
                               path.joinpath("all"))

    def save_word_ontology(self, dialogs: Sequence[Dialog], path: pathlib.Path):
        self.logger.info(f"saving utt ontology at {path}...")
        turns = [turn for dialog in dialogs for turn in dialog.turns]
        speakers = {turn.speaker for turn in turns}
        shell = utils.ShellUtils()
        shell.mkdir(path, silent=True)
        for speaker in speakers:
            speaker_path = path.joinpath(f"{speaker}.txt")
            speaker_turns = [turn for turn in turns if turn.speaker == speaker]
            speaker_utts = {turn.text for turn in speaker_turns}
            speaker_words = {word for utt in speaker_utts
                             for word in utt.split()}
            utils.save_lines(list(sorted(speaker_words)), speaker_path)
        all_path = path.joinpath("all.txt")
        words = {word for turn in turns for word in turn.text.split()}
        utils.save_lines(list(sorted(words)), all_path)

    def save_word_ontologies(self, data, path: pathlib.Path):
        self.logger.info(f"saving utt ontologies at {path}...")
        for split, dialogs in data.items():
            split_dir = path.joinpath(split)
            self.save_word_ontology(dialogs, split_dir)
        self.save_word_ontology(list(itertools.chain(*data.values())),
                                path.joinpath("all"))

    def save_asr_word_ontology(self, dialogs: Sequence[Dialog],
                               path: pathlib.Path):
        self.logger.info(f"saving utt ontology at {path}...")
        turns = [turn for dialog in dialogs for turn in dialog.turns]
        speakers = {turn.speaker for turn in turns}
        shell = utils.ShellUtils()
        shell.mkdir(path, silent=True)
        for speaker in speakers:
            speaker_path = path.joinpath(f"{speaker}.txt")
            speaker_turns = [turn for turn in turns if turn.speaker == speaker]
            speaker_words = {word for turn in speaker_turns
                             for asr in turn.asr for word in asr.split()}
            utils.save_lines(list(sorted(speaker_words)), speaker_path)
        all_path = path.joinpath("all.txt")
        words = {word for turn in turns
                 for asr in turn.asr for word in asr.split()}
        utils.save_lines(list(sorted(words)), all_path)

    def save_asr_word_ontologies(self, data, path: pathlib.Path):
        self.logger.info(f"saving utt ontologies at {path}...")
        for split, dialogs in data.items():
            split_dir = path.joinpath(split)
            self.save_asr_word_ontology(dialogs, split_dir)
        self.save_asr_word_ontology(list(itertools.chain(*data.values())),
                                    path.joinpath("all"))

    def save_ontologies(self, data, path: pathlib.Path):
        self.save_asv_ontologies(data, path.joinpath("asv"))
        self.save_utt_ontologies(data, path.joinpath("utt"))
        self.save_word_ontologies(data, path.joinpath("word"))
        self.save_asr_ontologies(data, path.joinpath("asr"))
        self.save_asr_word_ontologies(data, path.joinpath("asr-word"))

    def save_splits(self, data, path: pathlib.Path):
        self.logger.info("saving all splits...")
        original_dir = path.joinpath("original")
        preprocessed_dir = path.joinpath("preprocessed")
        self.shell.mkdir(original_dir, True)
        self.shell.mkdir(preprocessed_dir, True)
        for split, dialogs in data.items():
            split = SPLIT_MAP.get(split, split)
            utils.save_json([dialog.to_json() for dialog in dialogs],
                            original_dir.joinpath(f"{split}.json"))
            self.logger.info(f"preprocessing split '{split}'...")
            utils.save_json([dialog.to_json() for dialog in
                             map(self.preprocessor.preprocess, tqdm(dialogs))],
                            preprocessed_dir.joinpath(f"{split}.json"))

    def prepare_woz(self, gdrive, save_dir: pathlib.Path):
        self.logger.info("preparing WoZ2.0 dataset...")
        self.shell.mkdir(save_dir, silent=True)
        download_path = save_dir.joinpath("woz.tar.gz")
        self.shell.download_gdrive(gdrive, download_path, True)
        raw_dir, json_dir = save_dir.joinpath("raw"), save_dir.joinpath("json")
        self.shell.mkdir(json_dir, silent=True)
        self.shell.extract(download_path, raw_dir)
        data = self.woz_adapter.load(raw_dir)
        self.save_splits(data, json_dir)
        self.save_ontologies(data, save_dir.joinpath("ontology"))


def main(args):
    logging.config.dictConfig(utils.load_yaml(args.logging_config))
    data_dir = pathlib.Path(args.data_dir).absolute()
    shell = utils.ShellUtils()
    shell.mkdir(data_dir, silent=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    preparer = DataPreparer(
        preprocessor=DialogPreprocessor(
            lowercase=not args.disable_lowercase,
            replace_number=args.replace_number,
            tokenizer=args.tokenizer,
            special_chars=args.special_chars
        ),
        woz_adapter=woz.WoZAdapter(
            normalize_slot_values=not args.woz_disable_slot_norm
        )
    )
    for dataset in args.datasets:
        if dataset == "woz":
            preparer.prepare_woz(args.woz_gdrive, data_dir.joinpath("woz"))
        else:
            raise ValueError(f"unsupported dataset: {dataset}")


def create_parser():
    parser = yaap.Yaap()
    parser.add_pth("logging-config", must_exist=True,
                   default=(pathlib.Path(".").absolute()
                            .joinpath("tests/config/logging.yml")),
                   help="Path to logging configuration.")
    parser.add_pth("data-dir", is_dir=True,
                   default=pathlib.Path(".").absolute().joinpath("data"),
                   help="Directory for managing datasets.")
    parser.add_str("datasets", is_list=True,
                   choices=("woz",),
                   default=("woz",),
                   help="Specific datasets to prepare. If none provided, all "
                        "possible datasets will be prepared.")
    parser.add_str("woz-gdrive",
                   default="1cxmOQ68I68J9O_3vyb06vrdgAXAt8Ev0",
                   help="Google Drive id for WoZ2.0 dataset.")
    parser.add_int("seed", default=2019, help="Data seed.")
    # sentence preprocessing
    parser.add_bol("disable-lowercase",
                   help="Whether to convert sentences to lowercase.")
    parser.add_str("replace-number",
                   help="Replace numbers with the specified string.")
    parser.add_str("tokenizer", default="spacy",
                   choices=("corenlp", "space", "spacy"),
                   help="Sentences will be tokenized with the tokenization "
                        "method if specified.")
    parser.add_str("special-chars", default=".?'",
                   help="Valid special characters. Words that contain any "
                        "characters that are not part of the special "
                        "characters or not ascii letters will be discarded.")
    # dataset-specific settings
    parser.add_bol("woz-disable-slot-norm",
                   help="Whether to disable WoZ2.0 slot-value normalization.")
    return parser


if __name__ == "__main__":
    main(utils.parse_args(create_parser()))
