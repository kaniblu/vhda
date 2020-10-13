__all__ = ["Glad"]

import sys
import time
import random
import string
import logging
import logging.config
import pathlib
import tempfile
import itertools
from dataclasses import dataclass, field
from typing import Union, Sequence, Tuple, Optional

import yaap

import utils
from datasets import Dialog


@dataclass
class Glad:
    train_data: Sequence[Dialog] = field(repr=False, compare=False, hash=False)
    valid_data: Sequence[Dialog] = field(repr=False, compare=False, hash=False)
    test_data: Sequence[Dialog] = field(repr=False, compare=False, hash=False)
    glad_dir: Union[str, pathlib.Path] = \
        pathlib.Path(__file__).parent.joinpath("../dst/faster-glad").absolute()
    gpu: int = None
    model: str = "glad"
    exp_dir: str = "exp"
    exp_name: Optional[str] = None
    batch_size: int = 50
    epochs: int = 50
    early_stop_criterion: str = "joint_goal"
    emb_dropout: float = 0.2
    local_dropout: float = 0.2
    global_dropout: float = 0.2
    seed: int = None
    save_dir: Optional[pathlib.Path] = None
    _shell: utils.ShellUtils = field(init=False,
                                     default_factory=utils.ShellUtils)
    _tmp_dir: pathlib.Path = None

    def __post_init__(self):
        self.glad_dir = pathlib.Path(self.glad_dir)
        if self.seed is None:
            self.seed = int(time.time() * 1000) % 0xFFFFFFFF
        if self.exp_name is None:
            self.exp_name = "".join(random.choice(string.ascii_letters)
                                    for _ in range(10))
        if self._tmp_dir is None:
            self._tmp_dir = pathlib.Path(tempfile.mkdtemp())
        if self.save_dir is not None:
            if (self.save_dir.exists() and
                    utils.has_element(self.save_dir.glob("*"))):
                raise FileExistsError(f"save dir not empty: {self.save_dir}")
            self._shell.mkdir(self.save_dir, silent=True)

    @property
    def root_dir(self):
        return self._tmp_dir.joinpath("glad")

    @property
    def python_bin(self):
        return sys.executable

    @property
    def data_dir(self):
        return self.root_dir.joinpath("data/woz")

    @property
    def raw_dir(self):
        return self.data_dir.joinpath("raw")

    @property
    def instance_dir(self):
        return (self.root_dir.joinpath(self.exp_dir)
                .joinpath(self.model).joinpath(self.exp_name))

    def save_raw(self):
        self._shell.mkdir(self.raw_dir, True)
        paths = {split: str(self.raw_dir.joinpath(f"{split}.json"))
                 for split in ("train", "dev", "test")}
        save_woz(self.train_data, paths["train"], True)
        save_woz(self.valid_data, paths["dev"], True)
        save_woz(self.test_data, paths["test"], True)

    def prepare_data(self):
        process = utils.Process(
            args=(self.python_bin, "preprocess_data.py"),
            cwd=str(self.root_dir),
            aux_env=dict(PYTHONPATH="", PYTHONUNBUFFERED="1"),
            print_stdout=True,
            print_stderr=True
        )
        exit_code, stdout, stderr = process.run()
        if exit_code:
            raise utils.ProcessError(
                f"exit code: {exit_code}; "
                f"msg: {stderr}"
            )

    def train(self):
        kwargs = dict(
            dexp=self.exp_dir,
            model=self.model,
            nick=self.exp_name,
            epoch=self.epochs,
            batch_size=self.batch_size,
            stop=self.early_stop_criterion,
            seed=self.seed,
        )
        if self.gpu is not None:
            kwargs["gpu"] = self.gpu
        args = ((self.python_bin, "train.py") +
                tuple(itertools.chain(*((f"--{k}", str(v))
                                        for k, v in kwargs.items()))))
        args += ("--dropout",)
        for key, dropout in zip(("emb", "local", "global"),
                                (self.emb_dropout, self.local_dropout,
                                 self.global_dropout)):
            args += (f"{key}={dropout:.4f}",)
        process = utils.Process(
            args=args,
            cwd=str(self.root_dir),
            aux_env=dict(PYTHONPATH="", PYTHONUNBUFFERED="1"),
            print_stdout=True,
            print_stderr=True
        )
        exit_code, stdout, stderr = process.run()
        if exit_code:
            raise utils.ProcessError(
                f"exit code: {exit_code}; "
                f"msg: {stderr}"
            )

    def test(self) -> Tuple[Sequence, dict]:
        temp_path = tempfile.mktemp()
        kwargs = dict(
            split="test",
            fout=temp_path
        )
        if self.gpu is not None:
            kwargs["gpu"] = self.gpu
        args = (self.python_bin, "evaluate.py", self.instance_dir)
        args += tuple(itertools.chain(*((f"--{k}", str(v))
                                        for k, v in kwargs.items())))
        process = utils.Process(
            args=args,
            cwd=str(self.root_dir),
            aux_env=dict(PYTHONPATH="", PYTHONUNBUFFERED="1"),
            print_stdout=True,
            print_stderr=True
        )
        exit_code, stdout, stderr = process.run()
        if exit_code:
            raise utils.ProcessError(
                f"exit code: {exit_code}; "
                f"msg: {stderr}"
            )
        lines = stdout.splitlines()
        res = None
        for i in range(1, len(lines) + 1):
            if "{" in lines[-i]:
                res = eval("\n".join(lines[-i:]))
                break
        pred = utils.load_json(temp_path)
        return pred, res

    def clean(self):
        self._shell.remove(self._tmp_dir, recursive=True, silent=True)

    def create_instance(self):
        self._shell.mkdir(self._tmp_dir, silent=True)
        self._shell.copy_dir(src=self.glad_dir,
                             dst=self.root_dir,
                             exclude=[
                                 self.glad_dir.joinpath("data"),
                                 self.glad_dir.joinpath("exp")
                             ],
                             overwrite=True)

    def backup(self):
        if self.save_dir is None:
            raise RuntimeError(f"no save dir specified for backup")
        backup_files = ("train.log", "config.json")
        for fname in backup_files:
            src = self.instance_dir.joinpath(fname)
            if not src.exists():
                raise FileNotFoundError(f"{fname} file not found at "
                                        f"instance dir: {src}")
            self._shell.copy(src, self.save_dir.joinpath(fname))

    def run_all(self) -> Tuple[Sequence, dict]:
        self.clean()
        self.create_instance()
        self.save_raw()
        self.prepare_data()
        self.train()
        ret = self.test()
        if self.save_dir is not None:
            self.backup()
        self.clean()
        return ret


def try_path(path: pathlib.Path):
    if path.exists():
        return path
    else:
        return


def main(args):
    if args.logging_config is not None:
        logging.config.dictConfig(utils.load_yaml(args.logging_config))
    logger = logging.getLogger("GladRunner")
    save_dir = pathlib.Path(args.save_dir)
    shell = utils.ShellUtils()
    shell.mkdir(save_dir, silent=True)
    logger.info(f"running glad...")
    if args.data_format == "woz":
        load_fn = load_woz
    elif args.data_format == "json":
        load_fn = utils.chain_func(
            lambda x: list(map(Dialog.from_json, x)),
            utils.load_json
        )
    elif args.data_format == "dstc":
        load_fn = load_dstc2
    else:
        raise ValueError(f"unsupported data type: {args.data_type}")
    data_dir = pathlib.Path(args.data_dir)
    train_data = load_fn(str(data_dir.joinpath("train.json")))
    valid_data = load_fn(str(try_path(data_dir.joinpath("valid.json")) or
                             data_dir.joinpath("dev.json")))
    test_data = load_fn(str(data_dir.joinpath("test.json")))
    glad = Glad(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        glad_dir=args.glad_dir,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        emb_dropout=args.emb_dropout,
        local_dropout=args.local_dropout,
        global_dropout=args.global_dropout,
        save_dir=save_dir.joinpath("exp"),
        seed=args.seed,
        early_stop_criterion=args.early_stop_criterion,
        gpu=args.gpu
    )
    pred, res = glad.run_all()
    logger.info("saving results...")
    utils.save_json(pred, str(save_dir.joinpath("pred.json")))
    utils.save_json(res, str(save_dir.joinpath("eval.json")))
    logger.info("done!")


def create_parser():
    parser = yaap.Yaap()
    parser.add_pth("logging-config", must_exist=True,
                   default=(pathlib.Path(__file__).parent
                            .joinpath("../examples/logging.yml")),
                   help="Path to logging configuration file.")
    parser.add_pth("data-dir", is_dir=True, must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("examples")),
                   help="Path to the data dir. Must contain 'train.json', "
                        "'valid.json' and 'test.json'.")
    parser.add_str("data-format", default="json",
                   choices=("woz", "json", "dstc"),
                   help="Data format of the data to be loaded.")
    parser.add_pth("glad-dir", is_dir=True, must_exist=True,
                   default=(pathlib.Path(__file__).parent
                            .joinpath("../dst/glad").absolute()),
                   help="Directory to an existing glad codebase.")
    parser.add_pth("save-dir", is_dir=True, default="out-glad",
                   help="Directory for saving output files.")
    parser.add_int("max-epochs", min_bound=1, default=50,
                   help="Maximum epochs to train models.")
    parser.add_int("batch-size", min_bound=1, default=50,
                   help="Mini-batch size during stochastic gd.")
    parser.add_flt("emb-dropout", default=0.2,
                   help="Embedding dropout.")
    parser.add_flt("local-dropout", default=0.2,
                   help="Local dropout.")
    parser.add_flt("global-dropout", default=0.2,
                   help="Global dropout.")
    parser.add_str("early-stop-criterion", default="joint_goal",
                   choices=("joint_goal", "turn_inform",
                            "turn_request", "hmean"))
    parser.add_int("seed",
                   help="Random seed.")
    parser.add_int("gpu", help="Index of specific GPU device to use.")
    return parser


if __name__ == "__main__":
    main(utils.parse_args(create_parser()))
