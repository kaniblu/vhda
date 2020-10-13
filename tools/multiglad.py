__all__ = ["multi_glad", "MultiGladArguments"]

import logging
import logging.config
import pathlib
from dataclasses import dataclass
from typing import Sequence, Optional

import yaap
import inflect

import utils
import datasets
from datasets import Dialog
from .glad import Glad
from .reduce_json import reduce_json


@dataclass
class MultiGladArguments(utils.Arguments):
    train_data: Sequence[Dialog]
    valid_data: Sequence[Dialog]
    test_data: Sequence[Dialog]
    glad_dir: pathlib.Path = pathlib.Path(__file__).parent.joinpath("dst/glad")
    save_dir: pathlib.Path = pathlib.Path("out-dst")
    max_epochs: int = 50
    batch_size: int = 50
    emb_dropout: float = 0.0
    local_dropout: float = 0.2
    global_dropout: float = 0.2
    early_stop_criterion: str = "joint_goal"
    runs: int = 10
    gpu: Optional[int] = None


def multi_glad(args: MultiGladArguments) -> dict:
    save_dir = args.save_dir
    shell = utils.ShellUtils()
    shell.mkdir(save_dir, silent=True)
    logger = logging.getLogger("MultiGlad")
    engine = inflect.engine()
    for i in range(1, args.runs + 1):
        logger.info(f"[{i:02d}] running glad {engine.ordinal(i)} time...")
        trial_dir = save_dir.joinpath(f"trial-{i:02d}")
        shell.mkdir(trial_dir, silent=True)
        glad = Glad(
            train_data=args.train_data,
            valid_data=args.valid_data,
            test_data=args.test_data,
            glad_dir=args.glad_dir,
            save_dir=trial_dir.joinpath("exp"),
            epochs=args.max_epochs,
            batch_size=args.batch_size,
            emb_dropout=args.emb_dropout,
            local_dropout=args.local_dropout,
            global_dropout=args.global_dropout,
            early_stop_criterion=args.early_stop_criterion,
            gpu=args.gpu
        )
        pred, res = glad.run_all()
        logger.info(f"[{i:02d}] saving results...")
        utils.save_json(pred, str(trial_dir.joinpath("pred.json")))
        utils.save_json(res, str(trial_dir.joinpath("eval.json")))
        logger.info(f"[{i:02d}] {engine.ordinal(i)} glad run finished.")
    logger.info(f"aggregating results...")
    result_paths = (str(save_dir.joinpath(f"trial-{i:02d}/eval.json"))
                    for i in range(1, args.runs + 1))
    results = list(map(utils.load_yaml, result_paths))
    summary = reduce_json(results)
    utils.save_json(summary, str(save_dir.joinpath("eval-summary.json")))
    utils.save_json(args.to_json(), save_dir.joinpath("args.json"))
    logger.info(f"done!")
    return summary


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
    parser.add_int("runs", default=10,
                   help="Number of runs to execute and aggregate.")
    parser.add_int("gpu", help="Index of specific GPU device to use.")
    return parser


def try_path(path: pathlib.Path):
    if path.exists():
        return path
    else:
        return


def main():
    args = utils.parse_args(create_parser())
    if args.logging_config is not None:
        logging.config.dictConfig(utils.load_yaml(args.logging_config))
    logger = logging.getLogger(__file__)
    logger.info("loading data...")
    if args.data_format == "woz":
        load_fn = datasets.load_woz
    elif args.data_format == "json":
        load_fn = utils.chain_func(
            lambda data: list(map(Dialog.from_json, data)),
            utils.load_json
        )
    else:
        raise ValueError(f"unsupported data format: {args.data_format}")
    data_dir = pathlib.Path(args.data_dir)
    train_data = load_fn(str(data_dir.joinpath("train.json")))
    valid_data = load_fn(str(try_path(data_dir.joinpath("valid.json")) or
                             data_dir.joinpath("dev.json")))
    test_data = load_fn(str(data_dir.joinpath("test.json")))
    multi_glad(MultiGladArguments(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        glad_dir=pathlib.Path(args.glad_dir),
        save_dir=pathlib.Path(args.save_dir),
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        emb_dropout=args.emb_dropout,
        local_dropout=args.local_dropout,
        global_dropout=args.global_dropout,
        early_stop_criterion=args.early_stop_criterion,
        runs=args.runs,
        gpu=args.gpu
    ))


if __name__ == "__main__":
    main()
