__all__ = ["evaluate", "EvaluateArugments", "main"]

import pprint
import logging
import logging.config
import pathlib
from dataclasses import dataclass
from typing import Sequence

import yaap
import torch
import torchmodels

import utils
import train
import models
import datasets
from datasets import Dialog
from datasets import DialogProcessor
from models import AbstractTDA
from loopers import EvaluatingInferencer
from loopers import LossInferencer
from loopers import LogInferencer
from loopers import Inferencer
from loopers import LogGenerator
from loopers import EvaluatingGenerator
from loopers import TDAGenerator
from loopers import Generator
from embeds import HDF5Embeddings
from embeds import GloveFormatEmbeddings
from embeds import BinaryEmbeddings
from evaluators import DialogStateEvaluator
from evaluators import EmbeddingEvaluator
from evaluators import BLEUEvaluator
from evaluators import RougeEvaluator
from evaluators import SentLengthEvaluator
from evaluators import PosteriorEvaluator
from evaluators import SpeakerEvaluator
from evaluators import DistinctEvaluator
from evaluators import DialogLengthEvaluator
from evaluators import WordEntropyEvaluator
from evaluators import LanguageNoveltyEvaluator
from evaluators import StateEntropyEvaluator
from evaluators import DistinctStateEvaluator
from evaluators import StateCountEvaluator
from evaluators import StateNoveltyEvaluator


@dataclass
class FinegrainedEvaluator(
    LogInferencer,
    EvaluatingInferencer,
    LossInferencer,
    Inferencer
):
    def on_batch_started(self, batch):
        ret = super().on_batch_started(batch)
        self.model.eval()
        return ret


@dataclass
class DialogGenerationEvaluator(
    LogGenerator,
    EvaluatingGenerator,
    TDAGenerator,
    Generator
):
    pass


@dataclass
class EvaluateArugments(utils.Arguments):
    model: AbstractTDA
    train_data: Sequence[Dialog]
    test_data: Sequence[Dialog]
    processor: DialogProcessor
    embed_type: str = "glove"
    embed_path: pathlib.Path = \
        (pathlib.Path(__file__).absolute()
         .joinpath("tests/data/glove/glove.840B.300d.woz.txt"))
    device: torch.device = torch.device("cpu")
    batch_size: int = 32
    beam_size: int = 4
    max_conv_len: int = 20
    max_sent_len: int = 30


def evaluate(args: EvaluateArugments) -> utils.TensorMap:
    model, device = args.model, args.device
    logger = logging.getLogger("evaluate")
    dataset = datasets.DialogDataset(args.test_data, args.processor)
    logger.info("preparing evaluation environment...")
    loss = train.create_loss(
        model=model,
        vocabs=args.processor.vocabs,
        enable_kl=True,
        kl_mode="kl-mi+"
    )
    runners = {
        "fine": FinegrainedEvaluator(
            model=model,
            processor=args.processor,
            device=args.device,
            loss=loss,
            evaluators=[
                PosteriorEvaluator(),
                SpeakerEvaluator(args.processor.vocabs.speaker),
                DialogStateEvaluator(args.processor.vocabs)
            ],
            run_end_report=False
        ),
        "dial": DialogGenerationEvaluator(
            model=model,
            processor=args.processor,
            batch_size=args.batch_size,
            device=args.device,
            beam_size=args.beam_size,
            max_conv_len=args.max_conv_len,
            max_sent_len=args.max_sent_len,
            evaluators=[
                BLEUEvaluator(args.processor.vocabs),
                DistinctEvaluator(args.processor.vocabs),
                EmbeddingEvaluator(
                    vocab=args.processor.vocabs.word,
                    embeds=dict(
                        glove=GloveFormatEmbeddings,
                        hdf5=HDF5Embeddings,
                        bin=BinaryEmbeddings
                    )[args.embed_type](args.embed_path).preload()
                ),
                SentLengthEvaluator(args.processor.vocabs),
                RougeEvaluator(args.processor.vocabs),
                DialogLengthEvaluator(),
                WordEntropyEvaluator(
                    datasets.DialogDataset(args.train_data, args.processor)
                ),
                LanguageNoveltyEvaluator(
                    datasets.DialogDataset(args.train_data, args.processor)
                ),
                StateEntropyEvaluator(
                    datasets.DialogDataset(args.train_data, args.processor)
                ),
                StateCountEvaluator(args.processor.vocabs),
                DistinctStateEvaluator(args.processor.vocabs),
                StateNoveltyEvaluator(
                    datasets.DialogDataset(args.train_data, args.processor)
                )
            ],
            run_end_report=False
        ),
        # missing: 1-turn and 3-turn generation
    }
    logger.info("commencing evaluation...")
    eval_stats = dict()
    model.eval()
    for name, runner in runners.items():
        if isinstance(runner, Inferencer):
            stats = runner(datasets.create_dataloader(
                dataset=dataset,
                batch_size=args.batch_size,
                drop_last=False,
                shuffle=False
            ))
        elif isinstance(runner, Generator):
            stats = runner(dataset.data)[1]
        else:
            raise TypeError(f"unsupported runner type: {type(runner)}")
        eval_stats.update({k: v.detach().cpu().item()
                           for k, v in stats.items()})
    logger.info(f"evaluation summary: {pprint.pformat(eval_stats)}")
    return eval_stats


def create_parser():
    parser = yaap.Yaap()
    # data options
    parser.add_pth("data-dir", is_dir=True, must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("tests/data/json")),
                   help="Path to the data dir. Must contain 'train.json' and "
                        "'dev.json'.")
    parser.add_str("eval-splits", is_list=True,
                   default=("train", "dev", "test"),
                   choices=("train", "dev", "test"),
                   help="List of splits to evaluate on.")
    parser.add_pth("processor-path", required=True, must_exist=True,
                   help="Path to the processor pickle file.")
    # model options
    parser.add_pth("model-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/vhda-mini.yml")),
                   help="Path to the model configuration file.")
    parser.add_pth("ckpt-path", required=True, must_exist=True,
                   help="Path to the model checkpoint.")
    parser.add_int("gpu", min_bound=0,
                   help="GPU device to use. (e.g. 0, 1, etc.)")
    # embedding evaluation options
    parser.add_str("embed-type", default="glove",
                   choices=("glove", "bin", "hdf5"),
                   help="Type of embedding to load for evaluation.")
    parser.add_pth("embed-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("tests/data/glove/"
                                      "glove.840B.300d.woz.txt")),
                   help="Path to embedding file for evaluation.")
    # input/output options
    parser.add_pth("logging-config", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/logging.yml")),
                   help="Path to a logging config file (yaml/json).")
    parser.add_pth("save-dir", default="out",
                   help="Path to save evaluation results.")
    parser.add_bol("overwrite", help="Whether to overwrite save dir.")
    # inference options
    parser.add_int("batch-size", default=32,
                   help="Mini-batch size.")
    parser.add_int("beam-size", default=4)
    parser.add_int("max-conv-len", default=20)
    parser.add_int("max-sent-len", default=30)
    parser.add_int("seed", help="Random seed.")
    return parser


def main(args=None):
    args = utils.parse_args(create_parser(), args)
    if args.logging_config is not None:
        logging.config.dictConfig(utils.load_yaml(args.logging_config))
    save_dir = pathlib.Path(args.save_dir)
    shell = utils.ShellUtils()
    if (not args.overwrite and
            save_dir.exists() and utils.has_element(save_dir.glob("*.json"))):
        raise FileExistsError(f"save directory ({save_dir}) is not empty")
    shell.mkdir(save_dir, silent=True)
    logger = logging.getLogger("evaluate")
    utils.seed(args.seed)
    logger.info("loading data...")
    data_dir = pathlib.Path(args.data_dir)
    data = {
        split: list(map(Dialog.from_json,
                        utils.load_json(data_dir.joinpath(f"{split}.json"))))
        for split in (set(args.eval_splits) | {"train"})
    }
    processor: DialogProcessor = utils.load_pickle(args.processor_path)
    logger.info("preparing model...")
    torchmodels.register_packages(models)
    model_cls = torchmodels.create_model_cls(models, args.model_path)
    model: models.AbstractTDA = model_cls(processor.vocabs)
    model.reset_parameters()
    model.load_state_dict(torch.load(args.ckpt_path))
    device = torch.device("cpu")
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    model = model.to(device)
    for split in args.eval_splits:
        dialogs = data[split]
        logger.info(f"running evaluation on '{split}' split...")
        eval_args = EvaluateArugments(
            model=model,
            train_data=tuple(data["train"]),
            test_data=tuple(dialogs),
            processor=processor,
            embed_type=args.embed_type,
            embed_path=args.embed_path,
            device=device,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            max_conv_len=args.max_conv_len,
            max_sent_len=args.max_sent_len
        )
        utils.save_json(eval_args.to_json(),
                        save_dir.joinpath(f"eval-{split}-args.json"))
        with torch.no_grad():
            results = evaluate(eval_args)
        save_path = save_dir.joinpath(f"eval-{split}.json")
        logger.info(f"'{split}' results saved to {save_path}")
        utils.save_json(results, save_path)
    logger.info("done!")


if __name__ == "__main__":
    main()
