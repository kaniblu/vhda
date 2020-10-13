__all__ = ["main"]

import random
import pathlib
import logging
import itertools
from dataclasses import dataclass
from typing import Sequence

import yaap
import torch
import torch.utils.data as td
import torchmodels

import utils
import models
import datasets
from datasets import BatchData
from datasets import Dialog
from datasets import DialogProcessor


@dataclass
class InterpolateInferencer:
    model: models.AbstractTDA
    processor: DialogProcessor
    device: torch.device = torch.device("cpu")
    asv_tensor: utils.Stacked1DTensor = None
    _num_instances: int = utils.private_field(default=None)

    def __post_init__(self):
        if self.asv_tensor is None:
            self.asv_tensor = self.processor.tensorize_state_vocab("goal_state")
        self.asv_tensor = self.asv_tensor.to(self.device)

    def prepare_data_batch(self, batch: BatchData) -> dict:
        return {
            "conv_lens": batch.conv_lens,
            "sent": batch.sent.value,
            "sent_lens": batch.sent.lens1,
            "speaker": batch.speaker.value,
            "goal": batch.goal.value,
            "goal_lens": batch.goal.lens1,
            "state": batch.state.value,
            "state_lens": batch.state.lens1,
            "asv": self.asv_tensor.value,
            "asv_lens": self.asv_tensor.lens
        }

    def prepare_z_batch(self, batch: torch.Tensor) -> dict:
        return {
            "zconv": batch,
            "asv": self.asv_tensor.value,
            "asv_lens": self.asv_tensor.lens
        }

    def encode(self, dataloader) -> torch.Tensor:
        self.model.eval()
        self.model.encode()
        zconv = []
        for batch in dataloader:
            batch = batch.to(self.device)
            zconv.append(self.model(self.prepare_data_batch(batch)).mu)
        return torch.cat(zconv, 0)

    def generate(self, dataloader) -> Sequence[Dialog]:
        self.model.eval()
        self.model.decode_optimal()
        dialogs = []
        for batch in dataloader:
            batch = batch.to(self.device)
            pred, _ = self.model(
                self.prepare_z_batch(batch),
                spkr_scale=0.0,
                goal_scale=1.0,
                state_scale=0.0,
                sent_scale=1.0
            )
            dialogs.extend(map(self.processor.lexicalize_global, pred))
        return dialogs


def create_parser():
    parser = yaap.Yaap(
        desc="Create z-interpolation between two random data points"
    )
    # data options
    parser.add_pth("data-dir", is_dir=True, must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("tests/data/json")),
                   help="Path to the data dir. Must contain 'train.json' and "
                        "'dev.json'.")
    parser.add_str("splits", is_list=True, default=("train",),
                   choices=("train", "dev", "test"),
                   help="List of splits to evaluate on.")
    parser.add_pth("processor-path", required=True, must_exist=True,
                   help="Path to the processor pickle file.")
    parser.add_str("anchor1", regex=r"(train|dev|test)-\d+",
                   help="Data index of the first anchor. If not provided, "
                        "a random data point will be chosen.")
    parser.add_str("anchor2", regex=r"(train|dev|test)-\d+",
                   help="Data index of the second anchor. If not provided, "
                        "a random data point will be chosen.")
    # interpolation options
    parser.add_int("steps", default=10,
                   help="Number of intermediate steps between two data points.")
    # model options
    parser.add_pth("model-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/vhda-mini.yml")),
                   help="Path to the model configuration file.")
    parser.add_pth("ckpt-path", required=True, must_exist=True,
                   help="Path to the model checkpoint.")
    parser.add_int("gpu", min_bound=0,
                   help="GPU device to use. (e.g. 0, 1, etc.)")
    # display options
    parser.add_pth("logging-config", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/logging.yml")),
                   help="Path to a logging config file (yaml/json).")
    parser.add_pth("save-dir", default="out", is_dir=True,
                   help="Directory to save output files.")
    parser.add_bol("overwrite", help="Whether to overwrite save dir.")
    return parser


def sample_data(data, idx: str = None):
    if idx is None:
        return random.choice(list(itertools.chain(*data.values())))
    split, idx = idx.split("-")
    idx = int(idx)
    return data[split][idx]


def main(args=None):
    args = utils.parse_args(create_parser(), args)
    if args.logging_config is not None:
        logging.config.dictConfig(utils.load_yaml(args.logging_config))
    save_dir = pathlib.Path(args.save_dir)
    if (not args.overwrite and
            save_dir.exists() and utils.has_element(save_dir.glob("*.json"))):
        raise FileExistsError(f"save directory ({save_dir}) is not empty")
    shell = utils.ShellUtils()
    shell.mkdir(save_dir, silent=True)
    logger = logging.getLogger("interpolate")
    data_dir = pathlib.Path(args.data_dir)
    data = {
        split: list(map(Dialog.from_json,
                        utils.load_json(data_dir.joinpath(f"{split}.json"))))
        for split in set(args.splits)
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
    samples = (
        sample_data(data, args.anchor1),
        sample_data(data, args.anchor2)
    )
    formatter = utils.DialogTableFormatter()
    logger.info(f"first sample: \n{formatter.format(samples[0])}")
    logger.info(f"second sample: \n{formatter.format(samples[1])}")
    logger.info("preparing environment...")
    dataloader = datasets.create_dataloader(
        dataset=datasets.DialogDataset(
            data=samples,
            processor=processor
        ),
        batch_size=1,
        shuffle=False,
        pin_memory=False
    )
    inferencer = InterpolateInferencer(
        model=model,
        processor=processor,
        device=device
    )
    logger.info("interpolating...")
    with torch.no_grad():
        zconv_a, zconv_b = inferencer.encode(dataloader)
        zconv = torch.stack([zconv_a + (zconv_b - zconv_a) / args.steps * i
                             for i in range(args.steps + 1)])
        gen_samples = inferencer.generate(td.DataLoader(zconv, shuffle=False))
    # use original data points for two extremes
    samples = [samples[0]] + list(gen_samples[1:-1]) + [samples[1]]
    logger.info("interpolation results: ")
    for i, sample in enumerate(samples):
        logger.info(f"interpolation step {i / args.steps:.2%}: \n"
                    f"{formatter.format(sample)}")
    logger.info("saving results...")
    json_dir = save_dir.joinpath("json")
    json_dir.mkdir(exist_ok=True)
    for i, sample in enumerate(samples, 1):
        utils.save_json(sample.to_json(), json_dir.joinpath(f"{i:02d}.json"))
    tbl_dir = save_dir.joinpath("table")
    tbl_dir.mkdir(exist_ok=True)
    for i, sample in enumerate(samples, 1):
        utils.save_lines([formatter.format(sample)],
                         tbl_dir.joinpath(f"{i:02d}.txt"))
    ltx_dir = save_dir.joinpath("latex")
    ltx_dir.mkdir(exist_ok=True)
    ltx_formatter = utils.DialogICMLLatexFormatter()
    for i, sample in enumerate(samples, 1):
        utils.save_lines([ltx_formatter.format(sample)],
                         ltx_dir.joinpath(f"{i:02d}.tex"))
    logger.info("done!")


if __name__ == "__main__":
    main()
