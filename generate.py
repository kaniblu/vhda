__all__ = ["generate", "GenerateArguments"]

import pathlib
import logging
import logging.config
from dataclasses import dataclass
from typing import Sequence, Optional

import yaap
import torch
import torchmodels

import utils
import models
import datasets
from loopers import Sample
from loopers import Generator
from loopers import LogGenerator
from loopers import TDAGenerator
from loopers import BeamSearchGenerator
from loopers import ValidatingGenerator
from datasets import Dialog


@dataclass
class BaseGenerator(LogGenerator, BeamSearchGenerator,
                    ValidatingGenerator, Generator):
    pass


@dataclass
class GenerateArguments(utils.Arguments):
    model: models.AbstractTDA
    processor: datasets.DialogProcessor
    data: Optional[Sequence[Dialog]] = None
    instances: Optional[int] = None
    batch_size: int = 32
    conv_scale: float = 1.0
    spkr_scale: float = 1.0
    goal_scale: float = 1.0
    state_scale: float = 1.0
    sent_scale: float = 1.0
    max_conv_len: int = 30
    max_sent_len: int = 30
    beam_size: int = 4
    validate_dst: bool = False
    validate_unique: bool = False
    device: torch.device = torch.device("cpu")


def generate(args: GenerateArguments) -> Sequence[Sample]:
    logger = logging.getLogger("generate")
    logger.info("preparing generation environment...")
    generator_cls = BaseGenerator
    generator_kwargs = dict(
        model=args.model,
        processor=args.processor,
        device=args.device,
        beam_size=args.beam_size,
        max_sent_len=args.max_sent_len,
        run_end_report=False
    )
    if isinstance(args.model, models.AbstractTDA):
        @dataclass
        class _TDAGenerator(TDAGenerator, generator_cls):
            pass

        generator_cls = _TDAGenerator
        generator_kwargs.update(dict(
            conv_scale=args.conv_scale,
            spkr_scale=args.spkr_scale,
            goal_scale=args.goal_scale,
            state_scale=args.state_scale,
            sent_scale=args.sent_scale,
            max_conv_len=args.max_conv_len
        ))

    def is_valid(sample):
        def is_valid_dst():
            try:
                datasets.DSTDialog.from_dialog(sample.output, force=False)
                return True
            except RuntimeError as e:
                return False

        def is_unique():
            return sample.input != sample.output

        return ((not args.validate_dst or is_valid_dst()) and
                (not args.validate_unique or is_unique()))

    generator_kwargs.update(dict(
        validator=is_valid
    ))
    generator = generator_cls(**generator_kwargs)
    logger.info("generating...")
    with torch.no_grad():
        samples, _ = generator(args.data, args.instances)
    return samples


def create_parser():
    parser = yaap.Yaap()
    # data options
    parser.add_pth("data-path", must_exist=True,
                   help="Path to the data. If not given, then the data "
                        "will be generated from the model's prior.")
    parser.add_pth("processor-path", must_exist=True, required=True,
                   help="Path to the processor pickle file.")
    # model options
    parser.add_pth("model-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/vhda-mini.yml")),
                   help="Path to the model configuration file.")
    parser.add_pth("ckpt-path", must_exist=True, required=True,
                   help="Path to the model checkpoint.")
    # model-specific options (TDA)
    parser.add_flt("conv-scale", default=1.0,
                   help="Scale to introduce into conv vector "
                        "for TDA generation.")
    parser.add_flt("spkr-scale", default=1.0,
                   help="Scale to introduce into spkr vector "
                        "for TDA generation.")
    parser.add_flt("goal-scale", default=1.0,
                   help="Scale to introduce into goal vector "
                        "for TDA generation.")
    parser.add_flt("state-scale", default=1.0,
                   help="Scale to introduce into state vector "
                        "for TDA generation.")
    parser.add_flt("sent-scale", default=1.0,
                   help="Scale to introduce into sent vector "
                        "for TDA generation.")
    # model-specific options (general)
    parser.add_int("beam-size", default=4,
                   help="Beam search beam size.")
    parser.add_int("max-sent-len", default=30,
                   help="Beam search maximum sentence length.")
    # generation options
    parser.add_int("batch-size", default=32,
                   help="Mini-batch size.")
    parser.add_bol("validate-dst",
                   help="Whether to validate generated samples "
                        "to be a valid dst dialogs.")
    parser.add_bol("validate-unique",
                   help="Whether to validate by checking uniqueness.")
    parser.add_int("instances",
                   help="Number of dialog instances to generate. "
                        "If not given, the same number of instances "
                        "as the data will be generated.")
    # misc options
    parser.add_pth("logging-config", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/logging.yml")),
                   help="Path to a logging config file (yaml/json).")
    parser.add_pth("save-dir", default=pathlib.Path("out"),
                   help="Directory to save output generation files.")
    parser.add_int("gpu", min_bound=0,
                   help="GPU device to use. (e.g. 0, 1, etc.)")
    parser.add_bol("overwrite", help="Whether to overwrite save dir.")
    parser.add_int("seed", help="Random seed.")
    return parser


def main():
    args = utils.parse_args(create_parser())
    if args.logging_config is not None:
        logging.config.dictConfig(utils.load_yaml(args.logging_config))
    save_dir = pathlib.Path(args.save_dir)
    if (not args.overwrite and
            save_dir.exists() and utils.has_element(save_dir.glob("*.json"))):
        raise FileExistsError(f"save directory ({save_dir}) is not empty")
    shell = utils.ShellUtils()
    shell.mkdir(save_dir, silent=True)
    logger = logging.getLogger("generate")
    utils.seed(args.seed)
    logger.info("loading data...")
    processor = utils.load_pickle(args.processor_path)
    data = None
    if args.data_path is not None:
        data = list(map(Dialog.from_json, utils.load_json(args.data_path)))
    logger.info("preparing model...")
    torchmodels.register_packages(models)
    model_cls = torchmodels.create_model_cls(models, args.model_path)
    model: models.AbstractTDA = model_cls(processor.vocabs)
    model.reset_parameters()
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    device = torch.device("cpu")
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    model = model.to(device)
    gen_args = GenerateArguments(
        model=model,
        processor=processor,
        data=data,
        instances=args.instances,
        batch_size=args.batch_size,
        conv_scale=args.conv_scale,
        spkr_scale=args.spkr_scale,
        goal_scale=args.goal_scale,
        state_scale=args.state_scale,
        sent_scale=args.sent_scale,
        validate_dst=args.validate_dst,
        validate_unique=args.validate_unique,
        device=device
    )
    utils.save_json(gen_args.to_json(), save_dir.joinpath("args.json"))
    with torch.no_grad():
        samples = generate(gen_args)
    utils.save_json([sample.output.to_json() for sample in samples],
                    save_dir.joinpath("gen-out.json"))
    utils.save_json([sample.input.to_json() for sample in samples],
                    save_dir.joinpath("gen-in.json"))
    utils.save_lines([str(sample.log_prob) for sample in samples],
                     save_dir.joinpath("logprob.txt"))
    logger.info("done!")


if __name__ == "__main__":
    main()
