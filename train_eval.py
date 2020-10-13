import pathlib
import itertools
import logging
import logging.config
from dataclasses import dataclass

import yaap
import torch
import torch.utils.tensorboard
import torchmodels

import utils
import train
import evaluate
import models
import models.dialog
import datasets
from datasets import Dialog
from loopers import LossInferencer
from loopers import EvaluatingInferencer
from loopers import LogInferencer
from loopers import Generator
from loopers import LogGenerator
from loopers import BeamSearchGenerator
from loopers import EvaluatingGenerator


@dataclass
class FinegrainedValidator(LogInferencer, EvaluatingInferencer, LossInferencer):
    pass


@dataclass
class GenerativeValidator(
    LogGenerator,
    EvaluatingGenerator,
    BeamSearchGenerator,
    Generator
):
    pass


def create_parser():
    parser = yaap.Yaap()
    # data options
    parser.add_pth("data-dir", is_dir=True, must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("tests/data/json")),
                   help="Path to the data dir. Must contain 'train.json' and "
                        "'dev.json'.")
    parser.add_str("eval-splits", default=("train", "dev", "test"),
                   choices=("train", "dev", "test"),
                   help="List of splits to evaluate on.")
    # model options
    parser.add_pth("model-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/vhda-mini.yml")),
                   help="Path to the model configuration file.")
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
    parser.add_int("report-every",
                   help="Report training statistics every N steps.")
    # training options
    parser.add_int("batch-size", default=32,
                   help="Mini-batch size.")
    parser.add_str("optimizer", default="adam", choices=("adam",),
                   help="Optimizer to use.")
    parser.add_flt("gradient-clip",
                   help="Clip gradients by norm size.")
    parser.add_flt("l2norm-weight",
                   help="Weight of l2-norm regularization.")
    parser.add_flt("learning-rate", default=0.001, min_bound=0,
                   help="Optimizer learning rate.")
    parser.add_int("epochs", default=10, min_bound=1,
                   help="Number of epochs to train for.")
    parser.add_str("kld-schedule",
                   help="KLD w schedule given as a list of data points. Each "
                        "data point is a pair of training step and target "
                        "dropout scale. Steps in-between data points will be "
                        "interpolated. e.g. '[(0, 1.0), (10000, 0.1)]'")
    parser.add_str("dropout-schedule",
                   help="Dropout schedule given as a list of data points. Each "
                        "data point is a pair of training step and target "
                        "dropout scale. Steps in-between data points will be "
                        "interpolated. e.g. '[(0, 1.0), (10000, 0.1)]'")
    parser.add_bol("disable-kl",
                   help="Whether to disable kl-divergence term.")
    parser.add_str("kl-mode", default="kl-mi",
                   help="KL mode: one of kl, kl-mi, kl-mi+.")
    # validation options
    parser.add_int("valid-batch-size", default=32,
                   help="Mini-batch sizes for validation inference.")
    parser.add_flt("validate-every", default=1,
                   help="Number of epochs in-between validations.")
    parser.add_bol("early-stop",
                   help="Whether to enable early-stopping.")
    parser.add_str("early-stop-criterion", default="~loss",
                   help="The training statistics key to use as criterion "
                        "for early-stopping. Prefix with '~' to denote "
                        "negation during comparison.")
    parser.add_int("early-stop-patience",
                   help="Number of epochs to wait without breaking "
                        "records until executing early-stopping.")
    parser.add_int("beam-size", default=4)
    parser.add_int("max-conv-len", default=20)
    parser.add_int("max-sent-len", default=30)
    # testing options
    parser.add_str("embed-type", default="glove",
                   choices=("glove", "bin", "hdf5"),
                   help="Type of embedding to load for emb. evaluation.")
    parser.add_pth("embed-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("tests/data/glove/"
                                      "glove.840B.300d.woz.txt")),
                   help="Path to embedding file for emb. evaluation.")
    # misc
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
    logger = logging.getLogger("train")
    utils.seed(args.seed)
    logger.info("loading data...")
    load_fn = utils.chain_func(lambda x: list(map(Dialog.from_json, x)),
                               utils.load_json)
    data_dir = pathlib.Path(args.data_dir)
    train_data = load_fn(str(data_dir.joinpath("train.json")))
    valid_data = load_fn(str(data_dir.joinpath("dev.json")))
    test_data = load_fn(str(data_dir.joinpath("test.json")))
    processor = datasets.DialogProcessor(
        sent_processor=datasets.SentProcessor(
            bos=True,
            eos=True,
            lowercase=True,
            tokenizer="space",
            max_len=30
        ),
        boc=True,
        eoc=True,
        state_order="randomized",
        max_len=30
    )
    processor.prepare_vocabs(
        list(itertools.chain(train_data, valid_data, test_data)))
    utils.save_pickle(processor, save_dir.joinpath("processor.pkl"))
    logger.info("preparing model...")
    utils.save_json(utils.load_yaml(args.model_path),
                    save_dir.joinpath("model.json"))
    torchmodels.register_packages(models)
    model_cls = torchmodels.create_model_cls(models, args.model_path)
    model: models.AbstractTDA = model_cls(processor.vocabs)
    model.reset_parameters()
    utils.report_model(logger, model)
    device = torch.device("cpu")
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    model = model.to(device)

    def create_scheduler(s):
        return utils.PiecewiseScheduler([utils.Coordinate(*t) for t in eval(s)])

    save_dir = pathlib.Path(args.save_dir)
    train_args = train.TrainArguments(
        model=model,
        train_data=tuple(train_data),
        valid_data=tuple(valid_data),
        processor=processor,
        device=device,
        save_dir=save_dir,
        report_every=args.report_every,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        optimizer=args.optimizer,
        gradient_clip=args.gradient_clip,
        l2norm_weight=args.l2norm_weight,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        kld_schedule=(utils.ConstantScheduler(1.0)
                      if args.kld_schedule is None else
                      create_scheduler(args.kld_schedule)),
        dropout_schedule=(utils.ConstantScheduler(1.0)
                          if args.dropout_schedule is None else
                          create_scheduler(args.dropout_schedule)),
        validate_every=args.validate_every,
        early_stop=args.early_stop,
        early_stop_criterion=args.early_stop_criterion,
        early_stop_patience=args.early_stop_patience,
        disable_kl=args.disable_kl,
        kl_mode=args.kl_mode
    )
    utils.save_json(train_args.to_json(), save_dir.joinpath("train-args.json"))
    record = train.train(train_args)
    utils.save_json(record.to_json(), save_dir.joinpath("final-summary.json"))
    eval_dir = save_dir.joinpath("eval")
    shell.mkdir(eval_dir, silent=True)
    eval_data = dict(list(filter(None, [
        ("train", train_data) if "train" in args.eval_splits else None,
        ("dev", valid_data) if "dev" in args.eval_splits else None,
        ("test", test_data) if "test" in args.eval_splits else None
    ])))
    for split, data in eval_data.items():
        eval_args = evaluate.EvaluateArugments(
            model=model,
            train_data=tuple(train_data),
            test_data=tuple(data),
            processor=processor,
            embed_type=args.embed_type,
            embed_path=args.embed_path,
            device=device,
            batch_size=args.valid_batch_size,
            beam_size=args.beam_size,
            max_conv_len=args.max_conv_len,
            max_sent_len=args.max_sent_len
        )
        utils.save_json(eval_args.to_json(),
                        eval_dir.joinpath(f"eval-{split}-args.json"))
        with torch.no_grad():
            eval_results = evaluate.evaluate(eval_args)
        save_path = eval_dir.joinpath(f"eval-{split}.json")
        utils.save_json(eval_results, save_path)
        logger.info(f"'{split}' results saved to {save_path}")
    logger.info("done!")


if __name__ == "__main__":
    main()
