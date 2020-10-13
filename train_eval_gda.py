import pprint
import logging
import pathlib
import itertools
import logging.config
from dataclasses import dataclass

import yaap
import inflect
import torch
import torch.utils.tensorboard
import torchmodels

import utils
import train
import evaluate
import models
import models.dialog
import datasets
import generate
from datasets import Dialog
from loopers import LossInferencer
from loopers import EvaluatingInferencer
from loopers import LogInferencer
from loopers import Generator
from loopers import LogGenerator
from loopers import BeamSearchGenerator
from loopers import EvaluatingGenerator
from tools import reduce_json
import dst.internal.run as dst_run
import dst.internal.models as dst_models
import dst.internal.models.dst as dst_pkg
import dst.internal.datasets as dst_datasets


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
    parser.add_pth("gen-model-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/vhda-mini.yml")),
                   help="Path to the generative model configuration file.")
    parser.add_pth("dst-model-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("dst/internal/configs/gce.yml")),
                   help="Path to the dst model configuration file.")
    parser.add_int("gpu", min_bound=0,
                   help="GPU device to use. (e.g. 0, 1, etc.)")
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
    # generation options
    parser.add_int("gen-runs", default=3,
                   help="Number of generation runs.")
    parser.add_bol("validate-dst",
                   help="Whether to validate by checking uniqueness.")
    parser.add_bol("validate-unique",
                   help="Whether to validate by checking uniqueness.")
    parser.add_flt("multiplier", default=1.0,
                   help="Ratio of dialog instances to generate. ")
    # DST options
    parser.add_int("dst-batch-size", default=32,
                   help="Mini-batch size.")
    parser.add_int("dst-runs", default=5,
                   help="Number of DST models to train and evaluate using "
                        "different seeds.")
    parser.add_int("dst-epochs", default=200,
                   help="Number of epochs to train DST. "
                        "The actual number of epochs will be scaled by "
                        "the multiplier.")
    parser.add_flt("dst-l2norm",
                   help="DST Weight of l2norm regularization.")
    parser.add_flt("dst-gradient-clip",
                   help="DST Clipping bounds for gradients.")
    parser.add_bol("dst-test-asr",
                   help="Whether to use asr information during testing.")
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
    engine = inflect.engine()
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
    utils.save_json(utils.load_yaml(args.gen_model_path),
                    save_dir.joinpath("model.json"))
    torchmodels.register_packages(models)
    model_cls = torchmodels.create_model_cls(models, args.gen_model_path)
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
        eval_results = evaluate.evaluate(eval_args)
        save_path = eval_dir.joinpath(f"eval-{split}.json")
        utils.save_json(eval_results, save_path)
        logger.info(f"'{split}' results saved to {save_path}")
    logger.info(f"will run {args.gen_runs} generation trials...")
    gen_summary = []
    dst_summary = []
    for gen_idx in range(1, args.gen_runs + 1):
        logger.info(f"running {engine.ordinal(gen_idx)} generation trial...")
        gen_dir = save_dir.joinpath(f"gen-{gen_idx:03d}")
        shell.mkdir(gen_dir, silent=True)
        gen_args = generate.GenerateArguments(
            model=model,
            processor=processor,
            data=train_data,
            instances=int(round(len(train_data) * args.multiplier)),
            batch_size=args.valid_batch_size,
            conv_scale=args.conv_scale,
            spkr_scale=args.spkr_scale,
            goal_scale=args.goal_scale,
            state_scale=args.state_scale,
            sent_scale=args.sent_scale,
            validate_dst=True,
            validate_unique=args.validate_unique,
            device=device
        )
        utils.save_json(gen_args.to_json(), gen_dir.joinpath("gen-args.json"))
        with torch.no_grad():
            samples = generate.generate(gen_args)
        utils.save_json([sample.output.to_json() for sample in samples],
                        gen_dir.joinpath("gen-out.json"))
        utils.save_json([sample.input.to_json() for sample in samples],
                        gen_dir.joinpath("gen-in.json"))
        utils.save_lines([str(sample.log_prob) for sample in samples],
                         gen_dir.joinpath("logprob.txt"))
        da_data = [sample.output for sample in samples]
        data = {"train": train_data, "dev": valid_data, "test": test_data}
        data["train"] += da_data
        # convert dialogs to dst dialogs
        data = {split: list(map(datasets.DSTDialog.from_dialog, dialogs))
                for split, dialogs in data.items()}
        for split, dialogs in data.items():
            logger.info(f"verifying '{split}' dataset...")
            for dialog in dialogs:
                dialog.compute_user_goals()
                dialog.validate()
        logger.info("preparing dst environment...")
        dst_processor = dst_datasets.DSTDialogProcessor(
            sent_processor=datasets.SentProcessor(
                bos=True,
                eos=True,
                lowercase=True,
                max_len=30
            )
        )
        dst_processor.prepare_vocabs(list(itertools.chain(*data.values())))
        train_dataset = dst_datasets.DSTDialogDataset(
            dialogs=data["train"],
            processor=dst_processor
        )
        train_dataloader = dst_datasets.create_dataloader(
            train_dataset,
            batch_size=args.dst_batch_size,
            shuffle=True,
            pin_memory=True
        )
        dev_dataloader = dst_run.TestDataloader(
            dialogs=data["dev"],
            processor=dst_processor,
            max_batch_size=args.dst_batch_size
        )
        test_dataloader = dst_run.TestDataloader(
            dialogs=data["test"],
            processor=dst_processor,
            max_batch_size=args.dst_batch_size
        )
        logger.info("saving dst processor object...")
        utils.save_pickle(dst_processor, gen_dir.joinpath("processor.pkl"))
        torchmodels.register_packages(dst_models)
        dst_model_cls = torchmodels.create_model_cls(dst_pkg,
                                                     args.dst_model_path)
        dst_model = dst_model_cls(dst_processor.vocabs)
        dst_model = dst_model.to(device)
        logger.info(str(model))
        logger.info(f"number of parameters DST: "
                    f"{utils.count_parameters(dst_model):,d}")
        logger.info(f"running {args.dst_runs} trials...")
        all_results = []
        for idx in range(1, args.dst_runs + 1):
            logger.info(f"running {engine.ordinal(idx)} dst trial...")
            trial_dir = gen_dir.joinpath(f"dst-{idx:03d}")
            logger.info("resetting parameters...")
            dst_model.reset_parameters()
            logger.info("preparing trainer...")
            runner = dst_run.Runner(
                model=dst_model,
                processor=dst_processor,
                device=device,
                save_dir=trial_dir,
                epochs=int(round(args.dst_epochs / (1 + args.multiplier))),
                loss="sum",
                l2norm=args.dst_l2norm,
                gradient_clip=args.dst_gradient_clip,
                train_validate=False,
                early_stop=True,
                early_stop_criterion="joint-goal",
                early_stop_patience=None,
                asr_method="scaled",
                asr_sigmoid_sum_order="sigmoid-sum",
                asr_topk=5
            )

            logger.info("commencing training...")
            record = runner.train(
                train_dataloader=train_dataloader,
                dev_dataloader=dev_dataloader,
                test_fn=None
            )
            logger.info("final summary: ")
            logger.info(pprint.pformat(record.to_json()))
            utils.save_json(record.to_json(),
                            trial_dir.joinpath("summary.json"))
            if not args.dst_test_asr:
                logger.info("commencing testing...")
                with torch.no_grad():
                    eval_results = runner.test(test_dataloader)
                logger.info("test results: ")
                logger.info(pprint.pformat(eval_results))
            else:
                logger.info("commencing testing (asr)...")
                with torch.no_grad():
                    eval_results = runner.test_asr(test_dataloader)
                logger.info("test(asr) results: ")
                logger.info(pprint.pformat(eval_results))
            eval_results["epoch"] = int(record.epoch)
            logger.info("test evaluation: ")
            logger.info(pprint.pformat(eval_results))
            utils.save_json(eval_results, trial_dir.joinpath("eval.json"))
            all_results.append(eval_results)
            dst_summary.append(eval_results)
        logger.info("aggregating results...")
        summary = reduce_json(all_results)
        logger.info("aggregated results: ")
        agg_summary = pprint.pformat({k: v["stats"]["mean"]
                                      for k, v in summary.items()})
        logger.info(pprint.pformat(agg_summary))
        gen_summary.append(agg_summary)
        utils.save_json(summary, gen_dir.joinpath("summary.json"))
    gen_summary = reduce_json(gen_summary)
    dst_summary = reduce_json(dst_summary)
    logger.info(f"aggregating generation trials ({args.gen_runs})...")
    logger.info(pprint.pformat({k: v["stats"]["mean"]
                                for k, v in gen_summary.items()}))
    logger.info(f"aggregating dst trials ({args.gen_runs * args.dst_runs})...")
    logger.info(pprint.pformat({k: v["stats"]["mean"]
                                for k, v in dst_summary.items()}))
    utils.save_json(gen_summary, save_dir.joinpath("gen-summary.json"))
    utils.save_json(dst_summary, save_dir.joinpath("dst-summary.json"))
    logger.info("done!")


if __name__ == "__main__":
    main()
