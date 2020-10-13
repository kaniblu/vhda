__all__ = ["main"]

import pprint
import logging
import pathlib
import itertools

import yaap
import inflect
import torch
import torchmodels

import utils
import models
import generate
import datasets
from tools import reduce_json
import dst.internal.run as dst_run
import dst.internal.models as dst_models
import dst.internal.models.dst as dst_pkg
import dst.internal.datasets as dst_datasets
from datasets import Dialog


def create_parser():
    parser = yaap.Yaap("Conduct generative data augmentation experiments.")
    # data options
    parser.add_pth("data-dir", is_dir=True, must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("tests/data/json")),
                   help="Path to a json-format dialogue dataset.")
    parser.add_pth("processor-path", must_exist=True, required=True,
                   help="Path to the processor pickle file.")
    # model options
    parser.add_pth("gen-model-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/vhda-mini.yml")),
                   help="Path to the generative model configuration file.")
    parser.add_pth("dst-model-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("dst/internal/configs/gce.yml")),
                   help="Path to the dst model configuration file.")
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
    parser.add_int("gen-runs", default=3,
                   help="Number of generations to run.")
    parser.add_int("gen-batch-size", default=32,
                   help="Mini-batch size.")
    parser.add_flt("multiplier", default=1.0,
                   help="Ratio of dialog instances to generate. ")
    parser.add_bol("validate-unique",
                   help="Whether to validate by checking uniqueness.")
    # DST options
    parser.add_int("dst-batch-size", default=32,
                   help="Mini-batch size.")
    parser.add_int("dst-runs", default=5,
                   help="Number of DST models to train and evaluate using "
                        "different seeds.")
    parser.add_int("epochs", default=200,
                   help="Number of epochs to train DST. "
                        "The actual number of epochs will be scaled by "
                        "the multiplier.")
    parser.add_flt("l2norm",
                   help="DST Weight of l2norm regularization.")
    parser.add_flt("gradient-clip",
                   help="DST Clipping bounds for gradients.")
    parser.add_bol("test-asr",
                   help="Whether to use asr information during testing.")
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


def main(args=None):
    args = utils.parse_args(create_parser(), args)
    if args.logging_config is not None:
        logging.config.dictConfig(utils.load_yaml(args.logging_config))
    save_dir = pathlib.Path(args.save_dir)
    if (not args.overwrite and
            save_dir.exists() and utils.has_element(save_dir.glob("*"))):
        raise FileExistsError(f"save directory ({save_dir}) is not empty")
    shell = utils.ShellUtils()
    engine = inflect.engine()
    shell.mkdir(save_dir, silent=True)
    logger = logging.getLogger("gda")
    utils.seed(args.seed)
    logger.info("loading data...")
    load_fn = utils.chain_func(
        lambda data: list(map(Dialog.from_json, data)),
        utils.load_json
    )
    processor = utils.load_pickle(args.processor_path)
    data_dir = pathlib.Path(args.data_dir)
    train_data = load_fn(str(data_dir.joinpath("train.json")))
    valid_data = load_fn(str(data_dir.joinpath("dev.json")))
    test_data = load_fn(str(data_dir.joinpath("test.json")))
    data = {"train": train_data, "dev": valid_data, "test": test_data}
    logger.info("preparing model...")
    torchmodels.register_packages(models)
    model_cls = torchmodels.create_model_cls(models, args.gen_model_path)
    model: models.AbstractTDA = model_cls(processor.vocabs)
    model.reset_parameters()
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    device = torch.device("cpu")
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    model = model.to(device)
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
            data=tuple(train_data),
            instances=int(round(len(train_data) * args.multiplier)),
            batch_size=args.gen_batch_size,
            conv_scale=args.conv_scale,
            spkr_scale=args.spkr_scale,
            goal_scale=args.goal_scale,
            state_scale=args.state_scale,
            sent_scale=args.sent_scale,
            validate_dst=True,
            validate_unique=args.validate_unique,
            device=device
        )
        utils.save_json(gen_args.to_json(), gen_dir.joinpath("args.json"))
        with torch.no_grad():
            samples = generate.generate(gen_args)
        utils.save_json([sample.output.to_json() for sample in samples],
                        gen_dir.joinpath("out.json"))
        utils.save_json([sample.input.to_json() for sample in samples],
                        gen_dir.joinpath("in.json"))
        utils.save_lines([str(sample.log_prob) for sample in samples],
                         gen_dir.joinpath("logprob.txt"))
        da_data = [sample.output for sample in samples]
        gen_data = {
            "train": data["train"] + da_data,
            "dev": data["dev"],
            "test": data["test"]
        }
        # convert dialogs to dst dialogs
        gen_data = {split: list(map(datasets.DSTDialog.from_dialog, dialogs))
                    for split, dialogs in gen_data.items()}
        for split, dialogs in gen_data.items():
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
        dst_processor.prepare_vocabs(list(itertools.chain(*gen_data.values())))
        train_dataset = dst_datasets.DSTDialogDataset(
            dialogs=gen_data["train"],
            processor=dst_processor
        )
        train_dataloader = dst_datasets.create_dataloader(
            train_dataset,
            batch_size=args.dst_batch_size,
            shuffle=True,
            pin_memory=True
        )
        dev_dataloader = dst_run.TestDataloader(
            dialogs=gen_data["dev"],
            processor=dst_processor,
            max_batch_size=args.dst_batch_size
        )
        test_dataloader = dst_run.TestDataloader(
            dialogs=gen_data["test"],
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
        logger.info(f"will run {args.dst_runs} trials...")
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
                epochs=int(round(args.epochs / (1 + args.multiplier))),
                loss="sum",
                l2norm=args.l2norm,
                gradient_clip=args.gradient_clip,
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
            if not args.test_asr:
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
            eval_results["criterion"] = record.value
            logger.info("test evaluation: ")
            logger.info(pprint.pformat(eval_results))
            utils.save_json(eval_results, trial_dir.joinpath("eval.json"))
            all_results.append(eval_results)
            dst_summary.append(eval_results)
        logger.info("aggregating results...")
        summary = reduce_json(all_results)
        logger.info("aggregated results: ")
        agg_results = {k: v["stats"]["mean"] for k, v in summary.items()}
        gen_summary.append(agg_results)
        logger.info(pprint.pformat(agg_results))
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
