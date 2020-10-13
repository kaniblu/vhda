import json
import pprint
import pathlib
import logging
import logging.config
import itertools
import functools

import torch
import torch.optim as op
import torchmodels

import utils
import models
import datasets
from tools import reduce_json
from .models import dst
from . import run as dst_run
from . import models as dst_models
from . import datasets as dst_datasets


def create_parser():
    parser = dst_run.create_parser()
    parser.add_int("runs", default=8, help="Number of runs.")
    return parser


def main():
    parser = create_parser()
    args = utils.parse_args(parser)
    if args.logging_config is not None:
        logging.config.dictConfig(utils.load_yaml(args.logging_config))
    logger = logging.getLogger("multirun")
    save_dir = pathlib.Path(args.save_dir)
    if (not args.overwrite and save_dir.exists() and
            utils.has_element(save_dir.glob("*"))):
        raise FileExistsError(f"save directory ({save_dir}) is not empty")
    save_dir.mkdir(exist_ok=True, parents=True)
    utils.save_yaml(vars(args), save_dir.joinpath("args.yml"))
    logger.info("preparing dataset...")
    data_dir = pathlib.Path(args.data_dir)
    data = {split: utils.load_json(data_dir.joinpath(f"{split}.json"))
            for split in ("train", "dev", "test")}
    data = {split: [datasets.DSTDialog.from_dialog(datasets.Dialog.from_json(d))
                    for d in dialogs]
            for split, dialogs in data.items()}
    logger.info("verifying dataset...")
    for split, dialogs in data.items():
        for dialog in dialogs:
            dialog.validate()
    processor = dst_datasets.DSTDialogProcessor(
        sent_processor=datasets.SentProcessor(
            bos=True,
            eos=True,
            lowercase=True,
            max_len=30
        )
    )
    processor.prepare_vocabs(
        list(itertools.chain(*(data["train"], data["dev"], data["test"]))))
    logger.info("saving processor object...")
    utils.save_pickle(processor, save_dir.joinpath("processor.pkl"))
    train_dataset = dst_datasets.DSTDialogDataset(
        dialogs=data["train"],
        processor=processor
    )
    train_dataloader = dst_datasets.create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    dev_dataloader = dst_run.TestDataloader(
        dialogs=data["dev"],
        processor=processor,
        max_batch_size=args.batch_size
    )
    test_dataloader = dst_run.TestDataloader(
        dialogs=data["test"],
        processor=processor,
        max_batch_size=args.batch_size
    )
    logger.info("preparing model...")
    torchmodels.register_packages(models)
    torchmodels.register_packages(dst_models)
    model_cls = torchmodels.create_model_cls(dst, args.model_path)
    model: dst.AbstractDialogStateTracker = model_cls(processor.vocabs)
    if args.gpu is None:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")
    model = model.to(device)
    logger.info(str(model))
    logger.info(f"number of parameters: {utils.count_parameters(model):,d}")
    logger.info(f"running {args.runs} trials...")
    all_results = []
    for idx in range(args.runs):
        logger.info(f"running trial-{idx + 1}...")
        run_save_dir = save_dir.joinpath(f"run-{idx + 1:03d}")
        logger.info("resetting parameters...")
        model.reset_parameters()
        logger.info("preparing trainer...")
        runner = dst_run.Runner(
            model=model,
            processor=processor,
            device=device,
            save_dir=run_save_dir,
            epochs=args.epochs,
            scheduler=(None if not args.scheduled_lr else
                       functools.partial(
                           getattr(op.lr_scheduler, args.scheduler_cls),
                           **json.loads(args.scheduler_kwargs)
                       )),
            loss=args.loss,
            l2norm=args.l2norm,
            gradient_clip=args.gradient_clip,
            train_validate=args.train_validate,
            early_stop=args.early_stop,
            early_stop_criterion=args.early_stop_criterion,
            early_stop_patience=args.early_stop_patience,
            asr_method=args.asr_method,
            asr_sigmoid_sum_order=args.asr_sigmoid_sum_order,
            asr_topk=args.asr_topk
        )
        logger.info("commencing training...")
        record = runner.train(
            train_dataloader=train_dataloader,
            dev_dataloader=dev_dataloader,
            test_fn=runner.test_asr if args.validate_asr else None
        )
        logger.info("final summary: ")
        logger.info(pprint.pformat(record.to_json()))
        utils.save_json(record.to_json(),
                        run_save_dir.joinpath("summary-final.json"))
        logger.info("commencing testing...")
        with torch.no_grad():
            eval_results = runner.test(test_dataloader)
        logger.info("test results: ")
        logger.info(pprint.pformat(eval_results))
        if args.test_asr:
            logger.info("commencing testing (asr)...")
            with torch.no_grad():
                eval_results = runner.test_asr(test_dataloader)
            logger.info("test(asr) results: ")
            logger.info(pprint.pformat(eval_results))
        eval_results["epoch"] = int(record.epoch)
        eval_results["criterion"] = record.value
        logger.info("test evaluation: ")
        logger.info(pprint.pformat(eval_results))
        if args.save_ckpt:
            logger.info("saving checkpoint...")
            torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                       run_save_dir.joinpath("ckpt.pth"))
        logger.info("done!")
        utils.save_json(eval_results, run_save_dir.joinpath("eval.json"))
        all_results.append(eval_results)
    logger.info("aggregating results...")
    summary = reduce_json(all_results)
    pprint.pprint({k: v["stats"]["mean"] for k, v in summary.items()})
    utils.save_json(summary, save_dir.joinpath("summary.json"))
    logger.info("done!")


if __name__ == "__main__":
    main()
