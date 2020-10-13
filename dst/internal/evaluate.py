import logging
import logging.config
import pathlib
import pprint

import yaap
import torch
import torchmodels

import utils
import models
import datasets
from . import models as dst_models
from .models import dst
from .run import TestDataloader
from .run import Runner


def create_parser():
    parser = yaap.Yaap()
    parser.add_pth("logging-config", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("../../tests/config/logging.yml")),
                   help="Path to a logging configuration file.")
    parser.add_pth("save-path", default="out.json",
                   help="Path to save evaluation results (json).")
    parser.add_pth("data-dir", is_dir=True, must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("../../tests/data/json")),
                   help="Path to a json-format dialogue dataset.")
    parser.add_pth("model-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/glad-mini.yml")))
    parser.add_pth("ckpt-path", must_exist=True, required=True,
                   help="Path to a checkpoint file.")
    parser.add_pth("processor-path", must_exist=True, required=True,
                   help="Path to a processor object.")
    parser.add_int("batch-size", default=32, help="Batch size.")
    parser.add_str("loss", default="sum", choices=("sum", "mean"),
                   help="Type of loss aggregation ('sum' or 'mean').")
    parser.add_bol("test-asr",
                   help="Whether to use asr information during testing.")
    parser.add_str("asr-method", default="scaled",
                   choices=("score", "uniform", "ones", "scaled"),
                   help="Type of aggregation method to use when summing output "
                        "scores during asr-enabled evaluation.")
    parser.add_str("asr-sigmoid-sum-order", default="sigmoid-sum",
                   help="The order of sum and sigmoid operations in ASR mode.")
    parser.add_int("asr-topk", min_bound=1,
                   help="Number of top-k candidates.")
    parser.add_int("gpu", help="gpu device id.")
    return parser


def main():
    parser = create_parser()
    args = utils.parse_args(parser)
    if args.logging_config is not None:
        logging.config.dictConfig(utils.load_yaml(args.logging_config))
    logger = logging.getLogger("evaluate")
    logger.info("preparing dataset...")
    data_dir = pathlib.Path(args.data_dir)
    data = [datasets.DSTDialog.from_dialog(datasets.Dialog.from_json(d))
            for d in utils.load_json(data_dir.joinpath("test.json"))]
    logger.info("verifying dataset...")
    for dialog in data:
        dialog.validate()
    processor = utils.load_pickle(args.processor_path)
    test_dataloader = TestDataloader(
        dialogs=data,
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
    model.load_state_dict(torch.load(args.ckpt_path))
    model = model.to(device)
    logger.info(f"number of parameters: {utils.count_parameters(model):,d}")
    logger.info("preparing evaluator...")
    runner = Runner(
        model=model,
        processor=processor,
        device=device,
        asr_method=args.asr_method,
        asr_sigmoid_sum_order=args.asr_sigmoid_sum_order,
        asr_topk=args.asr_topk
    )
    logger.info("commencing evaluation...")
    with torch.no_grad():
        test_fn = runner.test_asr if args.test_asr else runner.test
        eval_results = test_fn(test_dataloader)
    logger.info("done!")
    pprint.pprint(eval_results)
    utils.save_json(eval_results, args.save_path)


if __name__ == "__main__":
    main()
