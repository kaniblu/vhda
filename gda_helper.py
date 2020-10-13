import pathlib

import yaap

import utils
import gda


def create_parser():
    parser = yaap.Yaap("Conduct generative data augmentation experiments "
                       "(with easier arguments).")
    # data options
    parser.add_str("dataset", default="woz")
    parser.add_pth("exp-dir", is_dir=True, must_exist=True, required=True,
                   help="Path to an experiment folder.")
    # model options
    parser.add_str("dst-model", default="gce-ft-wed",
                   help="Name of the DST model.")
    # model-specific options (TDA)
    parser.add_flt("scale", is_list=True, num_elements=5,
                   default=(1.0, 1.0, 1.0, 1.0, 1.0),
                   help="Scale to introduce into conv, goal and sent vector "
                        "for TDA generation.")
    # generation options
    parser.add_int("gen-runs", default=3,
                   help="Number of generation trials")
    parser.add_flt("multiplier", default=1.0,
                   help="Ratio of dialog instances to generate. ")
    parser.add_bol("validate-unique",
                   help="Whether to validate by checking uniqueness.")
    # DST options
    parser.add_int("batch-size", default=100)
    parser.add_int("dst-runs", default=5,
                   help="Number of DST models to train and evaluate using "
                        "different seeds.")
    parser.add_int("epochs", default=200,
                   help="Number of epochs to train DST. "
                        "The actual number of epochs will be scaled by "
                        "the multiplier.")
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
    exp_dir = pathlib.Path(args.exp_dir)
    gda.main(list(filter(None, [
        "--data-dir", str(pathlib.Path(__file__).parent.absolute()
                          .joinpath(f"data/{args.dataset}/json/preprocessed")),
        "--processor-path", str(exp_dir.joinpath("processor.pkl")),
        "--gen-model-path", str(exp_dir.joinpath("model.json")),
        "--dst-model-path",
        str(pathlib.Path(__file__).parent.absolute()
            .joinpath(f"dst/internal/configs/{args.dst_model}.yml")),
        "--ckpt-path", str(exp_dir.joinpath("checkpoint-best.pth")),
        "--conv-scale", str(args.scale[0]),
        "--spkr-scale", str(args.scale[1]),
        "--goal-scale", str(args.scale[2]),
        "--state-scale", str(args.scale[3]),
        "--sent-scale", str(args.scale[4]),
        "--gen-runs", str(args.gen_runs),
        "--gen-batch-size", "32",
        "--multiplier", str(args.multiplier),
        "--dst-batch-size", str(args.batch_size),
        "--dst-runs", str(args.dst_runs),
        "--epochs", str(args.epochs),
        "--gradient-clip", str(2.0),
        "--test-asr" if args.test_asr else None,
        "--logging-config", str(args.logging_config),
        "--save-dir", str(args.save_dir),
        "--validate-unique" if args.validate_unique else None,
        "--gpu" if args.gpu is not None else None,
        str(args.gpu) if args.gpu is not None else None,
        "--overwrite" if args.overwrite else None,
        "--seed" if args.seed is not None else None,
        str(args.seed) if args.seed is not None else None
    ])))


if __name__ == "__main__":
    main()
