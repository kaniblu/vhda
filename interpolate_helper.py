__all__ = ["main"]

import pathlib

import yaap

import utils
import interpolate


def create_parser():
    parser = yaap.Yaap()
    # data options
    parser.add_str("dataset", default="woz")
    parser.add_pth("exp-dir", is_dir=True, must_exist=True, required=True,
                   help="Path to an experiment folder.")
    parser.add_str("dirname", default="z-interp",
                   help="Name of the resulting directory.")
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
    parser.add_int("gpu", min_bound=0,
                   help="GPU device to use. (e.g. 0, 1, etc.)")
    # display options
    parser.add_pth("logging-config", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/logging.yml")),
                   help="Path to a logging config file (yaml/json).")
    parser.add_bol("overwrite", help="Whether to overwrite save dir.")
    return parser


def main(args=None):
    args = utils.parse_args(create_parser(), args)
    exp_dir = pathlib.Path(args.exp_dir)
    interpolate.main(list(filter(None, [
        "--data-dir", str(pathlib.Path(__file__).parent.absolute()
                          .joinpath(f"data/{args.dataset}/json/preprocessed")),
        "--anchor1" if args.anchor1 is not None else None, args.anchor1,
        "--anchor2" if args.anchor2 is not None else None, args.anchor2,
        "--splits", "train",
        "--processor-path", str(exp_dir.joinpath("processor.pkl")),
        "--model-path", str(exp_dir.joinpath("model.json")),
        "--ckpt-path", str(exp_dir.joinpath("checkpoint-best.pth")),
        "--logging-config", args.logging_config,
        "--save-dir", str(exp_dir.joinpath(args.dirname)),
        "--steps", str(args.steps),
        "--overwrite" if args.overwrite else None,
    ])))


if __name__ == "__main__":
    main()
