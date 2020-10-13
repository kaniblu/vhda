import pathlib

import yaap

import utils
import evaluate


def create_parser():
    parser = yaap.Yaap()
    # data options
    parser.add_str("dataset", default="woz")
    parser.add_pth("exp-dir", is_dir=True, must_exist=True, required=True,
                   help="Path to an experiment folder.")
    # model options
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
    parser.add_bol("overwrite", help="Whether to overwrite save dir.")
    # inference options
    parser.add_int("seed", help="Random seed.")
    return parser


def main():
    args = utils.parse_args(create_parser())
    exp_dir = pathlib.Path(args.exp_dir)
    evaluate.main(list(filter(None, [
        "--data-dir", str(pathlib.Path(__file__).parent.absolute()
                          .joinpath(f"data/{args.dataset}/json/preprocessed")),
        "--eval-splits", "train", "dev", "test",
        "--processor-path", str(exp_dir.joinpath("processor.pkl")),
        "--model-path", str(exp_dir.joinpath("model.json")),
        "--ckpt-path", str(exp_dir.joinpath("checkpoint-best.pth")),
        "--embed-type", args.embed_type,
        "--embed-path" if args.embed_path is not None else None,
        args.embed_path,
        "--logging-config", args.logging_config,
        "--save-dir", str(exp_dir.joinpath("eval")),
        "--overwrite" if args.overwrite else None,
        "--seed" if args.seed is not None else None,
        str(args.seed) if args.seed is not None else None
    ])))


if __name__ == "__main__":
    main()
