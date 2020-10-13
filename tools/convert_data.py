__all__ = ["convert_data"]

import sys

import yaap

import utils
import datasets


def convert_data(args):
    if args.data_format == "woz":
        load_fn = datasets.load_woz
    elif args.data_format == "json":
        load_fn = utils.chain_func(
            lambda x: list(map(datasets.Dialog.from_json, x)),
            utils.load_json
        )
    elif args.data_format == "dstc":
        load_fn = datasets.load_dstc2
    else:
        raise ValueError(f"unsupported data type: {args.data_type}")
    data = load_fn(args.data_path)
    formatter = utils.DialogTableFormatter(max_col_len=args.max_column_length)
    if args.save_path is not None:
        out = open(args.save_path, "w")
    else:
        out = sys.stdout
    if args.save_format == "human":
        for i, dialog in enumerate(data):
            out.write(f"Dialog #{i + 1}\n")
            out.write("\n")
            out.write(formatter.format(dialog))
            out.write("\n\n")
    elif args.save_format == "json":
        data = [dialog.to_dict() for dialog in data]
        utils.save_json(data, out)
    elif args.save_format == "woz":
        data = [dialog.to_woz(dialog.meta.get("dialogue_idx", i), True)
                for i, dialog in enumerate(data)]
        utils.save_json(data, out)
    if args.save_path is not None:
        out.close()


def create_parser():
    parser = yaap.Yaap()
    parser.add_pth("data-path", must_exist=True, required=True,
                   help="Path to the dialog data.")
    parser.add_str("data-format", default="json",
                   choices=("woz", "json", "dstc"),
                   help="Data format of the data to be loaded.")
    parser.add_int("max-column-length", default=50,
                   help="Maximum length of each column of formatted table.")
    parser.add_str("save-format", default="human",
                   choices=("human", "woz", "json"),
                   help="Output data format.")
    parser.add_pth("save-path",
                   help="Path to save the resulting text file.")
    return parser


if __name__ == "__main__":
    convert_data(utils.parse_args(create_parser()))
