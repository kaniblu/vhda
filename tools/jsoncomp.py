import sys
import argparse
from dataclasses import dataclass

import utils


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_a")
    parser.add_argument("path_b")
    parser.add_argument("--set-list", action="store_true", default=False)
    return parser


class MismatchException(Exception):
    pass


@dataclass
class JsonComparer:
    set_list: bool = False

    def try_compare(self, a, b):
        try:
            self.compare(a, b)
            return True, None
        except MismatchException as e:
            return False, str(e)

    def compare(self, a, b):
        if type(a) != type(b):
            raise MismatchException(f"type different: {type(a)} != {type(b)}")
        if isinstance(a, dict):
            a_keys, b_keys = set(a.keys()), set(b.keys())
            if a_keys != b_keys:
                raise MismatchException(f"dictionaries have different key "
                                        f"sets: {a_keys} != {b_keys}")
            for key in a_keys:
                a_val, b_val = a[key], b[key]
                try:
                    self.compare(a_val, b_val)
                except MismatchException as e:
                    e_msg = str(e)
                    raise MismatchException(f"['{key}'] {e_msg}")
        elif isinstance(a, list):
            if len(a) != len(b):
                raise MismatchException(f"lists have different lengths: "
                                        f"{len(a)} != {len(b)}")
            if len(a) == 0:
                return
            if not self.set_list:
                for i, (a_val, b_val) in enumerate(zip(a, b)):
                    try:
                        self.compare(a_val, b_val)
                    except MismatchException as e:
                        e_msg = str(e)
                        raise MismatchException(f"[{i}] {e_msg}")
            else:
                match = utils.MaximumBipartiteMatching(
                    num_left=len(a),
                    num_right=len(a),
                    query_fn=lambda i, j: self.try_compare(a[i], b[j])[0]
                ).compute()
                if not match.is_all_matched:
                    raise MismatchException(
                        f"some of the elements did not match: "
                        f"left_elements={match.unmatched_left}, "
                        f"right_elements={match.unmatched_right}"
                    )
        elif isinstance(a, (str, int, float, bool)):
            if a != b:
                raise MismatchException(f"items do not match: {a} != {b}")
        else:
            raise TypeError(f"unknown json object type: {type(a)}")


def main(args):
    comparer = JsonComparer(
        set_list=args.set_list
    )
    try:
        comparer.compare(
            a=utils.load_yaml(args.path_a),
            b=utils.load_yaml(args.path_b)
        )
    except MismatchException as e:
        sys.stderr.write(f"JSON mismatch:\n")
        sys.stderr.write(str(e) + "\n")
        exit(1)


if __name__ == "__main__":
    main(create_parser().parse_args())
