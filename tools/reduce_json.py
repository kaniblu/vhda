__all__ = ["describe", "reduce_json"]

import sys
import argparse
from typing import Iterable, Sequence, Mapping, Union

import numpy as np
from scipy.stats import norm
from scipy.stats import kurtosis

import utils


def create_parser():
    parser = argparse.ArgumentParser(
        description="Aggregate a list of json dictionaries (of float or ints).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("json_paths", nargs="+")
    parser.add_argument("--key-mode", default="intersection",
                        choices=("intersection", "union"),
                        help="Specify the method to reduce keys: possible "
                             "choices are 'intersection' and 'union'.")
    parser.add_argument("--pad", type=float,
                        help="Padding values when a key does not exist. "
                             "Applicable for 'union' key mode only. If not "
                             "supplied, dictionaries with a key absent will "
                             "be excluded during the aggregation of the key.")
    parser.add_argument("--outlier-threshold", default=0.05, type=float,
                        help="significance level for detecting outliers.")
    parser.add_argument("--save-path")
    return parser


def union_all(sets: Iterable[set]) -> set:
    ret = set()
    for s in sets:
        ret |= s
    return ret


def intersect_all(sets: Iterable[set]) -> set:
    ret = None
    for s in sets:
        if ret is None:
            ret = s
        ret &= s
    if ret is None:
        return set()
    return ret


def describe(values) -> dict:
    if not values:
        return dict(
            min=None,
            max=None,
            mean=None,
            std=None,
            var=None,
            kurt=None,
            count=0
        )
    return dict(
        min=min(values),
        max=max(values),
        mean=sum(values) / len(values),
        std=float(np.std(values)),
        var=float(np.var(values)),
        kurt=float(kurtosis(values)),
        count=len(values)
    )


def separate_outlier(values, threshold=0.05):
    def norm_prob(mu, sigma, v):
        z = (v - mu) / sigma
        if z > 0:
            z = -z
        return norm.cdf(z)

    if not values:
        return [], []
    mean, std = np.mean(values), np.std(values)
    if not std:
        # all values the same
        return values, []
    prob = [norm_prob(mean, std, value) for value in values]
    return ([v for v, p in zip(values, prob) if p > threshold],
            [v for v, p in zip(values, prob) if p <= threshold])


def reduce_values(values, outlier_threshold=0.05):
    if not values:
        return dict(
            raw=[],
            stats=describe([])
        )
    inclusive, outlier = separate_outlier(values, outlier_threshold)
    return dict(
        raw=list(values),
        inclusive=inclusive,
        outlier=outlier,
        stats=describe(inclusive)
    )


def reduce_json(data: Sequence[Mapping[str, Union[int, float]]],
                union: bool = False, pad: float = None,
                outlier_threshold: float = 0.05) -> dict:
    if not union:
        key_fn = intersect_all
    else:
        key_fn = union_all
    return {k: reduce_values(list(filter(lambda x: x is not None,
                                         (d.get(k, pad) for d in data))),
                             outlier_threshold)
            for k in key_fn(set(d.keys()) for d in data)}


def main(args):
    if not args.json_paths:
        raise ValueError(f"must provide at least one json path")
    data = list(map(utils.load_yaml, args.json_paths))
    kwargs = dict()
    if args.key_mode == "intersection":
        kwargs["union"] = False
    elif args.key_mode == "union":
        kwargs["union"] = True
    else:
        raise ValueError(f"unsupported key mode: {args.key_mode}")
    kwargs["outlier_threshold"] = args.outlier_threshold
    kwargs["pad"] = args.pad
    agg = reduce_json(data, **kwargs)
    utils.save_json(agg, args.save_path or sys.stdout)


if __name__ == "__main__":
    main(create_parser().parse_args())
