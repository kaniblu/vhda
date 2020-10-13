__all__ = ["compute_ttest"]

import re
import sys

import yaap
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind

import utils


def create_parser():
    parser = yaap.Yaap(
        desc="Conducts t-test on a summary or a pair of summary files."
    )
    parser.add_pth("summary-path", required=True, must_exist=True)
    parser.add_pth("other-path", must_exist=True,
                   help="If supplied, the script will perform unpaired "
                        "t-test of two individual samples on common keys.")
    parser.add_str("expected-mean", is_list=True, regex=r"(.+)=(.+)",
                   help="Expected mean for summary samples. Must be "
                        "given as a pair of key and expected population "
                        "mean (e.g. 'loss=0.3'). Could supply multiple "
                        "pairs. Applicable only when no other summary "
                        "path is supplied.")
    parser.add_pth("save-path")
    return parser


def compute_ttest(sample, other_sample=None, pop_mean=None):
    if other_sample is not None:
        res = ttest_ind(sample, other_sample)
        return dict(t=res.statistic, p=res.pvalue)
    if pop_mean is not None:
        res = ttest_1samp(sample, pop_mean)
        return dict(t=res.statistic, p=res.pvalue)
    raise ValueError(f"must provide other sample or population mean")


def parse_kvp(kvps) -> dict:
    regex = re.compile(r"(.+)=(.+)")
    res = dict()
    for kvp in kvps:
        m = regex.match(kvp)
        if not m:
            raise ValueError(f"not a valid key-value pair: {kvp}")
        key = m.group(1)
        value = float(m.group(2))
        res[key] = value
    return res


def ttest(args):
    summary = utils.load_yaml(args.summary_path)
    result = None
    if args.other_path is not None:
        other_summary = utils.load_yaml(args.other_path)
        result = {k: compute_ttest(summary[k]["raw"], other_summary[k]["raw"])
                  for k in set(summary) & set(other_summary)}
    if args.expected_mean:
        expected_mean = parse_kvp(args.expected_mean)
        for k in expected_mean:
            if k not in summary:
                raise KeyError(f"key not found in summary: {k}")
        result = {k: compute_ttest(summary[k]["raw"], pop_mean=mean)
                  for k, mean in expected_mean.items()}
    if result is None:
        raise ValueError(f"must provide other summary file or a pop. mean")
    utils.save_json(result, args.save_path or sys.stdout)


if __name__ == "__main__":
    ttest(utils.parse_args(create_parser()))
