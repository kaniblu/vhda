__all__ = ["load_json", "save_json", "load_yaml", "save_yaml", "save_serial",
           "prod", "merge_dict", "randstr", "tokenize", "pad_iter",
           "TensorMap", "parse_args", "lstrip", "rstrip", "private_field",
           "chain_func", "has_element", "union", "intersect", "concat",
           "save_pickle", "load_pickle", "bucket", "merge_dicts",
           "load_lines", "save_lines", "seed", "EPS", "report_model"]

import io
import json
import yaml
import pickle
import random
import string
import pathlib
import argparse
import functools
import operator
from dataclasses import field, MISSING
from typing import Mapping, Sequence, Iterable, IO

import numpy as np
import torch
import yaap
import spacy

TensorMap = Mapping[str, torch.Tensor]
EPS = 1e-7


def report_model(logger, model):
    num_params = sum(torch.tensor(p.size()).prod().item()
                     for p in model.parameters() if p.requires_grad)
    logger.info(str(model))
    logger.info(f"number of parameters: {num_params:,d}")


def load_lines(path):
    with open(str(path), "r") as f:
        return [line.rstrip("\r\n") for line in f]


def save_lines(lines, path):
    with open(str(path), "w") as f:
        for line in lines:
            line = str(line)
            f.write(f"{line}\n")


def load_pickle(path):
    with open(str(path), "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(str(path), "wb") as f:
        pickle.dump(obj, f)


def concat(lists: Iterable[Iterable]) -> list:
    ret = []
    for l in lists:
        ret.extend(l)
    return ret


def intersect(sets: Iterable[set]) -> set:
    ret = None
    for s in sets:
        if ret is None:
            ret = s
        ret &= s
    return ret


def union(sets: Iterable[set]) -> set:
    ret = set()
    for s in sets:
        ret |= s
    return ret


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    def _write_json(stream):
        json.dump(
            obj, stream,
            ensure_ascii=False,
            indent=2
        )
        stream.write("\n")

    if isinstance(path, (IO, io.TextIOWrapper)):
        _write_json(path)
    else:
        with open(path, "w") as f:
            _write_json(f)


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    with open(path, "w") as f:
        return yaml.safe_dump(
            obj, f,
            default_flow_style=False,
            allow_unicode=True,
            indent=2
        )


def save_serial(obj, path):
    """Saves object as the serialization format specified by the extension
    name."""
    extmap = {
        ".json": save_json,
        ".yaml": save_yaml,
        ".yml": save_yaml
    }
    extname = pathlib.Path(path).suffix
    if not extname:
        raise ValueError(f"specify extension name to allow format detection.")
    if extname not in extmap:
        raise ValueError(f"unsupported extension: {extname}; "
                         f"use one of {list(extmap.keys())}")
    return extmap[extname](obj, path)


def pad_iter(x: Iterable, size: int, default=None):
    it = iter(x)
    for _ in range(size):
        try:
            yield next(it)
        except StopIteration:
            yield default


def prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)


def parse_args(parser: yaap.Yaap, args=None):
    return argparse.Namespace(**{k.replace("-", "_"): v
                                 for k, v in parser.parse(args).items()})


def merge_dict(a, b):
    ret = {}
    ret.update(a)
    ret.update(b)
    return ret


def merge_dicts(*dicts):
    ret = {}
    for d in dicts:
        ret.update(d)
    return ret


def randstr(l: int) -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(l))


def tokenize(s: str) -> Sequence[str]:
    global nlp
    if "nlp" not in globals():
        nlp = spacy.load("en_core_web_sm")
    return [t.text for t in nlp.tokenizer(s)]


def lstrip(l: Sequence, x=None):
    if not l:
        return l
    if x is None:
        def check(y):
            return bool(y)
    elif not callable(x):
        def check(y):
            return x == y
    else:
        check = x
    i = 0
    for e in l:
        if not check(e):
            break
        i += 1
    return l[i:]


def rstrip(l: Sequence, x=None):
    if not l:
        return l
    if x is None:
        def check(y):
            return bool(y)
    elif not callable(x):
        def check(y):
            return x == y
    else:
        check = x
    i = len(l)
    for e in reversed(l):
        if not check(e):
            break
        i -= 1
    return l[:i]


def private_field(default=MISSING, default_factory=MISSING):
    return field(init=False, repr=False, compare=False,
                 default=default, default_factory=default_factory)


def chain_func(*funcs):
    def _chain(*args, **kwargs):
        cur_args, cur_kwargs = args, kwargs
        ret = None
        for f in reversed(funcs):
            cur_args, cur_kwargs = (f(*cur_args, **cur_kwargs),), {}
            ret = cur_args[0]
        return ret

    return _chain


def has_element(it: Iterable):
    try:
        next(iter(it))
        return True
    except StopIteration:
        return False


def bucket(it: Iterable, compare_fn=None):
    """Organizes items in the iterable into buckets of the same objects.
    Order-sensitive and compare_fn that takes two objects and returns boolean
    can be used to customize comparison behavior.

    Arguments:
        it (Iterable): items.
        compare_fn (Callable): Callable with (Any, Any) -> Bool signature.

    Returns:
        Generator of bucketed items.
    """
    buffer = []
    for e in it:
        if not buffer:
            buffer.append(e)
        elif (compare_fn is None and e == buffer[0] or
              compare_fn is not None and compare_fn(e, buffer[0])):
            buffer.append(e)
        else:
            yield buffer
            buffer = [e]
    if buffer:
        yield buffer


def seed(s: int = None):
    np.random.seed(s)
    # CUDA call fail on RTX2080Ti torch == 1.3.0
    # if s is None:
    #     torch.seed()
    #     torch.cuda.seed()
    # else:
    #     torch.manual_seed(s)
    #     torch.cuda.manual_seed(s)
    random.seed(s)
