import pathlib
import random
import pprint
import tempfile

import torch

from utils import *
from datasets import *


def check_size(x, size):
    if not ((isinstance(x,
                        (Stacked1DTensor, DoublyStacked1DTensor, torch.Tensor))
             and isinstance(size, torch.Size)) or
            (isinstance(x, tuple) and isinstance(size, tuple))):
        raise TypeError(f"the output that is being checked is typed "
                        f"differently from the size spec: "
                        f"type(x) ({type(x)}) incompatible with "
                        f"type(size) ({type(size)}")
    if isinstance(x, (Stacked1DTensor, DoublyStacked1DTensor, torch.Tensor)):
        x_size = x.size()
        if len(x_size) != len(size):
            raise AssertionError(f"number of dimensions mismatch: "
                                 f"len({x_size}) != len({size})")
        for i, (a, b) in enumerate(zip(x_size, size)):
            if b == -1:
                continue
            if a != b:
                raise AssertionError(f"dimension size mismatch at {i}-th "
                                     f"dimension: {a} != {b}")
    elif isinstance(x, tuple):
        for _x, _size in zip(x, size):
            check_size(_x, _size)
    else:
        raise ValueError(f"unsupported x type: {type(x)}")


def test_load_woz():
    data = WoZAdapter().load(pathlib.Path("tests/data/woz-mini"))
    print(data.keys())
    data = WoZAdapter().load(pathlib.Path("tests/data/woz-mini"), "test")
    print(data.keys())
    data = [d for ds in data.values() for d in ds]
    for dialog in random.sample(data, 3):
        print(dialog)


def hash_json(data, ignore_order=False):
    if isinstance(data, dict):
        data = tuple(hash((hash(k), hash_json(v, ignore_order)))
                     for k, v in data.items())
        if ignore_order:
            data = tuple(sorted(data))
        return hash(data)
    elif isinstance(data, list):
        data = tuple(hash_json(x, ignore_order) for x in data)
        if ignore_order:
            data = tuple(sorted(data))
        return hash(data)
    else:
        return hash(data)


def test_convert_woz():
    adapter = WoZAdapter()
    data_path = pathlib.Path("examples/train-woz.json")
    converted_path = tempfile.mktemp()
    original_data = load_json(str(data_path))
    adapter.save_json(
        data=adapter.load_json(pathlib.Path("examples/train-woz.json")),
        path=pathlib.Path(converted_path)
    )
    converted_data = load_json(str(converted_path))
    original_data = {data["dialogue_idx"]: data for data in original_data}
    converted_data = {data["dialogue_idx"]: data for data in converted_data}
    for i, (g, p) in enumerate(zip(original_data, converted_data)):
        hash_g = hash_json(g, ignore_order=True)
        hash_p = hash_json(p, ignore_order=True)
        assert hash_g == hash_p, f"{g} != {p}"


def test_woz_dataloading():
    dialogs = WoZAdapter().load(pathlib.Path("tests/data/woz-mini"), "train")
    proc = DialogProcessor(
        sent_processor=SentProcessor(
            bos=True,
            eos=True,
            lowercase=True
        ),
        boc=True,
        eoc=True
    )
    proc.prepare_vocabs(dialogs["train"])
    dset = DialogDataset(
        data=dialogs["train"],
        processor=proc
    )
    data = create_dataloader(
        dataset=dset,
        batch_size=2,
        shuffle=True
    )
    sample = next(iter(data)).to_dict()
    sent, speaker, state = sample["sent"], sample["speaker"], sample["state"]
    check_size(sent, torch.Size([2, -1, -1]))
    check_size(speaker, torch.Size([2, -1]))
    check_size(state, torch.Size([2, -1, -1]))
    print(sent.size(), speaker.size(), state.size())
    pprint.pprint(dset.data[0].to_json())


if __name__ == "__main__":
    test_load_woz()
