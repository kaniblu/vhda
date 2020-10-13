__all__ = ["PaddingCollator"]

import utils
from dataclasses import dataclass
from typing import FrozenSet

import torch
import numpy as np


@dataclass
class PaddingCollator:
    pad_keys: FrozenSet[str] = frozenset()

    @staticmethod
    def stack(tensors, pad=False):
        if pad:
            return utils.pad_stack(tensors)
        else:
            return torch.stack(tensors)

    def __call__(self, batch, pad_override=False):
        if not len(batch):
            raise ValueError(f"must provide at least one sample")
        sample = batch[0]
        sample_type = type(sample)
        if sample_type == dict:
            return {k: self([s[k] for s in batch],
                            pad_override or (k in self.pad_keys))
                    for k in sample}
        elif sample_type == tuple:
            return tuple(self([s[i] for s in batch], pad_override)
                         for i in range(len(sample)))
        elif sample_type == int:
            return torch.LongTensor(batch)
        elif sample_type == float:
            return torch.FloatTensor(batch)
        elif sample_type == utils.Stacked1DTensor:
            x = utils.pad_stack([t.value for t in batch])
            s = utils.pad_stack([t.lens for t in batch])
            return utils.DoublyStacked1DTensor(
                value=x.value,
                lens=s.lens,
                lens1=s.value
            )
        elif sample_type == utils.DoublyStacked1DTensor:
            x = utils.pad_stack([t.value for t in batch])
            s1 = utils.pad_stack([t.lens for t in batch])
            s2 = utils.pad_stack([t.lens1 for t in batch])
            return utils.TriplyStacked1DTensor(
                value=x.value,
                lens=s1.value,
                lens1=s2.value,
                lens2=s2.lens
            )
        elif isinstance(sample, torch.Tensor):
            return self.stack(batch, pad_override)
        elif isinstance(sample, np.ndarray):
            return self.stack(list(map(torch.tensor, batch)), pad_override)
        else:
            return batch
