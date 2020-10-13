__all__ = ["mask", "init_gru", "init_lstm", "has_err", "has_inf", "has_nan",
           "Module", "Stacked1DTensor", "DoublyStacked1DTensor", "pad_stack",
           "to_sparse", "to_dense", "TriplyStacked1DTensor", "shift",
           "log_sum_exp", "count_parameters", "compare_tensors",
           "cat_stacked_tensors", "stack_stacked1dtensors", "sigmoid_inf"]

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torchmodels
import torch
import torch.nn as nn
import torch.nn.init as I

from .sugar import prod


def log_sum_exp(value, dim=None, keepdim=False):
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


class Module(torchmodels.Module):

    def __init__(self):
        super(Module, self).__init__()

        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def logger(self):
        return self._logger


def mask(lens: torch.Tensor, max_len=None):
    size = lens.size()
    lens = lens.view(-1)
    if max_len is None:
        max_len = lens.max().item()
    enum = torch.arange(0, max_len).to(lens)
    return (lens.unsqueeze(1) > enum.unsqueeze(0)).view(*size, max_len)


def init_gru(cell, gain=1):
    cell.reset_parameters()
    # orthogonal initialization of recurrent weights
    for _, hh, _, _ in cell.all_weights:
        for i in range(0, hh.size(0), cell.hidden_size):
            I.orthogonal_(hh[i:i + cell.hidden_size], gain=gain)


def init_lstm(cell, gain=1):
    init_gru(cell, gain)
    # positive forget gate bias (Jozefowicz et al., 2015)
    for _, _, ih_b, hh_b in cell.all_weights:
        x = len(ih_b)
        ih_b[x // 4:x // 2].data.fill_(1.0)
        hh_b[x // 4:x // 2].data.fill_(1.0)


def has_nan(tensor: torch.Tensor):
    return bool((tensor != tensor).any().item())


def has_inf(tensor: torch.Tensor):
    return bool(((tensor == float("inf")) |
                 (tensor == float("-inf"))).any().item())


def has_err(tensor: torch.Tensor):
    return has_nan(tensor) or has_inf(tensor)


def compare_tensors(a, b):
    return a.size() == b.size() and bool(a.eq(b).all().item())


@dataclass(frozen=True)
class Stacked1DTensor:
    value: torch.Tensor
    lens: torch.Tensor

    def __post_init__(self):
        # sanity check
        value, sizes = self.value, self.lens
        if len(sizes.view(-1)) == 0:
            return
        if len(sizes.size()) == 1:
            sizes = sizes.unsqueeze(-1)
        assert value.size(0) == sizes.size(0)
        assert len(value.size()) == sizes.size(1) + 1
        for i, (v, s) in enumerate(zip(value.size()[1:], sizes.t())):
            assert v >= s.max(), \
                f"{i}-th dimension cannot contain all sizes: {v} < {s.max()}"

    @property
    def tensors(self):
        return self.value, self.lens

    def __len__(self):
        return self.value.size(0)

    def to(self, device):
        return Stacked1DTensor(
            value=self.value.to(device),
            lens=self.lens.to(device)
        )

    def narrow(self, dim, start, length):
        if dim == 0:
            end = start + length
            lens = self.lens[start:end]
            return Stacked1DTensor(self.value[start:end, :lens.max()], lens)
        elif dim == 1:
            end = start + length
            return Stacked1DTensor(
                value=self.value[:, start:end],
                lens=(self.lens - start).clamp_(0, length)
            )
        else:
            raise ValueError(f"dimension must be 0 or 1")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"unsupported type: {type(other)}")
        return (compare_tensors(self.value, other.value) and
                compare_tensors(self.lens, other.lens))

    def __getitem__(self, item):
        if isinstance(item, str):
            raise TypeError(f"string type index not supported: '{item}'")
        if isinstance(item, (slice, Sequence, torch.Tensor)):
            lens = self.lens[item]
            return Stacked1DTensor(
                value=self.value[item, :lens.max()],
                lens=lens
            )
        elif isinstance(item, int):
            return self.value[item, :self.lens[item]]
        else:
            raise TypeError(f"unsupported index type: {type(item)}")

    def __iter__(self):
        for i in range(self.size(0)):
            yield self[i]

    def size(self, *args, **kwargs):
        return self.value.size(*args, **kwargs)


def cat_stacked_tensors(tensors):
    vectors = []
    for tensor in tensors:
        if not isinstance(tensor, Stacked1DTensor):
            raise TypeError(f"unsupported type: {type(tensor)}")
        for item in tensor:
            vectors.append(item)
    return pad_stack(vectors)


@dataclass(frozen=True)
class DoublyStacked1DTensor(Stacked1DTensor):
    lens1: torch.Tensor

    def __post_init__(self):
        # sanity check
        value, sizes, lens = self.value, self.lens1, self.lens
        if len(sizes.size()) == 2:
            sizes = sizes.unsqueeze(-1)
        assert value.size()[:2] == sizes.size()[:2]
        assert len(value.size()) == sizes.size(2) + 2
        for i, (v, s) in enumerate(zip(value.size()[2:], sizes)):
            assert v >= s.max(), \
                f"{i}-th dimension cannot contain all sizes: {v} < {s.max()}"
        assert value.size(0) == sizes.size(0) == lens.size(0)

    def to(self, device):
        return DoublyStacked1DTensor(
            value=self.value.to(device),
            lens=self.lens.to(device),
            lens1=self.lens1.to(device)
        )

    def narrow(self, dim, start, length):
        if dim == 0:
            end = start + length
            lens = self.lens[start:end]
            lens1 = self.lens1[start:end, :lens.max()]
            return DoublyStacked1DTensor(
                value=self.value[start:end, :lens.max(), :lens1.max()],
                lens=lens,
                lens1=lens1
            )
        elif dim == 1:
            end = start + length
            lens1 = self.lens1[:, start:end]
            return DoublyStacked1DTensor(
                value=self.value[:, start:end, lens1.max()],
                lens=(self.lens - start).clamp_(0, length),
                lens1=lens1
            )
        elif dim == 2:
            end = start + length
            return DoublyStacked1DTensor(
                value=self.value[:, :, start:end],
                lens=self.lens,
                lens1=(self.lens1 - start).clamp_(0, length)
            )
        else:
            raise ValueError(f"dimension must be 0, 1, or 2")

    def __eq__(self, other):
        return (super().__eq__(other) and
                compare_tensors(self.lens1, other.lens1))

    def __getitem__(self, item):
        if isinstance(item, str):
            raise TypeError(f"string type index not supported: '{item}'")
        if isinstance(item, (slice, Sequence, torch.Tensor)):
            lens = self.lens[item]
            lens1 = self.lens1[item, :lens.max()]
            return DoublyStacked1DTensor(
                value=self.value[item, :lens.max(), :lens1.max()],
                lens=lens,
                lens1=lens1
            )
        elif isinstance(item, int):
            lens = self.lens1[item, :self.lens[item]]
            return Stacked1DTensor(
                value=self.value[item, :self.lens[item], :lens.max()],
                lens=lens
            )
        else:
            raise TypeError(f"unsupported index type: {type(item)}")


@dataclass(frozen=True)
class TriplyStacked1DTensor(DoublyStacked1DTensor):
    lens2: torch.Tensor

    def __post_init__(self):
        # TODO: sanity check
        pass

    def to(self, device):
        ret = super().to(device)
        return TriplyStacked1DTensor(*ret.tensors, self.lens2.to(device))


def pad_stack(tensors: Sequence[torch.Tensor]):
    sizes = [t.size() for t in tensors]
    num_dims = len(sizes[0])
    if any(len(size) != num_dims for size in sizes):
        raise ValueError(f"number of dimensions must be the same: {sizes}")
    max_size = tuple(map(max, zip(*sizes)))
    ret = []
    for tensor in tensors:
        if tuple(tensor.size()) == max_size:
            ret.append(tensor)
            continue
        padded_tensor = tensor.new(*max_size).fill_(0.0)
        idx = tuple(slice(0, s) for s in tensor.size())
        padded_tensor[idx] = tensor
        ret.append(padded_tensor)
    lens = torch.LongTensor(sizes).to(tensors[0].device)
    if num_dims == 1:
        lens = lens.squeeze(-1)
    return Stacked1DTensor(torch.stack(ret), lens)


def stack_stacked1dtensors(tensors: Sequence[Stacked1DTensor]
                           ) -> DoublyStacked1DTensor:
    x = pad_stack([t.value for t in tensors])
    s = pad_stack([t.lens for t in tensors])
    return DoublyStacked1DTensor(
        value=x.value,
        lens=s.lens,
        lens1=s.value
    )


def to_sparse(x) -> Stacked1DTensor:
    """Converts a dense ByteTensor into a indexed LongTensor.

    Arguments:
        x ([... x N] ByteTensor)

    Returns:
        idx ([... x max_len] LongTensor)
        lens ([...] LongTensor)
    """
    size, n = x.size()[:-1], x.size(-1)
    x = x.view(-1, n)
    idx = torch.arange(n, device=x.device).unsqueeze(0).expand(x.size(0), -1)
    idx = idx.masked_fill(~x, -1)
    idx, _ = idx.sort(-1, True)
    lens = x.sum(1)
    max_len = lens.max().item()
    idx = idx[:, :max_len].masked_fill(~mask(lens, max_len), 0)
    return Stacked1DTensor(idx.view(*size, -1), lens.view(*size))


def to_dense(idx, lens, max_size=None):
    """Converts a sparse indices into a mask (ByteTensor)."""
    size, max_len = idx.size()[:-1], idx.size(-1)
    idx, lens = idx.view(prod(size), max_len), lens.view(-1)
    idx = idx.masked_fill(~mask(lens), 0)
    if max_size is None:
        max_size = idx.max().item() + 1
    x = torch.zeros(idx.size(0), max_size, device=idx.device).long()
    x = x.scatter_add(-1, idx, torch.ones_like(x))
    x[:, 0] -= (max_len - lens).long()
    return (x > 0).view(*size, max_size)


def shift(x: torch.Tensor, n: int = 1, dim=0, wrap=False):
    size = tuple(x.size())
    length = x.size(dim)
    if abs(n) > length:
        raise ValueError(f"cannot roll more than the dimension size: "
                         f"{n} > {length}")
    neg, n = n < 0, abs(n)
    if n == 0:
        return x
    if not neg:
        tensors = []
        if wrap:
            tensors.append(x.narrow(dim, length - n, n))
        else:
            pad_size = size[:dim] + (n,) + size[dim + 1:]
            tensors.append(x.new(*pad_size).zero_())
        tensors.append(x.narrow(dim, 0, length - n))
    else:
        tensors = [x.narrow(dim, n, length - n)]
        if wrap:
            tensors.append(x.narrow(dim, 0, n))
        else:
            pad_size = size[:dim] + (n,) + size[dim + 1:]
            tensors.append(x.new(*pad_size).zero_())
    return torch.cat(tensors, dim)


def count_parameters(model: nn.Module) -> int:
    return int(sum(np.prod(p.size())
                   for p in model.parameters() if p.requires_grad))


def sigmoid_inf(x):
    zero_mask = x == float("-inf")
    one_mask = x == float("inf")
    return (torch.sigmoid(x.masked_fill(zero_mask, 0).masked_fill(one_mask, 0))
            .masked_fill(zero_mask, 0).masked_fill(one_mask, 1))
