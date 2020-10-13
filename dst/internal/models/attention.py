__all__ = ["AbstractSelfAttention", "MultiheadSelfAttention",
           "ShallowSelfAttention", "ShallowSelfAttention2"]

import torch
import torch.nn as nn
import torchmodels

import utils


class AbstractSelfAttention(torchmodels.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, h, lens=None):
        """
        Arguments:
            h: [batch_size x N x dim] Tensor
            lens (optional): [batch_size] LongTensor

        Returns:
            o: [batch_size x dim] Tensor
        """
        raise NotImplementedError


class ShallowSelfAttention(AbstractSelfAttention):
    name = "shallow"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(self.dim, 1)

    def forward(self, h, lens=None):
        """
        Arguments:
            h: [batch_size x N x dim] Tensor
            lens (optional): [batch_size] LongTensor

        Returns:
            o: [batch_size x dim] Tensor
        """
        a = self.linear(h).squeeze(-1)
        if lens is not None:
            mask = ~utils.mask(lens, h.size(1))
            mask[lens == 0] = 0
            a = a.masked_fill(mask, float("-inf"))
        o = torch.bmm(torch.softmax(a, -1).unsqueeze(1), h).squeeze(1)
        if lens is not None and (lens == 0).any().item():
            o[lens == 0] = 0
        return o


class ShallowSelfAttention2(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(input_dim, 1)

    def forward(self, h, q, lens=None):
        """
        Arguments:
            h: [batch_size x N x hidden_dim] Tensor
            q: [batch_size x N x input_dim] Tensor
            lens (optional): [batch_size] LongTensor

        Returns:
            o: [batch_size x hidden_dim] Tensor
        """
        a = self.linear(q).squeeze(-1)
        if lens is not None:
            mask = ~utils.mask(lens, h.size(1))
            mask[lens == 0] = 0
            a = a.masked_fill(mask, float("-inf"))
        o = torch.bmm(torch.softmax(a, -1).unsqueeze(1), h).squeeze(1)
        if lens is not None and (lens == 0).any().item():
            o[lens == 0] = 0
        return o


class MultiheadSelfAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads=10, att_dim=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.att_dim = att_dim

        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(self.hidden_dim, self.att_dim, bias=False)
        self.linear2 = nn.Linear(self.att_dim, self.num_heads, bias=False)

    def forward(self, h, lens=None):
        """
        Arguments:
            h: [batch_size x N x dim] Tensor
            lens (optional): [batch_size] LongTensor

        Returns:
            o: [batch_size x dim] Tensor
        """
        a = self.linear2(self.tanh(self.linear1(h))).permute(0, 2, 1)
        if lens is not None:
            mask = ~utils.mask(lens, h.size(1))
            mask[lens == 0] = 0
            a = a.masked_fill(mask.unsqueeze(1), float("-inf"))
        o = torch.bmm(torch.softmax(a, 2), h)
        if lens is not None and (lens == 0).any().item():
            o[lens == 0] = 0
        return o
