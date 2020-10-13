__all__ = ["AbstractSequenceEncoder",
           "SelfAttentiveSequenceEncoder",
           "RNNSequenceEncoder"]

import torch
import torch.nn as nn
import torchmodels

from .attention import ShallowSelfAttention2
from .rnn import AbstractRNN


class AbstractSequenceEncoder(torchmodels.Module):

    def __init__(self, input_dim, query_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, lens=None, q=None):
        """
        Arguments:
             x (FloatTensor): [N x seq_len x input_dim]
             lens (LongTensor): [N]
             q (FloatTensor): [N x query_dim]

        Returns:
            final_output (FloatTensor): [N x hidden_dim]
        """
        raise NotImplementedError


class SelfAttentiveSequenceEncoder(AbstractSequenceEncoder):
    name = "self-attentional-rnn"

    def __init__(self, *args, rnn=AbstractRNN, dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnn_cls = rnn
        self.dropout = dropout

        self.rnn = self.rnn_cls(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        )
        self.attention = ShallowSelfAttention2(
            input_dim=self.hidden_dim + self.query_dim + self.hidden_dim,
            hidden_dim=self.hidden_dim
        )
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, lens=None, q=None):
        batch_size, seq_len, _ = x.size()
        o, c, h = self.rnn(x, lens)
        if q is None:
            q = x.new(batch_size, 0)
        o = self.dropout_layer(o)
        u = torch.cat([o, q.unsqueeze(1).expand(-1, o.size(1), -1),
                       h.unsqueeze(1).expand(-1, o.size(1), -1)], -1)
        h = self.dropout_layer(self.attention(o, u, lens))
        return o, h


class RNNSequenceEncoder(AbstractSequenceEncoder):
    name = "rnn"

    def __init__(self, *args, rnn=AbstractRNN, dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rnn_cls = rnn
        self.dropout = dropout

        self.rnn = self.rnn_cls(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        )
        self.output_layer = nn.Linear(
            in_features=self.hidden_dim + self.query_dim,
            out_features=self.hidden_dim
        )
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, lens=None, q=None):
        batch_size, seq_len, _ = x.size()
        o, c, h = self.rnn(x, lens)
        if q is None:
            q = x.new(batch_size, 0)
        o = self.dropout_layer(o)
        h = self.output_layer(torch.cat([self.dropout_layer(h), q], -1))
        return o, h
