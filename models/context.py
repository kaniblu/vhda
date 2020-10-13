__all__ = ["AbstractContextEncoder", "LSTMContextEncoder"]

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torchmodels

import utils


class AbstractContextEncoder(torchmodels.Module):

    def __init__(self, input_dim, ctx_dim):
        super(AbstractContextEncoder, self).__init__()
        self.input_dim = input_dim
        self.ctx_dim = ctx_dim

    def init_state(self, n: int):
        raise NotImplementedError

    def forward(self, x, lens=None, h=None):
        """
        Arguments:
            x (FloatTensor): [N x seq_len x input_dim]
            lens (optional, FloatTensor): [N]
            h (optional, FloatTensor): *

        Returns:
            hidden_outputs (FloatTensor): [N x seq_len x ctx_dim]
            final_output (FloatTensor): [N x ctx_dim]
            next_h (FloatTensor): *
        """
        raise NotImplementedError


class LSTMContextEncoder(AbstractContextEncoder):
    name = "lstm"

    def __init__(self, *args, num_layers=1, dropout=0.0, **kwargs):
        super(LSTMContextEncoder, self).__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.dropout = dropout

        self.rnn = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.ctx_dim,
            bidirectional=False,
            num_layers=self.num_layers,
            dropout=self.dropout
        )

    def init_state(self, n: int):
        p = next(iter(self.rnn.parameters()))
        h = p.new(n, self.num_layers * self.ctx_dim * 2)
        return h.zero_()

    def pack_state(self, h, c):
        batch_size = h.size(1)
        return (torch.cat([h, c], 2).permute(1, 0, 2)
                .contiguous().view(batch_size, -1))

    def unpack_state(self, s):
        s = s.view(s.size(0), -1, self.ctx_dim * 2).permute(1, 0, 2)
        return s[..., :self.ctx_dim], s[..., self.ctx_dim:]

    def forward(self, x, lens=None, h=None):
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.permute(1, 0, 2)
        if lens is not None:
            x = rnn.pack_padded_sequence(x, lens, enforce_sorted=False)
        if h is not None:
            h = self.unpack_state(h)
        o, h_prime = self.rnn(x, h)
        if lens is not None:
            o, _ = rnn.pad_packed_sequence(o, total_length=seq_len)
        o = o.permute(1, 0, 2)
        if lens is not None:
            o = o.masked_fill(~utils.mask(lens, seq_len).unsqueeze(-1), 0)
        return o, h_prime[0][-1], self.pack_state(*h_prime)

    def reset_parameters(self):
        super(LSTMContextEncoder, self).reset_parameters()
        utils.init_lstm(self.rnn)
