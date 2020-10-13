__all__ = ["AbstractDecodingRNN", "LSTMDecodingRNN"]

import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torchmodels

import utils


class AbstractDecodingRNN(torchmodels.Module):

    def __init__(self, input_dim, init_dim, hidden_dim):
        super(AbstractDecodingRNN, self).__init__()
        self.input_dim = input_dim
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim

    def encode_hidden_state(self, h):
        raise NotImplementedError

    def forward_cell(self, x, s):
        raise NotImplementedError

    def forward(self, h, x, lens=None):
        """
        Arguments:
            h (FloatTensor): [N x init_dim]
            x (FloatTensor): [N x seq_len x input_dim]
            lens (optional, LongTensor): [N]

        Returns:
            hidden_outputs (FloatTensor): [N x seq_len x hidden_dim]
        """
        raise NotImplementedError


class LSTMDecodingRNN(AbstractDecodingRNN):
    name = "lstm"

    def __init__(self, *args, num_layers=1, dropout=0.0, **kwargs):
        super(LSTMDecodingRNN, self).__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.dropout = dropout

        self.init_layer_c = torchmodels.Linear(
            in_features=self.init_dim,
            out_features=self.num_layers * self.hidden_dim
        )
        self.init_layer_h = torchmodels.Linear(
            in_features=self.init_dim,
            out_features=self.num_layers * self.hidden_dim
        )
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=False,
            dropout=self.dropout
        )

    def encode_hidden_state(self, h):
        h, c = self.init_layer_h(h), self.init_layer_c(h)
        return h, c

    def forward_cell(self, x, s):
        batch_size = x.size(0)
        h, c = s
        h = h.view(batch_size, -1, self.hidden_dim)
        c = c.view(batch_size, -1, self.hidden_dim)
        h, c = h.permute(1, 0, 2).contiguous(), c.permute(1, 0, 2).contiguous()
        o, (h_prime, c_prime) = self.lstm(x.unsqueeze(0), (h, c))
        h_prime = h_prime.permute(1, 0, 2).contiguous().view(batch_size, -1)
        c_prime = c_prime.permute(1, 0, 2).contiguous().view(batch_size, -1)
        return o.squeeze(0), (h_prime, c_prime)

    def forward(self, h, x, lens=None):
        batch_size, seq_len = h.size(0), x.size(1)
        h, c = self.encode_hidden_state(h)
        h = h.view(batch_size, -1, self.hidden_dim)
        c = c.view(batch_size, -1, self.hidden_dim)
        h, c = h.permute(1, 0, 2).contiguous(), c.permute(1, 0, 2).contiguous()
        x = x.permute(1, 0, 2)
        nil_mask = None
        if lens is not None:
            nil_mask = lens == 0
            if not nil_mask.any().item():
                nil_mask = None
            if nil_mask is not None:
                lens[nil_mask] = 1
            x = rnn.pack_padded_sequence(x, lens, enforce_sorted=False)
        o, _ = self.lstm(x, (h, c))
        if lens is not None:
            o, _ = rnn.pad_packed_sequence(o, total_length=seq_len)
        o = o.permute(1, 0, 2)
        if lens is not None:
            if nil_mask is not None:
                lens[nil_mask] = 0
            o = o.masked_fill(~utils.mask(lens).unsqueeze(-1), 0)
        return o

    def reset_parameters(self):
        super().reset_parameters()
        utils.init_lstm(self.lstm)
