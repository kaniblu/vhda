__all__ = ["AbstractSentDecoder", "RNNSentDecoder"]

import torch
import torchmodels

import utils
from .word import AbstractWordEncoder
from .decoding_rnn import AbstractDecodingRNN
from .beamsearch import BeamSearcher


class AbstractSentDecoder(torchmodels.Module):

    def __init__(self, vocab: utils.Vocabulary,
                 word_encoder: AbstractWordEncoder, hidden_dim):
        super(AbstractSentDecoder, self).__init__()
        self.vocab = vocab
        self.word_encoder = word_encoder
        self.hidden_dim = hidden_dim

    def inference(self, h, w, lens=None, **kwargs):
        raise NotImplementedError

    def generate(self, h, **kwargs):
        raise NotImplementedError

    def forward(self, h, w, lens=None, **kwargs):
        """
        Arguments:
            h (FloatTensor): [N x hidden_dim]
            w (LongTensor): [N x seq_len x word_dim]
            lens (optional, LongTensor): [N]

        Returns:
            logits (FloatTensor): [N x seq_len x vocab_size]
        """
        return self.inference(h, w, lens, **kwargs)


class RNNSentDecoder(AbstractSentDecoder):
    name = "rnn"

    def __init__(self, *args, rnn_dim=200,
                 decoding_rnn=AbstractDecodingRNN, **kwargs):
        super(RNNSentDecoder, self).__init__(*args, **kwargs)
        self.rnn_dim = rnn_dim
        self.rnn_cls = decoding_rnn

        self.rnn = self.rnn_cls(
            input_dim=self.word_dim,
            init_dim=self.hidden_dim,
            hidden_dim=self.rnn_dim
        )
        self.linear = torchmodels.Linear(
            in_features=self.rnn_dim,
            out_features=self.word_dim
        )

    @property
    def word_dim(self):
        return self.word_encoder.word_dim

    def output_layer(self, o):
        o_size = o.size()
        emb = self.word_encoder.weight.permute(1, 0)
        o = self.linear(o).view(-1, self.word_dim)
        return torch.mm(o, emb).view(*o_size[:-1], -1)

    def inference(self, h, w, lens=None, **kwargs):
        return self.output_layer(self.rnn(h, w, lens))

    def generate(self, h, beam_size=8, max_len=30, **kwargs):
        pred, lens, prob = BeamSearcher(
            vocab=self.vocab,
            embedder=self.word_encoder,
            cell=self.rnn.forward_cell,
            classifier=self.output_layer,
            beam_size=beam_size,
            max_len=max_len
        ).search(self.rnn.encode_hidden_state(h))
        return pred[:, 0], lens[:, 0], prob[:, 0]
