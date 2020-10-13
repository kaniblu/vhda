__all__ = ["AbstractSentEncoder", "RNNSentEncoder"]

import torchmodels
from torchmodels.modules.pooling import AbstractPooling
from torchmodels.modules.feedforward import AbstractFeedForward

import utils
from .word import AbstractWordEncoder
from .rnn import AbstractRNN


class AbstractSentEncoder(torchmodels.Module):

    def __init__(self, vocab: utils.Vocabulary, output_dim):
        super(AbstractSentEncoder, self).__init__()
        self.vocab = vocab
        self.output_dim = output_dim

    def forward(self, w, lens=None):
        """
        Arguments:
             w (LongTensor): [N x seq_len]
             lens (LongTensor): [N]

        Returns:
            final_output (FloatTensor): [N x hidden_dim]
        """
        raise NotImplementedError


class RNNSentEncoder(AbstractSentEncoder):
    name = "rnn"

    def __init__(self, *args, word_dim=300, hidden_dim=512,
                 word_encoder=AbstractWordEncoder,
                 rnn=AbstractRNN,
                 pooling=AbstractPooling,
                 output_layer=AbstractFeedForward, **kwargs):
        super(RNNSentEncoder, self).__init__(*args, **kwargs)
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.word_cls = word_encoder
        self.rnn_cls = rnn
        self.pooling_cls = pooling
        self.output_layer_cls = output_layer

        self.word_encoder = self.word_cls(
            vocab=self.vocab,
            word_dim=self.word_dim
        )
        self.rnn = self.rnn_cls(
            input_dim=self.word_dim,
            hidden_dim=self.hidden_dim
        )
        self.pooling = self.pooling_cls(
            dim=self.hidden_dim
        )
        self.output_layer = self.output_layer_cls(
            input_dim=self.hidden_dim,
            output_dim=self.output_dim
        )

    def forward(self, w, lens=None):
        w = self.word_encoder(w, lens)
        o, _, _ = self.rnn(w, lens)
        o = self.pooling(o, lens)
        o = o.masked_fill((lens == 0).unsqueeze(-1), 0)
        return self.output_layer(o)
