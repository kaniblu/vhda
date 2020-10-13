__all__ = ["BeamSearcher"]

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from utils import private_field
from utils import Vocabulary


@dataclass
class BeamSearcher:
    """
    Init Arguments:
        embedder (Callable): an embedding object that has the following
            call signature:
                Call Arguments:
                    input (LongTensor): [N x seq_len]
                    lens (optional, LongTensor): [N]
                Call Returns:
                    output (FloatTensor): [N x seq_lenx embed_dim]
        cell (Callable): a rnn-cell-like object that has the following call
            signature:
                Call Arguments:
                    input (FloatTensor): [N x embed_dim]
                    state (optional, FloatTensor or tuple of):
                        [N x hidden_dim], ...
                Call Returns:
                    output (FloatTensor): [N x hidden_dim]
                    next_state (FloatTensor, FloatTensor or tuple of):
                        [N x hidden_dim], ...
        classifier (Callable): a feedforward-like object that has the following
            call signature:
                Call Arguments:
                    input (FloatTensor): [N1 x ... x Nn x hidden_dim]
                Call Returns:
                    output (FloatTensor): [N1 x ... x Nn x vocab_size]
    """
    embedder: Callable
    cell: Callable
    classifier: Callable
    vocab: Vocabulary
    beam_size: int = 8
    max_len: int = 30
    bos: str = "<bos>"
    eos: str = "<eos>"
    _eos_idx: Optional[int] = private_field(default=None)

    def __post_init__(self):
        if self.bos not in self.vocab:
            raise ValueError(f"bos token {self.bos} not found in vocab.")
        if self.eos not in self.vocab:
            raise ValueError(f"eos token {self.eos} not found in vocab.")
        self._eos_idx = self.vocab[self.eos]

    def search(self, s0):
        """Perform beam search.

        Arguments:
            s0 (torch.FloatTensor or tuple of): initial cell state
                [N x hidden_dim], ...

        Returns:
            sample (torch.LongTensor): [N x beam_size x seq_len]
            lens (torch.LongTensor): [N x beam_size]
            prob (torch.FloatTensor): [N x beam_size]
        """
        s0_sample = s0[0] if isinstance(s0, tuple) else s0
        batch_size = s0_sample.size(0)
        logit0 = torch.zeros(len(self.vocab)).to(s0_sample).fill_(float("-inf"))
        logit0[self.vocab[self.bos]] = 0
        word = s0_sample.new(batch_size, self.beam_size, 0).long()
        done = s0_sample.new(batch_size, self.beam_size).fill_(0).bool()
        prob = s0_sample.new(batch_size, self.beam_size).fill_(1.0)
        lens = s0_sample.new(batch_size, self.beam_size).fill_(0).long()
        s = (tuple(t.unsqueeze(1)
                   .expand(batch_size, self.beam_size, -1).contiguous()
                   for t in s0)
             if isinstance(s0, tuple) else s0)
        while not done.all() and lens.max() < self.max_len:
            seq_len = word.size(2)
            if not seq_len:
                logit = logit0
                prob_prime, word_prime = torch.softmax(logit, 0).sort(0, True)
                prob_prime = prob_prime[:self.beam_size]
                word_prime = word_prime[:self.beam_size]
                prob = prob_prime.unsqueeze(0).expand(batch_size, -1)
                word = (word_prime.unsqueeze(0).unsqueeze(-1)
                        .expand(batch_size, -1, -1)).contiguous()
                lens += 1
                continue
            emb = self.embedder(word.view(-1, word.size(-1)), lens.view(-1))
            emb = emb[:, -1, :].view(batch_size, self.beam_size, -1)
            if isinstance(s, tuple):
                s_flat = tuple(t.view(-1, t.size(-1)) for t in s)
            else:
                s_flat = s.view(-1, s.size(-1))
            o, s_prime = self.cell(emb.view(-1, emb.size(-1)), s_flat)
            if isinstance(s_prime, tuple):
                s = tuple(t.view(batch_size, self.beam_size, -1)
                          for t in s_prime)
            else:
                s = s_prime.view(batch_size, self.beam_size, -1)
            logit = self.classifier(o).view(batch_size, self.beam_size, -1)
            if self._eos_idx is not None:
                logit_eos = torch.full_like(logit, float("-inf"))
                logit_eos[:, :, self._eos_idx] = 0
                logit = logit.masked_scatter(done.unsqueeze(-1), logit_eos)
            vocab_size = logit.size(-1)
            prob_prime = prob.unsqueeze(-1) * torch.softmax(logit, 2)
            prob_prime, prob_idx = \
                prob_prime.view(batch_size, -1).sort(1, True)
            prob = prob_prime[:, :self.beam_size]
            prob_idx = prob_idx[:, :self.beam_size].long()
            beam_idx = prob_idx / vocab_size
            word_prime = prob_idx % vocab_size
            word = torch.cat([
                word.gather(1, beam_idx.unsqueeze(-1).expand_as(word)),
                word_prime.unsqueeze(-1)
            ], 2)
            lens = lens.gather(1, beam_idx) + (~done).long()
            if self._eos_idx is not None:
                done = (word_prime == self._eos_idx) | done.gather(1, beam_idx)
        return word, lens, prob
