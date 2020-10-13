__all__ = ["AbstractDialogStateTracker", "GenericDST"]

import torch
import torch.distributions as dist
import torch.nn as nn
import torchmodels

import utils
from datasets import VocabSet
from models.word import AbstractWordEncoder
from .dst_encoder import AbstractDSTEncoder


class AbstractDialogStateTracker(torchmodels.Module):

    def __init__(self, vocabs: VocabSet):
        super().__init__()
        self.vocabs = vocabs

    def forward(self, slot, w, w_lens, a, a_lats, a_lens, ont, ont_lens):
        """
        Arguments:
            slot: [] LongTensor
            w: [batch_size x max_sent_len] LongTensor
            w_lens: [batch_size] LongTensor
            a: [batch_size x max_act_lat x max_act_len] LongTensor
            a_lats: [batch_size] LongTensor
            a_lens: [batch_size x max_act_lat] LongTensor
            ont: [num_ont x max_ont_len] LongTensor
            ont_lens: [num_ont] LongTensor

        Returns:
            y: [batch_size x num_ont] FloatTensor
        """
        raise NotImplementedError


class GenericDST(AbstractDialogStateTracker):
    name = "generic"

    def __init__(self, *args,
                 word_dim=300,
                 hidden_dim=600,
                 word_dropout=0.0,
                 word_emb_dropout=0.0,
                 word_encoder=AbstractWordEncoder,
                 dst_encoder=AbstractDSTEncoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.word_dropout = word_dropout
        self.word_emb_dropout = word_emb_dropout
        self.word_encoder_cls = word_encoder
        self.dst_encoder_cls = dst_encoder

        self.word_encoder = self.word_encoder_cls(
            vocab=self.vocabs.word,
            word_dim=self.word_dim
        )
        self.sent_encoder = self.dst_encoder_cls(
            ontology=self.vocabs.speaker_state["user"],
            input_dim=self.word_dim,
            hidden_dim=self.hidden_dim
        )
        self.act_encoder = self.dst_encoder_cls(
            ontology=self.vocabs.speaker_state["user"],
            input_dim=self.word_dim,
            hidden_dim=self.hidden_dim
        )
        self.ont_encoder = self.dst_encoder_cls(
            ontology=self.vocabs.speaker_state["user"],
            input_dim=self.word_dim,
            hidden_dim=self.hidden_dim
        )
        self.sent_linear = nn.Linear(self.hidden_dim, 1)
        self.act_weight = nn.Parameter(torch.tensor(0.5))
        if self.word_dropout:
            self.word_dropout_dist = dist.Bernoulli(self.word_dropout)
        else:
            self.word_dropout_dist = None
        self.word_emb_dropout_layer = nn.Dropout(self.word_emb_dropout)

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.xavier_normal_(self.sent_linear.weight)

    @staticmethod
    def attend(h, q, lens=None):
        """
        Arguments:
            h: [batch_size x N x dim] FloatTensor
            q: [batch_size x M x dim] FloatTensor
            lens (optional): [batch_size] LongTensor

        Returns:
            [batch_size x M x dim] FloatTensor
        """
        a = torch.bmm(h, q.permute(0, 2, 1)).permute(0, 2, 1)
        if lens is not None:
            mask = ~utils.mask(lens, h.size(1))
            mask[lens == 0] = 0
            a = a.masked_fill(mask.unsqueeze(1), float("-inf"))
        o = torch.bmm(torch.softmax(a, -1), h)
        if lens is not None and (lens == 0).any().item():
            o[lens == 0] = 0
        return o

    def word_emb(self, w):
        emb = self.word_encoder(w)
        if self.training and self.word_dropout_dist is not None:
            emb[self.word_dropout_dist.sample(w.size()).bool()] = 0
        return self.word_emb_dropout_layer(emb)

    def forward(self, slot, w, w_lens, a, a_lats, a_lens, ont, ont_lens):
        batch_size, max_act_lat, max_act_len = a.size()
        h_w, c_w = self.sent_encoder(slot, self.word_emb(w), w_lens)
        c_a = self.act_encoder(
            slot,
            self.word_emb(a).view(-1, max_act_len, self.word_dim),
            a_lens.view(-1)
        )[1].view(batch_size, max_act_lat, -1)
        c_v = self.ont_encoder(slot, self.word_emb(ont), ont_lens)[1]
        q_w = self.attend(
            h=h_w,
            q=c_v.unsqueeze(0).expand(batch_size, -1, -1),
            lens=w_lens
        )
        y_w = self.sent_linear(q_w).squeeze(-1)
        q_a = self.attend(c_a, c_w.unsqueeze(1), a_lats).squeeze(1)
        y_a = torch.mm(q_a, c_v.t())
        return y_w + self.act_weight * y_a
