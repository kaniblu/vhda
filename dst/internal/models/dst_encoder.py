__all__ = ["AbstractDSTEncoder", "GLADEncoder"]

import torch
import torch.nn as nn
import torchmodels

import nltk

import utils
import models.rnn
import models.word
from .attention import ShallowSelfAttention
from .attention import ShallowSelfAttention2
from .attention import MultiheadSelfAttention
from datasets import StateVocabSet


class AbstractDSTEncoder(torchmodels.Module):
    """An abstract class for general encoders used in Dialogue State Tracking.
    A general signature of this type of encoder is

        encoder_slot(X) -> H, c

    Where X is a 2d input feature with the length of the first dimension being
    variable
    """

    def __init__(self, ontology: StateVocabSet, input_dim, hidden_dim):
        super().__init__()
        self.ontology = ontology
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, slot, x, lens):
        """
        Arguments:
            slot: [] LongTensor
            x: [batch_size x N x input_dim] FloatTensor
            lens: [batch_size] LongTensor

        Returns:
            H: [batch_size x N x hidden_dim] FloatTensor,
            c: [batch_size x hidden_dim] FloatTensor
        """
        raise NotImplementedError


class GLADEncoder(AbstractDSTEncoder):
    """arXiv:1805.09655"""
    name = "glad"

    def __init__(self, *args,
                 global_dropout=0.2,
                 local_dropout=0.2,
                 separate_gating=False,
                 rnn_global=models.rnn.AbstractRNN,
                 rnn_local=models.rnn.AbstractRNN, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_dropout = global_dropout
        self.local_dropout = local_dropout
        self.separate_gating = separate_gating
        self.rnn_global_cls = rnn_global
        self.rnn_local_cls = rnn_local

        self.rnn_global = self.rnn_global_cls(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        )
        self.rnn_local = torchmodels.ModuleList([self.rnn_local_cls(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        ) for _ in range(self.num_slots)])
        self.beta1 = nn.Parameter(torch.zeros(self.num_slots))
        if self.separate_gating:
            self.beta2 = nn.Parameter(torch.zeros(self.num_slots))
        else:
            self.beta2 = None
        self.attention_global = ShallowSelfAttention(self.hidden_dim)
        self.attention_local = torchmodels.ModuleList([
            ShallowSelfAttention(self.hidden_dim)
            for _ in range(self.num_slots)
        ])
        self.dropout_g = nn.Dropout(self.global_dropout)
        self.dropout_l = nn.Dropout(self.local_dropout)

    @property
    def num_slots(self):
        return len(self.ontology.act_slot)

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.uniform_(self.beta1, -0.01, 0.01)
        if self.separate_gating:
            nn.init.uniform_(self.beta2, -0.01, 0.01)

    def forward(self, slot, x, lens):
        h_g, _, _ = self.rnn_global(x, lens)
        h_l, _, _ = self.rnn_local[slot.item()](x, lens)
        beta1 = torch.sigmoid(self.beta1[slot])
        h = self.dropout_g(h_g) * beta1 + self.dropout_l(h_l) * (1 - beta1)
        c_g = self.attention_global(h, lens)
        c_l = self.attention_local[slot.item()](h, lens)
        if self.separate_gating:
            beta2 = torch.sigmoid(self.beta2[slot])
        else:
            beta2 = beta1
        c = self.dropout_g(c_g) * beta2 + self.dropout_l(c_l) * (1 - beta2)
        return h, c


class GCEEncoder(AbstractDSTEncoder):
    """arXiv:1812.00899"""
    name = "gce"

    def __init__(self, *args,
                 dropout=0.2,
                 token_dim=300,
                 act_slot_dim=100,
                 token_encoder=models.word.AbstractWordEncoder,
                 act_slot_rnn=models.rnn.AbstractRNN,
                 base_rnn=models.rnn.AbstractRNN, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout
        self.token_dim = token_dim
        self.act_slot_dim = act_slot_dim
        self.token_encoder_cls = token_encoder
        self.act_slot_rnn_cls = act_slot_rnn
        self.base_rnn_cls = base_rnn

        self.vocab = self.create_token_vocab()
        self.token_encoder = self.token_encoder_cls(
            vocab=self.vocab,
            word_dim=self.token_dim
        )
        as_tensor = self.create_act_slot_tensor(self.vocab)
        self.act_slot_tensor = \
            nn.Parameter(self.token_encoder(as_tensor.value), False)
        self.act_slot_lens = nn.Parameter(as_tensor.lens, False)
        self.act_slot_rnn = self.act_slot_rnn_cls(
            input_dim=self.token_dim,
            hidden_dim=self.act_slot_dim
        )
        self.base_rnn = self.base_rnn_cls(
            input_dim=self.input_dim + self.act_slot_dim,
            hidden_dim=self.hidden_dim
        )
        self.attention = ShallowSelfAttention2(
            input_dim=self.hidden_dim + self.act_slot_dim,
            hidden_dim=self.hidden_dim
        )
        self.dropout_layer = nn.Dropout(self.dropout)

    def create_act_slot_tensor(self, vocab):
        act_slots = []
        for as_idx, (act, slot) in self.ontology.act_slot.i2f.items():
            tokens = (list(nltk.casual_tokenize(act)) +
                      list(nltk.casual_tokenize(slot)) + ["<eos>"])
            tokens = [vocab[token] for token in tokens]
            act_slots.append((as_idx, torch.LongTensor(tokens)))
        act_slots = list(sorted(act_slots, key=lambda x: x[0]))
        act_slots = utils.pad_stack([act_slot[1] for act_slot in act_slots])
        return act_slots

    def create_token_vocab(self):
        factory = utils.VocabularyFactory(reserved=["<eos>"])
        for act in self.ontology.act.f2i:
            factory.update(nltk.casual_tokenize(act))
        for slot in self.ontology.slot.f2i:
            factory.update(nltk.casual_tokenize(slot))
        return factory.get_vocab()

    @property
    def num_slots(self):
        return len(self.ontology.act_slot)

    def forward(self, slot, x, lens):
        act_slot_tokens = self.act_slot_tensor[slot, :self.act_slot_lens[slot]]
        slot_emb, _, _ = self.act_slot_rnn(act_slot_tokens.unsqueeze(0))
        slot_emb = slot_emb.squeeze(0).max(0)[0].unsqueeze(0).unsqueeze(0)
        slot_emb = slot_emb.expand(x.size(0), x.size(1), -1)
        h, _, _ = self.base_rnn(torch.cat([x, slot_emb], -1), lens)
        h = self.dropout_layer(h)
        c = self.attention(h, torch.cat([h, slot_emb], -1), lens)
        c = self.dropout_layer(c)
        return h, c


class GLADStarEncoder(AbstractDSTEncoder):
    """arXiv:1908.07795"""
    name = "glad-star"

    def __init__(self, *args,
                 num_heads=10,
                 att_dim=100,
                 global_dropout=0.2,
                 local_dropout=0.2,
                 separate_gating=False,
                 rnn_global=models.rnn.AbstractRNN,
                 rnn_local=models.rnn.AbstractRNN, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.att_dim = att_dim
        self.global_dropout = global_dropout
        self.local_dropout = local_dropout
        self.separate_gating = separate_gating
        self.rnn_global_cls = rnn_global
        self.rnn_local_cls = rnn_local

        self.rnn_global = self.rnn_global_cls(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        )
        self.rnn_local = nn.ModuleList([self.rnn_local_cls(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim
        ) for _ in range(self.num_slots)])
        self.beta1 = nn.Parameter(torch.zeros(self.num_slots))
        if self.separate_gating:
            self.beta2 = nn.Parameter(torch.zeros(self.num_slots))
        else:
            self.beta2 = None
        self.attention_global = MultiheadSelfAttention(
            hidden_dim=self.hidden_dim,
            att_dim=self.att_dim,
            num_heads=self.num_heads
        )
        self.attention_local = nn.ModuleList([
            MultiheadSelfAttention(
                hidden_dim=self.hidden_dim,
                att_dim=self.att_dim,
                num_heads=self.num_heads
            ) for _ in range(self.num_slots)
        ])
        self.dropout_g = nn.Dropout(self.global_dropout)
        self.dropout_l = nn.Dropout(self.local_dropout)
        self.output_layer = nn.Linear(
            in_features=self.num_heads * self.hidden_dim,
            out_features=self.hidden_dim
        )

    @property
    def num_slots(self):
        return len(self.ontology.act_slot)

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.uniform_(self.beta1, -0.01, 0.01)
        if self.separate_gating:
            nn.init.uniform_(self.beta2, -0.01, 0.01)

    def forward(self, slot, x, lens):
        h_g, _, _ = self.rnn_global(x, lens)
        h_l, _, _ = self.rnn_local[slot.item()](x, lens)
        beta1 = torch.sigmoid(self.beta1[slot])
        h = self.dropout_g(h_g) * beta1 + self.dropout_l(h_l) * (1 - beta1)
        c_g = self.attention_global(h, lens)
        c_l = self.attention_local[slot.item()](h, lens)
        if self.separate_gating:
            beta2 = torch.sigmoid(self.beta2[slot])
        else:
            beta2 = beta1
        c = self.dropout_g(c_g) * beta2 + self.dropout_l(c_l) * (1 - beta2)
        return h, self.output_layer(c.view(c.size(0), -1))


class SimpleRNNEncoder(AbstractDSTEncoder):
    """arXiv:1812.00899"""
    name = "simple-rnn"

    def __init__(self, *args,
                 dropout=0.2,
                 token_dim=300,
                 act_slot_dim=100,
                 token_encoder=models.word.AbstractWordEncoder,
                 act_slot_rnn=models.rnn.AbstractRNN,
                 base_rnn=models.rnn.AbstractRNN, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout = dropout
        self.token_dim = token_dim
        self.act_slot_dim = act_slot_dim
        self.token_encoder_cls = token_encoder
        self.act_slot_rnn_cls = act_slot_rnn
        self.base_rnn_cls = base_rnn

        self.vocab = self.create_token_vocab()
        self.token_encoder = self.token_encoder_cls(
            vocab=self.vocab,
            word_dim=self.token_dim
        )
        as_tensor = self.create_act_slot_tensor(self.vocab)
        self.act_slot_tensor = \
            nn.Parameter(self.token_encoder(as_tensor.value), False)
        self.act_slot_lens = nn.Parameter(as_tensor.lens, False)
        self.act_slot_rnn = self.act_slot_rnn_cls(
            input_dim=self.token_dim,
            hidden_dim=self.act_slot_dim
        )
        self.base_rnn = self.base_rnn_cls(
            input_dim=self.input_dim + self.act_slot_dim,
            hidden_dim=self.hidden_dim
        )
        self.dropout_layer = nn.Dropout(self.dropout)

    def create_act_slot_tensor(self, vocab):
        act_slots = []
        for as_idx, (act, slot) in self.ontology.act_slot.i2f.items():
            tokens = (list(nltk.casual_tokenize(act)) +
                      list(nltk.casual_tokenize(slot)) + ["<eos>"])
            tokens = [vocab[token] for token in tokens]
            act_slots.append((as_idx, torch.LongTensor(tokens)))
        act_slots = list(sorted(act_slots, key=lambda x: x[0]))
        act_slots = utils.pad_stack([act_slot[1] for act_slot in act_slots])
        return act_slots

    def create_token_vocab(self):
        factory = utils.VocabularyFactory(reserved=["<eos>"])
        for act in self.ontology.act.f2i:
            factory.update(nltk.casual_tokenize(act))
        for slot in self.ontology.slot.f2i:
            factory.update(nltk.casual_tokenize(slot))
        return factory.get_vocab()

    @property
    def num_slots(self):
        return len(self.ontology.act_slot)

    def forward(self, slot, x, lens):
        act_slot_tokens = self.act_slot_tensor[slot, :self.act_slot_lens[slot]]
        slot_emb, _, _ = self.act_slot_rnn(act_slot_tokens.unsqueeze(0))
        slot_emb = slot_emb.squeeze(0).max(0)[0].unsqueeze(0).unsqueeze(0)
        slot_emb = slot_emb.expand(x.size(0), x.size(1), -1)
        h, _, _ = self.base_rnn(torch.cat([x, slot_emb], -1), lens)
        mask = ~utils.mask(lens, h.size(1)).unsqueeze(-1)
        c = self.dropout_layer(h).masked_fill(mask, float("-inf")).max(1)[0]
        return h, c
