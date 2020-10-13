__all__ = ["AbstractWordEncoder", "EmbeddingWordEncoder",
           "PretrainedWordEncoder", "FastTextWordEncoder", "GloveWordEncoder",
           "HDF5WordEncoder", "CompositeWordEncoder", "VZhongWordEncoder"]

import importlib
from typing import Optional, Sequence

import torch
import torch.nn as nn

import utils
import embeds


class AbstractWordEncoder(utils.Module):

    def __init__(self, vocab: utils.Vocabulary, word_dim):
        super(AbstractWordEncoder, self).__init__()
        self.vocab = vocab
        self.word_dim = word_dim

    @property
    def weight(self):
        raise NotImplementedError

    def forward(self, x, lens=None):
        """Encodes sequences of one-hot word encodings in a fixed vector space.

        Arguments:
            x (LongTensor): batch_size x seq_len
            lens (optional, LongTensor): batch_size

        Returns:
            (FloatTensor): batch_size x seq_len x word_dim
        """
        raise NotImplementedError


class EmbeddingWordEncoder(AbstractWordEncoder):
    name = "embedding"

    def __init__(self, *args, **kwargs):
        super(EmbeddingWordEncoder, self).__init__(*args, **kwargs)

        self.embedding = nn.Embedding(
            num_embeddings=len(self.vocab) + 1,  # +1 for padding
            padding_idx=len(self.vocab),
            embedding_dim=self.word_dim
        )

    @property
    def weight(self):
        return self.embedding.weight[:len(self.vocab)]

    def forward(self, x, lens=None):
        if lens is not None:
            x = x.masked_fill(~utils.mask(lens, x.size(-1)), len(self.vocab))
        return self.embedding(x)


class PretrainedWordEncoder(EmbeddingWordEncoder):
    name = None

    def __init__(self, *args, freeze=False, **kwargs):
        super(PretrainedWordEncoder, self).__init__(*args, **kwargs)
        self.freeze = freeze

        for p in self.embedding.parameters():
            p.requires_grad = not self.freeze

    def load_vector(self, w: str) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def reset_parameters(self):
        super(PretrainedWordEncoder, self).reset_parameters()
        self.load_embeddings()

    def load_embeddings(self):
        count = 0
        for w, idx in self.vocab.f2i.items():
            if isinstance(w, str):
                v = self.load_vector(w)
            elif isinstance(w, Sequence) and isinstance(w[0], str):
                v = None
                for e in w:
                    if v is None:
                        v = self.load_vector(e)
                    else:
                        r = self.load_vector(e)
                        if r is None:
                            continue
                        else:
                            v += r
            else:
                raise TypeError(f"unsupported word type: {type(w)}")
            if v is None:
                continue
            # allow word vectors to be clipped or padded
            # if len(v) != self.word_dim:
            #     self.logger.warning(f"loaded word vector ('{w}'') does not "
            #                         f"match `word_dim`: "
            #                         f"{len(v)} != {self.word_dim}")
            dim = min(self.word_dim, len(v))
            self.embedding.weight.detach()[idx, :dim] = v[:dim]
            count += 1
        self.logger.info(f"loaded {count} word vectors")


class FastTextWordEncoder(PretrainedWordEncoder):
    name = "fasttext"

    def __init__(self, *args,
                 fasttext_path: str = "",
                 model_path: str = "", **kwargs):
        super(FastTextWordEncoder, self).__init__(*args, **kwargs)
        self.fasttext_path = fasttext_path
        self.model_path = model_path

        assert (self.fasttext_path is not None and self.fasttext_path and
                self.model_path is not None and self.model_path), \
            f"fasttext_path and model_path must be provided"
        self.fasttext = None

    def reset_parameters(self):
        with embeds.FastText(self.fasttext_path, self.model_path) as ft:
            self.fasttext = embeds.FastTextEmbeddings(ft)
            return super().reset_parameters()

    def load_vector(self, w: str) -> Optional[torch.Tensor]:
        return torch.tensor(self.fasttext[w])


class GloveWordEncoder(PretrainedWordEncoder):
    name = "glove"

    def __init__(self, *args, glove_path: str = "", **kwargs):
        super(GloveWordEncoder, self).__init__(*args, **kwargs)
        self.glove_path = glove_path

        assert self.glove_path, \
            f"must provide a valid glove path: '{self.glove_path}'"
        self.vectors = embeds.GloveFormatEmbeddings(
            path=self.glove_path,
            words=set(self.vocab.f2i)
        )
        self.vectors.preload()

    def load_vector(self, w: str) -> Optional[torch.Tensor]:
        if w not in self.vectors:
            return
        return torch.FloatTensor(self.vectors[w])


class HDF5WordEncoder(PretrainedWordEncoder):
    name = "hdf5"

    def __init__(self, *args, path: str = "", **kwargs):
        super(HDF5WordEncoder, self).__init__(*args, **kwargs)
        self.path = path

        assert self.path, \
            f"must provide a valid glove path: '{self.path}'"
        self.vectors = embeds.HDF5Embeddings(self.path)
        self.vectors.preload()

    def load_vector(self, w: str) -> Optional[torch.Tensor]:
        if w not in self.vectors:
            return
        return torch.FloatTensor(self.vectors[w])


class VZhongWordEncoder(PretrainedWordEncoder):
    name = "vzhong-embeddings"

    def __init__(self, *args,
                 cls_name="GloveEmbedding",
                 cls_kwargs="{}",
                 default="zero", **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_name = cls_name
        self.cls_kwargs = cls_kwargs
        self.default = default

        vzhong_embeddings = importlib.import_module("embeddings")
        self.emb_cls = getattr(vzhong_embeddings, self.cls_name)
        self.emb_obj = self.emb_cls(**eval(self.cls_kwargs))

    def load_vector(self, w: str) -> Optional[torch.Tensor]:
        return torch.tensor(self.emb_obj.emb(w, default=self.default))


class CompositeWordEncoder(AbstractWordEncoder):
    name = "composite"

    def __init__(self, *args,
                 first_dim=100, second_dim=100,
                 first=AbstractWordEncoder,
                 second=AbstractWordEncoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_dim = first_dim
        self.second_dim = second_dim
        self.first_cls = first
        self.second_cls = second

        if self.first_dim + self.second_dim != self.word_dim:
            raise ValueError(
                f"the sum of all dims must be equal to the word "
                f"dim: {self.first_dim} + {self.second_dim} != {self.word_dim}"
            )
        self.first = self.first_cls(self.vocab, self.first_dim)
        self.second = self.second_cls(self.vocab, self.second_dim)

    @property
    def weight(self):
        return torch.cat([self.first.weight, self.second.weight], 1)

    def forward(self, x, lens=None):
        return torch.cat([self.first(x, lens), self.second(x, lens)], -1)
