__all__ = ["AbstractLabelEncoder", "EmbeddingLabelEncoder",
           "PretrainedLabelEncoder", "FasttextLabelEncoder",
           "GloveLabelEncoder"]

from typing import Optional

import torch
import torch.nn as nn

import utils
import embeddings


class AbstractLabelEncoder(utils.Module):

    def __init__(self, vocab: utils.Vocabulary, label_dim):
        super(AbstractLabelEncoder, self).__init__()
        self.vocab = vocab
        self.label_dim = label_dim

    def forward(self, x):
        """Encodes one-hot encodings into a continuous vector space.

        Arguments:
            x ([...] LongTensor)

        Returns:
            ([... x label_dim] FloatTensor)
        """
        raise NotImplementedError

    def weight(self):
        raise NotImplementedError


class EmbeddingLabelEncoder(AbstractLabelEncoder):
    name = "embedding"

    def __init__(self, *args, allow_padding=True, **kwargs):
        super(EmbeddingLabelEncoder, self).__init__(*args, **kwargs)
        self.allow_padding = allow_padding

        self.embedding = nn.Embedding(
            num_embeddings=len(self.vocab) + (1 if self.allow_padding else 0),
            padding_idx=len(self.vocab) if self.allow_padding else None,
            embedding_dim=self.label_dim
        )

    def forward(self, x):
        return self.embedding(x)

    def weight(self):
        return self.embedding.weight[:len(self.vocab)]


class PretrainedLabelEncoder(EmbeddingLabelEncoder):
    name = None

    def __init__(self, *args, freeze=False, **kwargs):
        super(PretrainedLabelEncoder, self).__init__(*args, **kwargs)
        self.freeze = freeze

        for p in self.embedding.parameters():
            p.requires_grad = not self.freeze

    def load_vector(self, w: str) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def reset_parameters(self):
        super(PretrainedLabelEncoder, self).reset_parameters()
        self.load_embeddings()

    def load_embeddings(self):
        count = 0
        for w, idx in self.vocab.f2i.items():
            v = self.load_vector(w)
            if v is None:
                continue
            if len(v) != self.label_dim:
                self.logger.warning(f"loaded label vector ('{w}'') does not "
                                    f"match `label_dim`: "
                                    f"{len(v)} != {self.label_dim}")
            dim = min(self.label_dim, len(v))
            self.embedding.weight.detach()[idx, :dim] = v[:dim]
            count += 1
        self.logger.info(f"loaded {count} label vectors")


class FasttextLabelEncoder(PretrainedLabelEncoder):
    name = "fasttext"

    def __init__(self, *args,
                 fasttext_path: str = "",
                 model_path: str = "", **kwargs):
        super(FasttextLabelEncoder, self).__init__(*args, **kwargs)
        self.fasttext_path = fasttext_path
        self.model_path = model_path

        assert (self.fasttext_path is not None and self.fasttext_path and
                self.model_path is not None and self.model_path), \
            f"fasttext_path and model_path must be provided"
        self.fasttext = embeddings.FastTextEmbeddings(embeddings.FastText(
            ft_path=self.fasttext_path,
            model_path=self.model_path
        ))
        self.fasttext.preload()

    def load_vector(self, w: str) -> Optional[torch.Tensor]:
        return torch.tensor(self.fasttext[w])


class GloveLabelEncoder(PretrainedLabelEncoder):
    name = "glove"

    def __init__(self, *args, glove_path: str = "", **kwargs):
        super(GloveLabelEncoder, self).__init__(*args, **kwargs)
        self.glove_path = glove_path

        assert self.glove_path, \
            f"must provide a valid glove path: '{self.glove_path}'"
        self.vectors = embeddings.GloveFormatEmbeddings(
            path=self.glove_path,
            words=set(self.vocab.f2i)
        )
        self.vectors.preload()

    def load_vector(self, w: str) -> Optional[torch.Tensor]:
        if w not in self.vectors:
            return
        return torch.FloatTensor(self.vectors[w])


class HDF5LabelEncoder(PretrainedLabelEncoder):
    name = "hdf5"

    def __init__(self, *args, path: str = "", **kwargs):
        super(HDF5LabelEncoder, self).__init__(*args, **kwargs)
        self.path = path

        assert self.path, \
            f"must provide a valid glove path: '{self.path}'"
        self.vectors = embeddings.HDF5Embeddings(self.path)
        self.vectors.preload()

    def load_vector(self, w: str) -> Optional[torch.Tensor]:
        if w not in self.vectors:
            return
        return torch.FloatTensor(self.vectors[w])
