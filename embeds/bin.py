__all__ = ["BinaryEmbeddings"]

from dataclasses import dataclass, field

import numpy as np
from gensim.models import KeyedVectors

from .embeddings import Embeddings


@dataclass
class BinaryEmbeddings(Embeddings):
    path: str
    _w2v: KeyedVectors = field(init=False, default=None)

    def __post_init__(self):
        pass

    @property
    def dim(self) -> int:
        return self._w2v.vector_size

    def preload(self):
        self._w2v = KeyedVectors.load_word2vec_format(self.path, binary=True)
        return self

    def __getitem__(self, item) -> np.ndarray:
        return self._w2v[item]

    def __contains__(self, item):
        return item in self._w2v

    def __iter__(self):
        for w in self._w2v.vocab:
            yield w, self._w2v[w]
