__all__ = ["GloveFormatEmbeddings"]

from dataclasses import dataclass, field
from typing import Set, MutableMapping

import tqdm
import numpy as np

from .embeddings import Embeddings


@dataclass
class GloveFormatEmbeddings(Embeddings):
    path: str
    words: Set[str] = None
    progress: bool = True
    _dim: int = field(
        init=False, repr=False, hash=False, compare=False,
        default=None
    )
    vectors: MutableMapping[str, np.ndarray] = field(
        init=False, repr=False, hash=False, compare=False,
        default_factory=dict
    )

    def preload(self):
        with open(self.path, "r") as f:
            num_words, dim = f.readline().split()
            num_words, self._dim = int(num_words), int(dim)
            for line in tqdm.tqdm(
                    f,
                    total=num_words,
                    desc="loading vectors",
                    dynamic_ncols=True,
                    disable=not self.progress
            ):
                tokens = line.split()
                word = " ".join(tokens[:-self.dim])
                if self.words is not None and word not in self.words:
                    continue
                vec = np.array([float(v) for v in tokens[-self.dim:]])
                self.vectors[word] = vec
        return self

    @property
    def dim(self) -> int:
        return self._dim

    def __contains__(self, item):
        return item in self.vectors

    def __getitem__(self, item) -> np.ndarray:
        return self.vectors[item]

    def __iter__(self):
        return iter(self.vectors.items())
