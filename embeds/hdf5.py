__all__ = ["HDF5Embeddings"]

from dataclasses import dataclass, field
import base64
import json

import h5py
import numpy as np

from .embeddings import Embeddings


@dataclass
class HDF5Embeddings(Embeddings):
    path: str
    _file: h5py.File = field(init=False, repr=False, default=None)
    _vocab: dict = field(init=False, repr=False, default=None)
    _array: np.ndarray = field(init=False, repr=False, default=None)

    def preload(self):
        self._file = h5py.File(self.path, "r")
        data = base64.decodebytes(self._file["vocab"][()])
        self._vocab = json.loads(data.decode("utf-8"))
        self._array = self._file["array"]
        return self

    @property
    def dim(self) -> int:
        return self._array.shape[1]

    def __getitem__(self, item):
        return self._array[self._vocab[item]]

    def __contains__(self, item):
        return item in self._vocab

    def __iter__(self):
        for w in self._vocab:
            yield w

    def items(self):
        for w, i in self._vocab.items():
            yield w, self._array[i]

    def __len__(self):
        return len(self._vocab)
