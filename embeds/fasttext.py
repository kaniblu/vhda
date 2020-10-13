__all__ = ["FastText", "FastTextEmbeddings"]

import subprocess
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from utils import UnsupportedOperationsError
from .embeddings import Embeddings


@dataclass
class FastText:
    ft_path: str
    model_path: str
    dtype: type = np.float32
    process: Optional[subprocess.Popen] = field(
        init=False, hash=False, compare=False, repr=False,
        default=None
    )

    def query(self, word):
        self.process.stdin.write(f"{word}\n".encode())
        self.process.stdin.flush()
        line = self.process.stdout.readline().decode()
        line = " ".join(line.split()[1:])
        return np.fromstring(line, dtype=self.dtype, sep=" ")

    def __enter__(self):
        self.process = subprocess.Popen(
            args=[self.ft_path, "print-word-vectors", self.model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process is not None:
            self.process.kill()
            self.process = None


@dataclass
class FastTextEmbeddings(Embeddings):
    ft: FastText
    _dim: int = field(
        init=False, compare=False, hash=False, repr=False,
        default=None
    )

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.ft.query("."))
        return self._dim

    def preload(self):
        self.ft.__enter__()
        return self

    def __getitem__(self, item) -> np.ndarray:
        return self.ft.query(item)

    def __contains__(self, item):
        return True

    def __iter__(self):
        raise UnsupportedOperationsError
