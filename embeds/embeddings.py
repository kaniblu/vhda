__all__ = ["Embeddings"]

import numpy as np


class Embeddings:
    """
    An abstract embedding object.

    All embedding objects must inherit this interface. `name` property
    must be defined first.

    Embedding objects are designed to be hashable, such that the same type of
    embeddings can be loaded only once.
    """

    @property
    def dim(self) -> int:
        """Returns embedding dimensions."""
        raise NotImplementedError()

    def preload(self):
        """Load actual data here if necessary. Must return self."""
        raise NotImplementedError()

    def __getitem__(self, item) -> np.ndarray:
        """Returns a numpy array of the embedding."""
        raise NotImplementedError()

    def __contains__(self, item):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()
