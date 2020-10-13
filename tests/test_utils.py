import torch

import utils


def test_dense_sparse():
    x = torch.randint(0, 2, (3, 4, 5)).byte()
    y = utils.to_dense(*utils.to_sparse(x))
    assert (x == y).all()


def test_roll():
    x = torch.randn(3, 4, 5)
    y = utils.shift(utils.shift(x, 2, 1, True), 2, 1, True)
    assert (x == y).all()


if __name__ == "__main__":
    test_roll()
