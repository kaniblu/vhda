__all__ = ["MultiGaussian", "MultiGaussianLayer"]

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.init as ti
import torch.distributions as dist

import utils


@dataclass
class MultiGaussian:
    mu: torch.Tensor
    var_logit: torch.Tensor
    _unit_var_logit: ClassVar[torch.Tensor] = \
        torch.tensor(1.0).exp().add_(-1).log()

    def __post_init__(self):
        assert self.mu.size() == self.var_logit.size()

    @classmethod
    def unit(cls, *size):
        return MultiGaussian(torch.zeros(*size),
                             torch.zeros(*size).fill_(cls._unit_var_logit))

    @property
    def var(self):
        return nn.functional.softplus(self.var_logit)

    @property
    def logvar(self):
        return (self.var + utils.EPS).log()

    @property
    def std(self):
        # PyTorch BUG?: NaNs can occur *occasionally* in the gradients of `sqrt`
        return (self.var + utils.EPS).sqrt()

    @property
    def batch_size(self):
        return self.mu.size(0)

    @property
    def dim(self):
        return self.mu.size(1)

    def view_(self, *size):
        self.mu = self.mu.view(*size)
        self.var_logit = self.var_logit.view(*size)
        return self

    def view(self, *size):
        return MultiGaussian(self.mu, self.var_logit).view_(*size)

    def to(self, device):
        return MultiGaussian(
            mu=self.mu.to(device),
            var_logit=self.var_logit.to(device)
        )

    def kl_div(self, other=None, reduction="sum"):
        """Computes kl div with another MultiGaussian object. If no argument
        is provided, a standard gaussian distribution is assumed.

            KLD(self || other)

        Arguments:
            other (optional, MultiGaussian): another MultiGaussian
            reduction (str): valid reductions are 'mean', 'sum', and 'none'.

        Returns:
            kld (FloatTensor): [N]
        """
        if other is None:
            other = self.unit(*self.size()).to(self.mu.device)
        elif not isinstance(other, MultiGaussian):
            raise TypeError(f"unable to compute kl div with differently "
                            f"typed objects: {type(other)}")
        kld = (0.5 * (other.logvar - self.logvar) +
               (self.var + (self.mu - other.mu).pow(2)) /
               (2 * (other.var + utils.EPS)) - 0.5)
        if reduction == "sum":
            return kld.sum(-1)
        elif reduction == "mean":
            return kld.mean(-1)
        elif reduction == "none":
            return kld
        else:
            raise ValueError(f"unsupported reduction method: {reduction}")

    def sample(self, scale=1.0):
        return (self.mu +
                scale * self.mu.new(self.mu.size()).normal_() * self.std)

    def log_prob(self, z):
        return dist.Normal(self.mu, self.std).log_prob(z).sum(-1)

    def size(self, *args, **kwargs):
        return self.mu.size(*args, **kwargs)


class MultiGaussianLayer(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(MultiGaussianLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.mu_linear = nn.Linear(
            in_features=self.input_dim,
            out_features=self.hidden_dim
        )
        self.logvar_linear = nn.Linear(
            in_features=self.input_dim,
            out_features=self.hidden_dim
        )

    def reset_parameters(self):
        ti.xavier_normal_(self.mu_linear.weight)
        ti.xavier_normal_(self.logvar_linear.weight)

    def forward(self, x):
        return MultiGaussian(self.mu_linear(x), self.logvar_linear(x))
