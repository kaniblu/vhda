__all__ = ["PosteriorEvaluator", "estimate_mi"]

import math
from dataclasses import dataclass
from typing import Union, Sequence, Optional

import torch

import utils
from evaluators.evaluator import FinegrainedEvaluator
from datasets import Dialog
from datasets import BatchData
from utils import TensorMap
from models import MultiGaussian


def estimate_agg_posterior(z: MultiGaussian, z_samples=None
                           ) -> torch.Tensor:
    batch_size, zdim = z.mu.size()
    if z_samples is None:
        z_samples = z.sample()
    log_qzx = MultiGaussian(
        mu=(z.mu.unsqueeze(0).expand(batch_size, batch_size, -1)
            .reshape(-1, zdim)),
        var_logit=(z.var_logit.unsqueeze(0).expand(batch_size, batch_size, -1)
                   .reshape(-1, zdim))
    ).log_prob(z_samples.unsqueeze(1).expand(batch_size, batch_size, -1)
               .reshape(-1, zdim))
    return (utils.log_sum_exp(log_qzx.reshape(batch_size, batch_size), 1)
            - math.log(batch_size))


def estimate_prior(z: MultiGaussian, z_samples=None
                   , prior: MultiGaussian = None) -> torch.Tensor:
    batch_size, zdim = z.mu.size()
    if z_samples is None:
        z_samples = z.sample()
    if prior is None:
        prior = MultiGaussian(batch_size, zdim).to(z.mu.device)
    log_prior = prior.log_prob(z_samples)
    return log_prior


def estimate_kld_aggpost_prior(z: MultiGaussian, z_samples=None):
    if z_samples is None:
        z_samples = z.sample()
    return (estimate_agg_posterior(z, z_samples) -
            estimate_prior(z, z_samples))


def estimate_mi(z: MultiGaussian, z_samples=None, lens=None):
    zdim = z.mu.size(-1)
    size = z.mu.size()
    z = z.view(-1, zdim)
    if z_samples is None:
        z_samples = z.sample()
    ret = z.log_prob(z_samples) - estimate_agg_posterior(z, z_samples)
    if len(size) == 3:
        return (ret.view(*size[:-1])
                .masked_fill(~utils.mask(lens, size[1]), 0).sum(-1))
    else:
        return ret.view(*size[:-1])


@dataclass
class PosteriorEvaluator(FinegrainedEvaluator):

    @staticmethod
    def posterior_statistics(post: dict, prior: dict) -> TensorMap:
        stats = dict()
        for m in ("conv", "goal", "state", "sent"):
            m_post: MultiGaussian = post[m]
            m_prior: MultiGaussian = prior[m]
            stats[f"{m}-post-mu-mu"] = m_post.mu.mean()
            stats[f"{m}-post-mu-var"] = m_post.mu.var()
            stats[f"{m}-post-var-mu"] = m_post.var.mean()
            stats[f"{m}-post-var-var"] = m_post.var.var()
            stats[f"{m}-post-mvr"] = m_post.mu.var() / m_post.var.mean()
            stats[f"{m}-prior-mu-mu"] = m_prior.mu.mean()
            stats[f"{m}-prior-mu-var"] = m_prior.mu.var()
            stats[f"{m}-prior-var-mu"] = m_prior.var.mean()
            stats[f"{m}-prior-var-var"] = m_prior.var.var()
            # stats[f"{m}-prior-mvr"] = m_prior.mu.var() / m_prior.var.mean()
            # m_post, m_prior = MultiGaussian(
            #     mu=m_post.mu[:, 0],
            #     logvar=m_post.logvar[:, 0]
            # ), MultiGaussian(
            #     mu=m_prior.mu[:, 0],
            #     logvar=m_prior.logvar[:, 0]
            # )
            # stats[f"{m}-0-post-mu-mu"] = m_post.mu.mean()
            # stats[f"{m}-0-post-mu-var"] = m_post.mu.var()
            # stats[f"{m}-0-post-var-mu"] = m_post.var.mean()
            # stats[f"{m}-0-post-var-var"] = m_post.var.var()
            # stats[f"{m}-0-post-mvr"] = m_post.mu.var() / m_post.var.mean()
            # stats[f"{m}-0-prior-mu-mu"] = m_prior.mu.mean()
            # stats[f"{m}-0-prior-mu-var"] = m_prior.mu.var()
            # stats[f"{m}-0-prior-var-mu"] = m_prior.var.mean()
            # stats[f"{m}-0-prior-var-var"] = m_prior.var.var()
            # stats[f"{m}-0-prior-mvr"] = m_prior.mu.var() / m_prior.var.mean()
        return stats

    def reset(self):
        pass

    def update(self, batch: Union[Sequence[Dialog], BatchData],
               pred: Union[Sequence[Dialog], BatchData], outputs
               ) -> Optional[TensorMap]:
        logit, post, prior = outputs
        stats = dict()
        stats.update(self.posterior_statistics(post, prior))
        stats["conv-mi"] = estimate_mi(post["conv"]).mean()
        return stats

    def get(self) -> TensorMap:
        return {}
