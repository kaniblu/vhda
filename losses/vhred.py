__all__ = ["VHREDLoss"]

from dataclasses import dataclass
from typing import Tuple, ClassVar

import torch
import torch.nn as nn

import utils
from datasets import BatchData
from datasets import VocabSet
from .loss import Loss


@dataclass
class VHREDLoss(Loss):
    vocabs: VocabSet
    enable_kl: bool = True
    kld_weight: utils.Scheduler = utils.ConstantScheduler(1.0)
    _ce: ClassVar[nn.CrossEntropyLoss] = \
        nn.CrossEntropyLoss(reduction="none", ignore_index=-1)

    def compute(self, batch: BatchData, outputs, step: int = None
                ) -> Tuple[torch.Tensor, utils.TensorMap]:
        step = step or 0
        logit, post, prior = outputs
        batch_size = batch.batch_size
        max_conv_len = batch.max_conv_len
        max_sent_len = batch.max_sent_len
        w_logit, zsent_post, zsent_prior = \
            logit["sent"], post["sent"], prior["sent"]
        conv_lens, sent_lens = batch.conv_lens, batch.sent.lens1
        conv_mask = utils.mask(conv_lens, max_conv_len)
        sent_lens = sent_lens.masked_fill(~conv_mask, 0)
        sent_mask = utils.mask(sent_lens, max_sent_len)
        kld_sent = zsent_post.kl_div(zsent_prior).masked_fill(~conv_mask, 0)
        w_target = (batch.sent.value
                    .masked_fill(~sent_mask, -1)
                    .view(-1, max_sent_len))[..., 1:]
        sent_loss = self._ce(
            w_logit[:, :, :-1].contiguous().view(-1, len(self.vocabs.word)),
            w_target.contiguous().view(-1)
        ).view(batch_size, max_conv_len, -1).sum(-1)
        kld_weight = self.kld_weight.get(step)
        loss_kld = kld_sent.sum(-1)
        loss_recon = sent_loss.sum(-1)
        nll = loss_recon + loss_kld
        if self.enable_kl:
            loss = loss_recon + kld_weight * loss_kld
        else:
            loss = loss_recon
        stats = {
            "nll": nll.mean(),
            "loss": loss.mean(),
            "loss-recon": loss_recon.mean(),
            "loss-sent": sent_loss.sum(-1).mean(),
            "loss-sent-turn": sent_loss.sum() / conv_lens.sum(),
            "loss-sent-word": sent_loss.sum() / sent_lens.sum(),
            "ppl-turn": (sent_loss.sum() / conv_lens.sum()).exp(),
            "ppl-word": (sent_loss.sum() / sent_lens.sum()).exp(),
            "kld-weight": torch.tensor(kld_weight),
            "kld-sent": kld_sent.sum(-1).mean(),
            "kld-sent-turn": kld_sent.sum() / conv_lens.sum(),
            "kld": loss_kld.mean()
        }
        return loss.mean(), stats
