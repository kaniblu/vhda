__all__ = ["VHDAWithoutGoalActLoss"]

from dataclasses import dataclass
from typing import Tuple, ClassVar

import torch
import torch.nn as nn

import utils
from datasets import BatchData
from datasets import VocabSet
from evaluators import estimate_mi
from .loss import Loss


@dataclass
class VHDAWithoutGoalActLoss(Loss):
    vocabs: VocabSet
    loss_sent_weight: float = 1.0
    loss_spkr_weight: float = 1.0
    kld_sent_weight: float = 1.0
    kld_conv_weight: float = 1.0
    kld_spkr_weight: float = 1.0
    enable_kl: bool = True
    calibrate_mi: bool = True
    kld_weight: utils.Scheduler = utils.ConstantScheduler(1.0)
    _ce: ClassVar[nn.CrossEntropyLoss] = \
        nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
    _bce: ClassVar[nn.BCEWithLogitsLoss] = \
        nn.BCEWithLogitsLoss(reduction="none")

    @property
    def num_asv(self):
        return len(self.vocabs.goal_state.asv)

    def compute(self, batch: BatchData, outputs, step: int = None
                ) -> Tuple[torch.Tensor, utils.TensorMap]:
        step = step or 0
        logit, post, prior = outputs
        batch_size = batch.batch_size
        max_conv_len = batch.max_conv_len
        max_sent_len = batch.max_sent_len
        w_logit, p_logit = \
            (logit[k] for k in ("sent", "speaker"))
        zconv_post, zsent_post, zspkr_post = \
            (post[k] for k in ("conv", "sent", "speaker"))
        zconv_prior, zsent_prior, zspkr_prior = \
            (prior[k] for k in ("conv", "sent", "speaker"))
        conv_lens, sent_lens = batch.conv_lens, batch.sent.lens1
        conv_mask = utils.mask(conv_lens, max_conv_len)
        sent_lens = sent_lens.masked_fill(~conv_mask, 0)
        sent_mask = utils.mask(sent_lens, max_sent_len)
        kld_conv = zconv_post.kl_div()
        kld_sent = zsent_post.kl_div(zsent_prior).masked_fill(~conv_mask, 0)
        kld_spkr = zspkr_post.kl_div(zspkr_prior).masked_fill(~conv_mask, 0)
        w_target = (batch.sent.value
                    .masked_fill(~sent_mask, -1)
                    .view(-1, max_sent_len))[..., 1:]
        p_target = batch.speaker.value.masked_fill(~conv_mask, -1)
        spkr_loss = self._ce(
            p_logit.view(-1, self.vocabs.num_speakers),
            p_target.view(-1)
        ).view(batch_size, max_conv_len)
        sent_loss = self._ce(
            w_logit[:, :, :-1].contiguous().view(-1, len(self.vocabs.word)),
            w_target.contiguous().view(-1)
        ).view(batch_size, max_conv_len, -1).sum(-1)
        kld_weight = self.kld_weight.get(step)
        loss_kld = (kld_conv * self.kld_conv_weight +
                    kld_sent.sum(-1) * self.kld_sent_weight +
                    kld_spkr.sum(-1) * self.kld_spkr_weight)
        loss_recon = (self.loss_sent_weight * sent_loss.sum(-1) +
                      self.loss_spkr_weight * spkr_loss.sum(-1))
        nll = loss_recon + loss_kld
        conv_mi = estimate_mi(zconv_post)
        if self.enable_kl:
            if self.calibrate_mi:
                loss = loss_recon + kld_weight * (loss_kld - conv_mi)
            else:
                loss = loss_recon + kld_weight * loss_kld
        else:
            loss = loss_recon
        stats = {
            "nll": nll.mean(),
            "conv-mi": conv_mi.mean(),
            "loss": loss.mean(),
            "loss-recon": loss_recon.mean(),
            "loss-sent": sent_loss.sum(-1).mean(),
            "loss-sent-turn": sent_loss.sum() / conv_lens.sum(),
            "loss-sent-word": sent_loss.sum() / sent_lens.sum(),
            "ppl-turn": (sent_loss.sum() / conv_lens.sum()).exp(),
            "ppl-word": (sent_loss.sum() / sent_lens.sum()).exp(),
            "loss-spkr": spkr_loss.sum(-1).mean(),
            "loss-spkr-turn": spkr_loss.sum() / conv_lens.sum(),
            "kld-weight": torch.tensor(kld_weight),
            "kld-sent": kld_sent.sum(-1).mean(),
            "kld-sent-turn": kld_sent.sum() / conv_lens.sum(),
            "kld-conv": kld_conv.sum(-1).mean(),
            "kld-spkr": kld_spkr.sum(-1).mean(),
            "kld-spkr-turn": kld_spkr.sum() / conv_lens.sum(),
            "kld": loss_kld.mean()
        }
        for spkr_idx, spkr in self.vocabs.speaker.i2f.items():
            if spkr == "<unk>":
                continue
            spkr_mask = p_target == spkr_idx
            spkr_sent_lens = sent_lens.masked_fill(~spkr_mask, 0)
            spkr_sent_loss = sent_loss.masked_fill(~spkr_mask, 0).sum()
            spkr_spkr_loss = spkr_loss.masked_fill(~spkr_mask, 0).sum()
            spkr_kld_sent = kld_sent.masked_fill(~spkr_mask, 0).sum()
            spkr_kld_spkr = kld_spkr.masked_fill(~spkr_mask, 0).sum()
            spkr_stats = {
                "loss-sent": spkr_sent_loss / batch_size,
                "loss-sent-turn": spkr_sent_loss / spkr_mask.sum(),
                "loss-sent-word": spkr_sent_loss / spkr_sent_lens.sum(),
                "ppl-turn": (spkr_sent_loss / spkr_mask.sum()).exp(),
                "ppl-word": (spkr_sent_loss / spkr_sent_lens.sum()).exp(),
                "loss-spkr": spkr_spkr_loss / batch_size,
                "loss-spkr-turn": spkr_spkr_loss / spkr_mask.sum(),
                "kld-sent": spkr_kld_sent / batch_size,
                "kld-sent-turn": spkr_kld_sent / spkr_mask.sum(),
                "kld-spkr": spkr_kld_spkr / batch_size,
                "kld-spkr-turn": spkr_kld_spkr / spkr_mask.sum(),
            }
            stats.update({f"{k}-{spkr}": v for k, v in spkr_stats.items()})
        return loss.mean(), stats
