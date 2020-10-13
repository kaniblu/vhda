__all__ = ["VHUSLoss"]

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
class VHUSLoss(Loss):
    vocabs: VocabSet
    enable_kl: bool = True
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
        s_logit, zstate_post, zstate_prior = \
            logit["state"], post["state"], prior["state"]
        conv_lens, sent_lens = batch.conv_lens, batch.sent.lens1
        conv_mask = utils.mask(conv_lens, max_conv_len)
        state_logit_mask = \
            (((s_logit != float("-inf")) & (s_logit != float("inf")))
             .masked_fill(~conv_mask.unsqueeze(-1), 0))
        kld_state = zstate_post.kl_div(zstate_prior).masked_fill(~conv_mask, 0)
        s_target = utils.to_dense(
            idx=batch.state.value,
            lens=batch.state.lens1,
            max_size=self.num_asv
        )
        p_target = batch.speaker.value.masked_fill(~conv_mask, -1)
        state_loss = (self._bce(s_logit, s_target.float())
                      .masked_fill(~state_logit_mask, 0)).sum(-1)
        kld_weight = self.kld_weight.get(step)
        nll = state_loss + kld_state
        loss = state_loss + kld_weight * kld_state
        state_mi = \
            (estimate_mi(zstate_post.view(batch_size * max_conv_len, -1))
             .view(batch_size, max_conv_len).masked_fill(~conv_mask, 0).sum(-1))
        stats = {
            "nll": nll.mean(),
            "state-mi": state_mi.mean(),
            "loss-state": state_loss.sum(-1).mean(),
            "loss-state-turn": state_loss.sum() / conv_lens.sum(),
            "loss-state-asv": state_loss.sum() / state_logit_mask.sum(),
            "kld-weight": torch.tensor(kld_weight),
            "kld-state": kld_state.sum(-1).mean(),
            "kld-state-turn": kld_state.sum() / conv_lens.sum(),
            "kld": kld_state.sum(-1).mean()
        }
        for spkr_idx, spkr in self.vocabs.speaker.i2f.items():
            if spkr == "<unk>":
                continue
            spkr_mask = p_target == spkr_idx
            spkr_state_mask = \
                state_logit_mask.masked_fill(~spkr_mask.unsqueeze(-1), 0)
            spkr_state_loss = state_loss.masked_fill(~spkr_mask, 0).sum()
            spkr_kld_state = kld_state.masked_fill(~spkr_mask, 0).sum()
            spkr_stats = {
                "loss-state": spkr_state_loss / batch_size,
                "loss-state-turn": spkr_state_loss / spkr_mask.sum(),
                "loss-state-asv": spkr_state_loss / spkr_state_mask.sum(),
                "kld-state": spkr_kld_state / batch_size,
                "kld-state-turn": spkr_kld_state / spkr_mask.sum(),
            }
            stats.update({f"{k}-{spkr}": v for k, v in spkr_stats.items()})
        return loss.mean(), stats
