__all__ = ["HDALoss"]

from dataclasses import dataclass
from typing import Tuple, ClassVar

import torch
import torch.nn as nn

import utils
from datasets import BatchData
from datasets import VocabSet
from .loss import Loss


@dataclass
class HDALoss(Loss):
    vocabs: VocabSet
    _ce: ClassVar[nn.CrossEntropyLoss] = \
        nn.CrossEntropyLoss(reduction="none", ignore_index=-1)
    _bce: ClassVar[nn.BCEWithLogitsLoss] = \
        nn.BCEWithLogitsLoss(reduction="none")

    @property
    def num_asv(self):
        return len(self.vocabs.goal_state.asv)

    def compute(self, batch: BatchData, outputs, step: int = None
                ) -> Tuple[torch.Tensor, utils.TensorMap]:
        logit, post, prior = outputs
        batch_size = batch.batch_size
        max_conv_len = batch.max_conv_len
        max_sent_len = batch.max_sent_len
        w_logit, p_logit, g_logit, s_logit = \
            (logit[k] for k in ("sent", "speaker", "goal", "state"))
        conv_lens, sent_lens = batch.conv_lens, batch.sent.lens1
        conv_mask = utils.mask(conv_lens, max_conv_len)
        sent_lens = sent_lens.masked_fill(~conv_mask, 0)
        sent_mask = utils.mask(sent_lens, max_sent_len)
        goal_logit_mask = (((g_logit != float("-inf")) &
                            (g_logit != float("inf")))
                           .masked_fill(~conv_mask.unsqueeze(-1), 0))
        state_logit_mask = \
            (((s_logit != float("-inf")) & (s_logit != float("inf")))
             .masked_fill(~conv_mask.unsqueeze(-1), 0))
        w_target = (batch.sent.value
                    .masked_fill(~sent_mask, -1)
                    .view(-1, max_sent_len))[..., 1:]
        g_target = utils.to_dense(
            idx=batch.goal.value,
            lens=batch.goal.lens1,
            max_size=self.num_asv
        )
        s_target = utils.to_dense(
            idx=batch.state.value,
            lens=batch.state.lens1,
            max_size=self.num_asv
        )
        p_target = batch.speaker.value.masked_fill(~conv_mask, -1)
        goal_loss = (self._bce(g_logit, g_target.float())
                     .masked_fill(~goal_logit_mask, 0)).sum(-1)
        state_loss = (self._bce(s_logit, s_target.float())
                      .masked_fill(~state_logit_mask, 0)).sum(-1)
        spkr_loss = self._ce(
            p_logit.view(-1, self.vocabs.num_speakers),
            p_target.view(-1)
        ).view(batch_size, max_conv_len)
        sent_loss = self._ce(
            w_logit[:, :, :-1].contiguous().view(-1, len(self.vocabs.word)),
            w_target.contiguous().view(-1)
        ).view(batch_size, max_conv_len, -1).sum(-1)
        loss_recon = (sent_loss.sum(-1) + state_loss.sum(-1) +
                      goal_loss.sum(-1) + spkr_loss.sum(-1))
        loss = nll = loss_recon
        stats = {
            "nll": nll.mean(),
            "loss": loss.mean(),
            "loss-recon": loss_recon.mean(),
            "loss-sent": sent_loss.sum(-1).mean(),
            "loss-sent-turn": sent_loss.sum() / conv_lens.sum(),
            "loss-sent-word": sent_loss.sum() / sent_lens.sum(),
            "ppl-turn": (sent_loss.sum() / conv_lens.sum()).exp(),
            "ppl-word": (sent_loss.sum() / sent_lens.sum()).exp(),
            "loss-goal": goal_loss.sum(-1).mean(),
            "loss-goal-turn": goal_loss.sum() / conv_lens.sum(),
            "loss-goal-asv": goal_loss.sum() / goal_logit_mask.sum(),
            "loss-state": state_loss.sum(-1).mean(),
            "loss-state-turn": state_loss.sum() / conv_lens.sum(),
            "loss-state-asv": state_loss.sum() / state_logit_mask.sum(),
            "loss-spkr": spkr_loss.sum(-1).mean(),
            "loss-spkr-turn": spkr_loss.sum() / conv_lens.sum()
        }
        for spkr_idx, spkr in self.vocabs.speaker.i2f.items():
            if spkr == "<unk>":
                continue
            spkr_mask = p_target == spkr_idx
            spkr_sent_lens = sent_lens.masked_fill(~spkr_mask, 0)
            spkr_goal_mask = \
                goal_logit_mask.masked_fill(~spkr_mask.unsqueeze(-1), 0)
            spkr_state_mask = \
                state_logit_mask.masked_fill(~spkr_mask.unsqueeze(-1), 0)
            spkr_sent_loss = sent_loss.masked_fill(~spkr_mask, 0).sum()
            spkr_goal_loss = goal_loss.masked_fill(~spkr_mask, 0).sum()
            spkr_state_loss = state_loss.masked_fill(~spkr_mask, 0).sum()
            spkr_spkr_loss = spkr_loss.masked_fill(~spkr_mask, 0).sum()
            spkr_stats = {
                "loss-sent": spkr_sent_loss / batch_size,
                "loss-sent-turn": spkr_sent_loss / spkr_mask.sum(),
                "loss-sent-word": spkr_sent_loss / spkr_sent_lens.sum(),
                "ppl-turn": (spkr_sent_loss / spkr_mask.sum()).exp(),
                "ppl-word": (spkr_sent_loss / spkr_sent_lens.sum()).exp(),
                "loss-goal": spkr_goal_loss / batch_size,
                "loss-goal-turn": spkr_goal_loss / spkr_mask.sum(),
                "loss-goal-asv": spkr_goal_loss / spkr_goal_mask.sum(),
                "loss-state": spkr_state_loss / batch_size,
                "loss-state-turn": spkr_state_loss / spkr_mask.sum(),
                "loss-state-asv": spkr_state_loss / spkr_state_mask.sum(),
                "loss-spkr": spkr_spkr_loss / batch_size,
                "loss-spkr-turn": spkr_spkr_loss / spkr_mask.sum()
            }
            stats.update({f"{k}-{spkr}": v for k, v in spkr_stats.items()})
        return loss.mean(), stats
