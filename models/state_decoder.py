__all__ = ["AbstractStateDecoder", "GenericStateDecoder"]

import torch
import torchmodels
import torch.nn as nn
from typing import Mapping

import utils
from datasets import VocabSet
from datasets import ActSlotValue
from datasets import StateVocabSet


class AbstractStateDecoder(utils.Module):

    def __init__(self, vocabs: VocabSet, state_type, input_dim, asv_dim):
        super(AbstractStateDecoder, self).__init__()
        self.vocabs = vocabs
        self.state_type = state_type
        self.input_dim = input_dim
        self.asv_dim = asv_dim

        if self.state_type not in {"goal", "state", "goal_state"}:
            raise ValueError(f"unknown state type: {self.state_type}")

    def forward(self, x, spkr, asv):
        """Decodes state labels from a hidden vector.

        Arguments:
            x ([N x input_dim] FloatTensor): hidden vectors
            spkr ([N] LongTensor): speaker annotation
            asv ([N x num_asv x asv_dim] FloatTensor):

        Returns:
            logits ([N x num_asv] FloatTensor)
        """
        raise NotImplementedError


class GenericStateDecoder(AbstractStateDecoder):
    name = "generic"

    def __init__(self, *args, hidden_dim=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim

        self.input_layer = torchmodels.Sequential(
            torchmodels.Linear(
                in_features=self.input_dim,
                out_features=self.asv_dim
            ),
        )
        # speaker mask is a [num_speakers x num asv] BoolTensor
        # that indicates active asvs for each speaker
        self.speaker_asv_mask = nn.Parameter(self._create_speaker_asv_mask(),
                                             requires_grad=False)

    def _create_speaker_asv_mask(self) -> torch.Tensor:
        masks = dict()
        global_vocabs: StateVocabSet = self.vocabs.goal_state
        spkr_vocabs: Mapping[str, StateVocabSet] = \
            getattr(self.vocabs, f"speaker_{self.state_type}")
        for spkr_idx, spkr in self.vocabs.speaker.i2f.items():
            if spkr == "<unk>":
                mask = torch.zeros(len(global_vocabs.asv)).bool()
                pad_asv = ActSlotValue("<pad>", "<pad>", "<pad>")
                pad_idx = global_vocabs.asv[pad_asv]
                mask[pad_idx] = True
            else:
                spkr_state = spkr_vocabs[spkr]
                mask = torch.tensor([global_vocabs.asv[idx] in spkr_state.asv
                                     for idx in range(len(global_vocabs.asv))])
            masks[spkr_idx] = mask
        masks = torch.stack([mask for _, mask in
                             sorted(masks.items(), key=lambda x: x[0])])
        return masks

    def forward(self, x, spkr, asv):
        x_size = x.size()
        h = self.input_layer(x).view(-1, self.asv_dim)
        logit = torch.mm(h, asv.permute(1, 0)).view(*x_size[:-1], asv.size(0))
        mask = self.speaker_asv_mask[spkr]
        return logit.masked_fill(~mask, float("-inf"))
