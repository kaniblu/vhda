__all__ = ["AbstractTDA", "VHCR", "VHDA", "HDA", "VHDAWithoutGoal",
           "VHDAWithoutGoalAct", "VHRED", "VHUS"]

import math

import torch
import torch.nn as nn
import torch.distributions as dist
import torchmodels

import utils
from utils import Stacked1DTensor
from utils import DoublyStacked1DTensor
from datasets import VocabSet
from datasets import BatchData
from datasets import ActSlotValue
from .state_decoder import AbstractStateDecoder
from .gaussian import MultiGaussian
from .gaussian import MultiGaussianLayer
from .seq_encoder import AbstractSequenceEncoder
from .sent_decoder import AbstractSentDecoder
from .context import AbstractContextEncoder
from .word import AbstractWordEncoder


class AbstractTDA(torchmodels.Module):
    """Abstract class for task-oriented dialog autoencoder

    Initialization Arguments:
        vocab (Vocabulary): word vocabulary object
        word_dim (int): word dimensions
        num_labels (list of int): a list of labels and their class sizes
        mode (str): forward behavior (default: inference)
            inference: teacher force all utterances in the decoding step and
                return relevant output logits.
    """

    def __init__(self, vocabs: VocabSet):
        super(AbstractTDA, self).__init__()
        self.vocabs = vocabs
        self.mode = "inference"

    def inference(self):
        self.mode = "inference"

    def genconv_post(self):
        self.mode = "genconv-post"

    def genconv_prior(self):
        self.mode = "genconv-prior"

    def encode(self):
        self.mode = "encode"

    def decode_optimal(self):
        self.mode = "decode-optimal"

    def _encode_impl(self, data, *args, **kwargs):
        """Encodes the dialog data into latent variables.

        Arguments:
            data (dict): Data to be used for posterior calculation.
                The data is specified using a dictionary with at
                least the following keys.

                conv_lens (LongTensor): [batch_size]
                sent (LongTensor): [batch_size x max_conv_len x max_sent_len]
                sent_lens (LongTensor): [batch_size x max_conv_len]
                speaker (LongTensor): [batch_size x max_conv_len]
                goal (LongTensor): [batch_size x max_conv_len x max_goal_lens]
                goal_lens (LongTensor): [batch_size x max_conv_len]
                state (LongTensor): [batch_size x max_conv_len x max_state_lens]
                state_lens (LongTensor): [batch_size x max_conv_len]
                asv (LongTensor): [num_asv x max_asv_len]
                asv_lens (LongTensor): [num_asv]

        Returns:
            post (dict): posterior distributions containing following keys.
                zconv (MultiGaussian): gaussian distribution object for
                    conversation latent variables.
        """
        raise NotImplementedError

    def _decode_optimal_impl(self, data, *args, **kwargs):
        """Decodes and generates a dialog from the given conversation
        latent variable vector. Must use optimal decoding method (in contrast
        to greedy method).


        Arguments:
            data (dict): Data to be used for decoding.
                The data is specified using a dictionary with at
                least the following keys.

                zconv (FloatTensor): [batch_size x conv_dim]
                asv (LongTensor): [num_asv x max_asv_len]
                asv_lens (LongTensor): [num_asv]
        """
        raise NotImplementedError

    @staticmethod
    def tuplize_data(data: dict):
        return tuple(data[k] for k in
                     ["conv_lens", "sent", "sent_lens", "speaker", "goal",
                      "goal_lens", "state", "state_lens", "asv", "asv_lens"])

    def _genconv_prior_impl(self, data, *args, **kwargs):
        """Generates entire conversations from the prior distribution

        Arguments:
            data (dict): Data to be used for posterior calculation.
                The data is specified using a dictionary with at
                least the following keys.

                n (LongTensor): a zero-dimensional Tensor that
                    contains the number of samples to generate.
                asv (LongTensor): [num_asv x max_asv_len]
                asv_lens (LongTensor): [num_asv]
        """
        raise NotImplementedError

    def _genconv_post_impl(self, data, *args, **kwargs):
        """Generates entire conversations from the posterior
        distribution depending on the input type.

        Arguments:
            data (dict): Data to be used for posterior calculation.
                The data is specified using a dictionary with at
                least the following keys.

                conv_lens (LongTensor): [batch_size]
                sent (LongTensor): [batch_size x max_conv_len x max_sent_len]
                sent_lens (LongTensor): [batch_size x max_conv_len]
                speaker (LongTensor): [batch_size x max_conv_len]
                goal (LongTensor): [batch_size x max_conv_len x max_goal_lens]
                goal_lens (LongTensor): [batch_size x max_conv_len]
                state (LongTensor): [batch_size x max_conv_len x max_state_lens]
                state_lens (LongTensor): [batch_size x max_conv_len]
                asv (LongTensor): [num_asv x max_asv_len]
                asv_lens (LongTensor): [num_asv]
        """
        raise NotImplementedError

    def _genturn_impl(self, data, *args, **kwargs):
        """Generates turns with given contexts from conditional priors.

        Arguments:
            data (dict): Data to be used for posterior calculation.
                The data is specified using a dictionary with at
                least the following keys.

                conv_lens (LongTensor): [batch_size]
                sent (LongTensor): [batch_size x max_conv_len x max_sent_len]
                sent_lens (LongTensor): [batch_size x max_conv_len]
                speaker (LongTensor): [batch_size x max_conv_len]
                goal (LongTensor): [batch_size x max_conv_len x max_goal_lens]
                goal_lens (LongTensor): [batch_size x max_conv_len]
                state (LongTensor): [batch_size x max_conv_len x max_state_lens]
                state_lens (LongTensor): [batch_size x max_conv_len]
                asv (LongTensor): [num_asv x max_asv_len]
                asv_lens (LongTensor): [num_asv]
        """
        raise NotImplementedError

    def _inference_impl(self, data, *args, **kwargs):
        """
        Arguments:
            data (dict): Data to be used for posterior calculation.
                The data is specified using a dictionary with at
                least the following keys.

                conv_lens (LongTensor): [batch_size]
                sent (LongTensor): [batch_size x max_conv_len x max_sent_len]
                sent_lens (LongTensor): [batch_size x max_conv_len]
                speaker (LongTensor): [batch_size x max_conv_len]
                goal (LongTensor): [batch_size x max_conv_len x max_goal_lens]
                goal_lens (LongTensor): [batch_size x max_conv_len]
                state (LongTensor): [batch_size x max_conv_len x max_state_lens]
                state_lens (LongTensor): [batch_size x max_conv_len]
                asv (LongTensor): [num_asv x max_asv_len]
                asv_lens (LongTensor): [num_asv]
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """Computes forward computation, depending on the mode currently set on.
        Supported modes are as follows.

            - Inference Mode: `self.inference_mode()`
            - Conversation Generation Mode: `self.gen_conv_mode()`
            - Utterance Generation Mode: `self.utt_conv_mode()`

        Returns:
            # TODO: update return signature
            logit_dict (Mapping):
                sent (FloatTensor):
                    [batch_size x max_conv_len x max_sent_len x vocab_size]
                speaker_logits (FloatTensor):
                    [batch_size x max_conv_len x num_speakers]
                goal (FloatTensor):
                    [batch_size x max_conv_len x num_labels x max_num_classes]
                turn (FloatTensor):
                    [batch_size x max_conv_len x num_labels x max_num_classes]
            post_dict (Mapping[str, MultiGaussian]):
                sent (MultiGaussian): sent posterior distribution
                speaker (MultiGaussian): speaker posterior distribution
                goal (MultiGaussian): goal posterior distribution
                turn (MultiGaussian): turn posterior distribution
                conv (MultiGaussian): conv posterior distribution
            prior_dict (Mapping[str, MultiGaussian]):
                sent (MultiGaussian): sent prior distribution
                speaker (MultiGaussian): speaker prior distribution
                goal (MultiGaussian): goal prior distribution
                turn (MultiGaussian): turn prior distribution
                conv (MultiGaussian): conv prior distribution
        """
        if self.mode == "inference":
            return self._inference_impl(*args, **kwargs)
        elif self.mode == "genconv-post":
            return self._genconv_post_impl(*args, **kwargs)
        elif self.mode == "genconv-prior":
            return self._genconv_prior_impl(*args, **kwargs)
        elif self.mode == "genturn":
            return self._genturn_impl(*args, **kwargs)
        elif self.mode == "encode":
            return self._encode_impl(*args, **kwargs)
        elif self.mode == "decode-optimal":
            return self._decode_optimal_impl(*args, **kwargs)
        else:
            raise ValueError(f"unsupported mode: {self.mode}")


class VHCR(AbstractTDA):
    name = "vhcr"

    def __init__(self, *args,
                 conv_dim=256, word_dim=100, sent_dim=128, ctx_dim=256,
                 zsent_dim=16, zconv_dim=16,
                 sent_dropout=0.0, word_dropout=0.0,
                 word_encoder=AbstractWordEncoder,
                 seq_encoder=AbstractSequenceEncoder,
                 sent_decoder=AbstractSentDecoder,
                 ctx_encoder=AbstractContextEncoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_dim = word_dim
        self.sent_dim = sent_dim
        self.conv_dim = conv_dim
        self.ctx_dim = ctx_dim
        self.zconv_dim = zconv_dim
        self.zsent_dim = zsent_dim
        self.word_encoder_cls = word_encoder
        self.seq_encoder_cls = seq_encoder
        self.sent_decoder_cls = sent_decoder
        self.ctx_encoder_cls = ctx_encoder
        self.sent_dropout = sent_dropout
        self.word_dropout = word_dropout

        self.word_encoder = self.word_encoder_cls(
            vocab=self.vocabs.word,
            word_dim=self.word_dim
        )
        self.sent_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=0,
            hidden_dim=self.sent_dim
        )
        self.conv_encoder = self.seq_encoder_cls(
            input_dim=self.sent_dim,
            query_dim=0,
            hidden_dim=self.conv_dim
        )
        self.ctx_encoder = self.ctx_encoder_cls(
            input_dim=(self.zconv_dim + self.sent_dim),
            ctx_dim=self.ctx_dim
        )
        self.sent_decoder = self.sent_decoder_cls(
            vocab=self.vocabs.word,
            word_encoder=self.word_encoder,
            hidden_dim=(self.ctx_dim + self.zconv_dim + self.zsent_dim)
        )
        self.zconv_post = MultiGaussianLayer(
            input_dim=self.conv_dim,
            hidden_dim=self.zconv_dim
        )
        self.zsent_prior = MultiGaussianLayer(
            input_dim=(self.ctx_dim + self.zconv_dim),
            hidden_dim=self.zsent_dim
        )
        self.zsent_post = MultiGaussianLayer(
            input_dim=(self.ctx_dim + self.zconv_dim + self.sent_dim),
            hidden_dim=self.zsent_dim
        )
        self.sent_unk = nn.Parameter(torch.zeros(self.sent_dim))
        self.word_unk = nn.Parameter(torch.zeros(self.word_dim))

    @property
    def num_speakers(self):
        return self.vocabs.num_speakers

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.sent_unk, 0, 1 / math.sqrt(self.sent_dim))
        nn.init.normal_(self.word_unk, 0, 1 / math.sqrt(self.word_dim))

    def dropout(self, x, dropout_rate, unk):
        if not self.training:
            return x
        x_size = x.size()
        x = x.view(-1, x.size(-1))
        idx = dist.Bernoulli(dropout_rate).sample((x.size(0),)).bool()
        if idx.any():
            x[idx] = unk
        return x.view(*x_size)

    def _inference_impl(self, data, sample_scale=1.0,
                        dropout_scale=1.0, **kwargs):
        conv_lens, w, w_lens, p, g, g_lens, s, s_lens, asv, asv_lens = \
            self.tuplize_data(data)
        batch_size, max_conv_len, max_sent_len = w.size()
        max_goal_len, max_state_len = g.size(-1), s.size(-1)
        num_asv, max_asv_len = asv.size()
        num_speakers = len(self.vocabs.speaker)
        conv_mask = utils.mask(conv_lens)
        w_emb = self.word_encoder(w, w_lens)

        u = self.sent_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = u.masked_fill(~conv_mask.unsqueeze(-1), 0)
        _, c = self.conv_encoder(
            u,
            conv_lens,
        )
        zconv_post: MultiGaussian = self.zconv_post(c)
        zconv = zconv_post.sample(sample_scale)
        zconv_exp = zconv.unsqueeze(1).expand(-1, max_conv_len, -1)
        zconv_exp_flat = zconv_exp.reshape(-1, self.zconv_dim)

        w_emb = self.dropout(w_emb,
                             dropout_rate=self.word_dropout * dropout_scale,
                             unk=self.word_unk)
        u = self.sent_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1)
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = self.dropout(u, self.sent_dropout * dropout_scale, self.sent_unk)
        context, _, _ = self.ctx_encoder(
            torch.cat([
                utils.shift(u, dim=1),
                zconv.unsqueeze(1).expand(-1, max_conv_len, -1)
            ], 2),
            conv_lens
        )
        zsent_post: MultiGaussian = self.zsent_post(
            torch.cat([
                u.view(-1, self.sent_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zsent_dim)
        zsent_prior: MultiGaussian = self.zsent_prior(
            torch.cat([
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zsent_dim)
        zsent = (zsent_post.sample(sample_scale)
                 .view(batch_size, max_conv_len, -1)
                 .masked_fill(~conv_mask.unsqueeze(-1), 0))
        w_logit = self.sent_decoder(
            (torch.cat([zsent, context, zconv_exp], 2)
             .view(batch_size * max_conv_len, -1)),
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1)
        ).view(batch_size, max_conv_len, max_sent_len, -1)
        # gibberish
        p_logit = w_logit.new(batch_size, max_conv_len, num_speakers).normal_()
        g_logit = w_logit.new(batch_size, max_conv_len, num_asv).normal_()
        s_logit = w_logit.new(batch_size, max_conv_len, num_asv).normal_()
        return (
            {  # logits
                "sent": w_logit,
                "speaker": p_logit,
                "goal": g_logit,
                "state": s_logit
            }, {  # posterior
                "conv": zconv_post,
                "sent": zsent_post,
                "speaker": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                            .to(w_logit.device)),
                "goal": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                         .to(w_logit.device)),
                "state": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                          .to(w_logit.device)),
            }, {  # prior
                "conv": (MultiGaussian.unit(*zconv_post.size())
                         .to(w_logit.device)),
                "sent": zsent_prior,
                "speaker": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                            .to(w_logit.device)),
                "goal": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                         .to(w_logit.device)),
                "state": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                          .to(w_logit.device)),
            }
        )

    def _encode_impl(self, data, **kwargs):
        conv_lens = data["conv_lens"]
        w, w_lens = data["sent"], data["sent_lens"]
        batch_size, max_conv_len, max_sent_len = w.size()
        conv_mask = utils.mask(conv_lens)
        w_emb = self.word_encoder(w, w_lens)
        u = self.sent_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1)
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = u.masked_fill(~conv_mask.unsqueeze(-1), 0)
        _, c = self.conv_encoder(u, conv_lens)
        return self.zconv_post(c)

    def _decode_optimal_impl(self, data,
                             conv_state=None,
                             init_u=None,
                             max_conv_len=20, max_sent_len=30, sent_scale=1.0,
                             beam_size=8, **kwargs):
        zconv = data["zconv"]
        batch_size = zconv.size(0)
        eoc_idx = self.vocabs.word["<eoc>"]
        asv_pad = ActSlotValue("<pad>", "<pad>", "<pad>")
        asv_pad_idx = self.vocabs.goal_state.asv[asv_pad]
        sents, spkrs, goals, states = list(), list(), list(), list()
        conv_lens = zconv.new(batch_size).long().zero_()
        conv_done = zconv.new(batch_size).bool().zero_()
        sent_logprob = zconv.new(batch_size).zero_()
        if init_u is None:
            u = zconv.new(batch_size, self.sent_dim).zero_()
        else:
            u = init_u
        if conv_state is None:
            conv_state = self.ctx_encoder.init_state(batch_size)
        for i in range(max_conv_len):
            context, _, conv_state = self.ctx_encoder(
                torch.cat([u, zconv], 1).unsqueeze(1),
                lens=u.new(batch_size).long().fill_(1),
                h=conv_state
            )
            context = context.view(-1, self.ctx_dim)
            zutt = self.zsent_prior(
                torch.cat([context, zconv], 1)
            ).sample(sent_scale)
            w, w_lens, w_prob = self.sent_decoder.generate(
                h=torch.cat([
                    zutt,
                    context.view(batch_size, self.ctx_dim),
                    zconv
                ], 1),
                beam_size=beam_size,
                max_len=max_sent_len
            )
            w_logprob = w_prob.log()
            sents.append(Stacked1DTensor(w, w_lens))
            _, u = self.sent_encoder(self.word_encoder(w, w_lens), w_lens)
            spkrs.append(w.new(batch_size).zero_().long())
            goals.append(Stacked1DTensor(
                value=w.new(batch_size, 0).zero_().long(),
                lens=w.new(batch_size).zero_().long()
            ))
            states.append(Stacked1DTensor(
                value=w.new(batch_size, 0).zero_().long(),
                lens=w.new(batch_size).zero_().long()
            ))
            conv_lens += 1 - conv_done.long()
            done = ((w == eoc_idx) & utils.mask(w_lens, w.size(1))).any(1)
            conv_done |= done
            sent_logprob = w_logprob.masked_fill(conv_done, 0)
            if conv_done.all().item():
                break
        sents = utils.stack_stacked1dtensors(sents)
        goals = utils.stack_stacked1dtensors(goals)
        states = utils.stack_stacked1dtensors(states)
        return BatchData(
            sent=DoublyStacked1DTensor(
                value=sents.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=sents.lens1.t().contiguous()
            ),
            speaker=Stacked1DTensor(
                value=torch.stack(spkrs).transpose(1, 0).contiguous(),
                lens=conv_lens
            ),
            goal=DoublyStacked1DTensor(
                value=goals.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=goals.lens1.t().contiguous()
            ),
            state=DoublyStacked1DTensor(
                value=states.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=states.lens1.t().contiguous()
            ),
        ), dict(
            logprob=sent_logprob,
            sent_logprob=sent_logprob,
            goal_logprob=sent_logprob.clone().fill_(float("nan")),
            state_logprob=sent_logprob.clone().fill_(float("nan")),
            spkr_logprob=sent_logprob.clone().fill_(float("nan")),
        )

    def _genconv_prior_impl(self, data, max_conv_len=20,
                            beam_size=4, max_sent_len=30,
                            conv_scale=1.0, spkr_scale=1.0,
                            goal_scale=1.0, state_scale=1.0, sent_scale=1.0):
        n = data["n"]
        zconv = MultiGaussian.unit(n.item(), self.ctx_dim).to(n.device)
        zconv_sample = zconv.sample(conv_scale)
        batch, info = self._decode_optimal_impl(
            data={"zconv": zconv_sample, "asv": data["asv"],
                  "asv_lens": data["asv_lens"]},
            eoc="<eoc>",
            max_conv_len=max_conv_len,
            spkr_scale=spkr_scale,
            goal_scale=goal_scale,
            state_scale=state_scale,
            sent_scale=sent_scale,
            beam_size=beam_size,
            max_sent_len=max_sent_len
        )
        info["zconv"] = zconv
        info["zconv-sample"] = zconv_sample
        return batch, info

    def _genconv_post_impl(self, data, max_conv_len=20,
                           beam_size=4, max_sent_len=30,
                           conv_scale=1.0, spkr_scale=1.0,
                           goal_scale=1.0, state_scale=1.0, sent_scale=1.0):
        zconv = self._encode_impl(data)
        zconv_sample = zconv.sample(conv_scale)
        batch, info = self._decode_optimal_impl(
            data={"zconv": zconv_sample, "asv": data["asv"],
                  "asv_lens": data["asv_lens"]},
            eoc="<eoc>",
            max_conv_len=max_conv_len,
            spkr_scale=spkr_scale,
            goal_scale=goal_scale,
            state_scale=state_scale,
            sent_scale=sent_scale,
            beam_size=beam_size,
            max_sent_len=max_sent_len
        )
        info["zconv"] = zconv
        info["zconv-sample"] = zconv_sample
        return batch, info


class VHUS(AbstractTDA):
    name = "vhus"

    def __init__(self, *args,
                 word_dim=150, asv_dim=200, conv_dim=200,
                 goal_dim=200, state_dim=200, zstate_dim=200,
                 asv_dropout=0.0,
                 word_encoder=AbstractWordEncoder,
                 seq_encoder=AbstractSequenceEncoder,
                 ctx_encoder=AbstractContextEncoder,
                 state_decoder=AbstractStateDecoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.asv_dim = asv_dim
        self.conv_dim = conv_dim
        self.word_dim = word_dim
        self.goal_dim = goal_dim
        self.state_dim = state_dim
        self.zstate_dim = zstate_dim
        self.asv_dropout = asv_dropout
        self.word_encoder_cls = word_encoder
        self.seq_encoder_cls = seq_encoder
        self.ctx_encoder_cls = ctx_encoder
        self.state_decoder_cls = state_decoder

        self.asv_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=0,
            hidden_dim=self.asv_dim
        )
        self.word_encoder = self.word_encoder_cls(
            vocab=self.vocabs.word,
            word_dim=self.word_dim
        )
        self.goal_encoder = self.seq_encoder_cls(
            input_dim=self.asv_dim,
            query_dim=0,
            hidden_dim=self.goal_dim
        )
        self.state_encoder = self.seq_encoder_cls(
            input_dim=self.asv_dim,
            query_dim=0,
            hidden_dim=self.state_dim
        )
        self.ctx_encoder = self.ctx_encoder_cls(
            input_dim=self.goal_dim + self.state_dim,
            ctx_dim=self.conv_dim
        )
        self.state_decoder = self.state_decoder_cls(
            vocabs=self.vocabs, state_type="state",
            # conv_rnn + z_conv + z_speaker + z_goal + z_turn
            input_dim=self.zstate_dim,
            asv_dim=self.asv_dim
        )
        self.zstate_post = MultiGaussianLayer(
            input_dim=self.conv_dim,
            hidden_dim=self.zstate_dim
        )
        self.zstate_prior = MultiGaussianLayer(
            input_dim=self.conv_dim,
            hidden_dim=self.zstate_dim
        )
        self.asv_unk = nn.Parameter(torch.zeros(self.asv_dim))
        self.spkr_eye = nn.Parameter(torch.eye(self.num_speakers),
                                     requires_grad=False)

    @property
    def num_speakers(self):
        return self.vocabs.num_speakers

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.asv_unk, 0, 1 / math.sqrt(self.word_dim))

    def dropout(self, x, dropout_rate, unk):
        if not self.training:
            return x
        x_size = x.size()
        x = x.view(-1, x.size(-1))
        idx = dist.Bernoulli(dropout_rate).sample((x.size(0),)).bool()
        if idx.any():
            x[idx] = unk
        return x.view(*x_size)

    def _inference_impl(self, data, sample_scale=1.0,
                        dropout_scale=1.0, **kwargs):
        conv_lens, w, w_lens, p, g, g_lens, s, s_lens, asv, asv_lens = \
            self.tuplize_data(data)
        batch_size, max_conv_len, max_sent_len = w.size()
        max_goal_len, max_state_len = g.size(-1), s.size(-1)
        num_asv, max_asv_len = asv.size()
        conv_mask = utils.mask(conv_lens)
        g_mask = utils.mask(g_lens) & conv_mask.unsqueeze(-1)
        s_mask = utils.mask(s_lens) & conv_mask.unsqueeze(-1)
        asv_emb = self.word_encoder(asv, asv_lens)
        _, asv_h = self.asv_encoder(asv_emb, asv_lens)

        g_asv = self.dropout(
            asv_h[g.masked_fill(~g_mask, 0)],
            dropout_rate=self.asv_dropout * dropout_scale,
            unk=self.asv_unk
        )
        v = self.goal_encoder(
            g_asv.view(-1, max_goal_len, self.asv_dim),
            g_lens.view(-1)
        )[1].view(batch_size, max_conv_len, self.goal_dim)
        s_asv = self.dropout(
            asv_h[s.masked_fill(~s_mask, 0)],
            dropout_rate=self.asv_dropout * dropout_scale,
            unk=self.asv_unk
        )
        t = self.state_encoder(
            s_asv.view(-1, max_state_len, self.asv_dim),
            s_lens.view(-1)
        )[1].view(batch_size, max_conv_len, self.state_dim)
        ctx, _, _ = self.ctx_encoder(torch.cat([v, t], -1), conv_lens)
        zstate_post: MultiGaussian = self.zstate_post(ctx)
        zstate_prior: MultiGaussian = self.zstate_prior(
            utils.shift(ctx, n=1, dim=1)
        )
        if self.training:
            zstate = zstate_post.sample()
        else:
            zstate = zstate_prior.sample()

        s_logit = self.state_decoder(
            zstate.view(batch_size * max_conv_len, -1),
            p.view(batch_size * max_conv_len),
            asv_h
        ).view(batch_size, max_conv_len, -1)
        # gibberish
        w_logit = s_logit.new(batch_size, max_conv_len,
                              max_sent_len, len(self.vocabs.word)).normal_()
        p_logit = s_logit.new(batch_size, max_conv_len,
                              self.num_speakers).normal_()
        g_logit = s_logit.new(batch_size, max_conv_len, num_asv).normal_()
        return (
            {  # logits
                "sent": w_logit,
                "speaker": p_logit,
                "goal": g_logit,
                "state": s_logit
            }, {  # posterior
                "conv": (MultiGaussian.unit(batch_size, 1)
                         .to(w_logit.device)),
                "sent": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                         .to(s_logit.device)),
                "speaker": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                            .to(w_logit.device)),
                "goal": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                         .to(w_logit.device)),
                "state": zstate_post,
            }, {  # prior
                "conv": (MultiGaussian.unit(batch_size, 1)
                         .to(w_logit.device)),
                "sent": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                         .to(w_logit.device)),
                "speaker": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                            .to(w_logit.device)),
                "goal": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                         .to(w_logit.device)),
                "state": zstate_prior,
            }
        )

    def _genconv_prior_impl(self, data, max_conv_len=20,
                            beam_size=4, max_sent_len=30,
                            conv_scale=1.0, spkr_scale=1.0,
                            goal_scale=1.0, state_scale=1.0, sent_scale=1.0):
        raise NotImplementedError

    def _encode_impl(self, data, *args, **kwargs):
        raise NotImplementedError

    def _decode_optimal_impl(self, data, *args, **kwargs):
        raise NotImplementedError

    def _genconv_post_impl(self, data, beam_size=4, max_sent_len=30,
                           sent_scale=1.0, **kwargs):
        conv_lens, w, w_lens, p, g, g_lens, s, s_lens, asv, asv_lens = \
            self.tuplize_data(data)
        batch_size, max_conv_len, max_sent_len = w.size()
        max_goal_len, max_state_len = g.size(-1), s.size(-1)
        num_asv, max_asv_len = asv.size()
        conv_mask = utils.mask(conv_lens)
        g_mask = utils.mask(g_lens) & conv_mask.unsqueeze(-1)
        s_mask = utils.mask(s_lens) & conv_mask.unsqueeze(-1)
        asv_emb = self.word_encoder(asv, asv_lens)
        _, asv_h = self.asv_encoder(asv_emb, asv_lens)

        g_asv = asv_h[g.masked_fill(~g_mask, 0)]
        v = self.goal_encoder(
            g_asv.view(-1, max_goal_len, self.asv_dim),
            g_lens.view(-1)
        )[1].view(batch_size, max_conv_len, self.goal_dim)
        s_asv = asv_h[s.masked_fill(~s_mask, 0)]
        t = self.state_encoder(
            s_asv.view(-1, max_state_len, self.asv_dim),
            s_lens.view(-1)
        )[1].view(batch_size, max_conv_len, self.state_dim)
        ctx, _, _ = self.ctx_encoder(torch.cat([v, t], -1), conv_lens)
        zstate_prior: MultiGaussian = self.zstate_prior(
            utils.shift(ctx, n=1, dim=1)
        )
        zstate = zstate_prior.sample()

        s_logit = self.state_decoder(
            zstate.view(batch_size * max_conv_len, -1),
            p.view(batch_size * max_conv_len),
            asv_h
        )
        s_prob = torch.sigmoid(s_logit)
        s_prime = utils.to_sparse(s_prob > 0.5)
        logprob = s_prob.masked_fill(s_prob == 0, 1.0).log().sum(-1)
        return BatchData(
            sent=DoublyStacked1DTensor(
                value=w.view(batch_size, max_conv_len, -1),
                lens=conv_lens,
                lens1=w_lens.view(batch_size, max_conv_len)
            ),
            speaker=Stacked1DTensor(
                value=w.new(batch_size, max_conv_len).zero_().long(),
                lens=conv_lens
            ),
            goal=DoublyStacked1DTensor(
                value=w.new(batch_size, max_conv_len, 0).zero_().long(),
                lens=conv_lens,
                lens1=w.new(batch_size, max_conv_len).zero_().long()
            ),
            state=DoublyStacked1DTensor(
                value=s_prime.value.view(batch_size, max_conv_len, -1),
                lens=conv_lens,
                lens1=s_prime.lens.view(batch_size, max_conv_len)
            ),
        ), dict(
            logprob=logprob,
            sent_logprob=logprob.clone().fill_(float("nan")),
            goal_logprob=logprob.clone().fill_(float("nan")),
            state_logprob=logprob,
            spkr_logprob=logprob.clone().fill_(float("nan")),
        )


class VHRED(AbstractTDA):
    name = "vhred"

    def __init__(self, *args,
                 conv_dim=256, word_dim=100, sent_dim=128, zsent_dim=16,
                 word_dropout=0.0, sent_dropout=0.0,
                 word_encoder=AbstractWordEncoder,
                 seq_encoder=AbstractSequenceEncoder,
                 sent_decoder=AbstractSentDecoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_dim = word_dim
        self.sent_dim = sent_dim
        self.conv_dim = conv_dim
        self.zsent_dim = zsent_dim
        self.word_encoder_cls = word_encoder
        self.seq_encoder_cls = seq_encoder
        self.sent_decoder_cls = sent_decoder
        self.word_dropout = word_dropout
        self.sent_dropout = sent_dropout

        self.word_encoder = self.word_encoder_cls(
            vocab=self.vocabs.word,
            word_dim=self.word_dim
        )
        self.sent_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=0,
            hidden_dim=self.sent_dim
        )
        self.conv_encoder = self.seq_encoder_cls(
            input_dim=self.sent_dim,
            query_dim=0,
            hidden_dim=self.conv_dim
        )
        self.sent_decoder = self.sent_decoder_cls(
            vocab=self.vocabs.word,
            word_encoder=self.word_encoder,
            hidden_dim=(self.conv_dim + self.zsent_dim)
        )
        self.zsent_post = MultiGaussianLayer(
            input_dim=(self.conv_dim + self.conv_dim),
            hidden_dim=self.zsent_dim
        )
        self.zsent_prior = MultiGaussianLayer(
            input_dim=self.conv_dim,
            hidden_dim=self.zsent_dim
        )
        self.word_unk = nn.Parameter(torch.zeros(self.word_dim))
        self.sent_unk = nn.Parameter(torch.zeros(self.sent_dim))

    @property
    def num_speakers(self):
        return self.vocabs.num_speakers

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.word_unk, 0, 1 / math.sqrt(self.word_dim))

    def dropout(self, x, dropout_rate, unk):
        if not self.training:
            return x
        x_size = x.size()
        x = x.view(-1, x.size(-1))
        idx = dist.Bernoulli(dropout_rate).sample((x.size(0),)).bool()
        if idx.any():
            x[idx] = unk
        return x.view(*x_size)

    def _inference_impl(self, data, sample_scale=1.0,
                        dropout_scale=1.0, **kwargs):
        conv_lens, w, w_lens, p, g, g_lens, s, s_lens, asv, asv_lens = \
            self.tuplize_data(data)
        batch_size, max_conv_len, max_sent_len = w.size()
        num_asv, max_asv_len = asv.size()
        num_speakers = len(self.vocabs.speaker)
        conv_mask = utils.mask(conv_lens)
        w_emb = self.word_encoder(w, w_lens)

        u = self.sent_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = u.masked_fill(~conv_mask.unsqueeze(-1), 0)
        u = self.dropout(u, self.sent_dropout, self.sent_unk)
        ctx, _ = self.conv_encoder(u, conv_lens)
        zsent_post: MultiGaussian = self.zsent_post(
            torch.cat([utils.shift(ctx, n=1, dim=1), ctx], -1)
        )
        zsent_prior: MultiGaussian = self.zsent_prior(ctx)
        zsent = zsent_post.sample()

        w_emb = self.dropout(w_emb,
                             dropout_rate=self.word_dropout * dropout_scale,
                             unk=self.word_unk)
        w_logit = self.sent_decoder(
            (torch.cat([zsent, ctx], 2)
             .view(batch_size * max_conv_len, -1)),
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1)
        ).view(batch_size, max_conv_len, max_sent_len, -1)
        # gibberish
        p_logit = w_logit.new(batch_size, max_conv_len, num_speakers).normal_()
        g_logit = w_logit.new(batch_size, max_conv_len, num_asv).normal_()
        s_logit = w_logit.new(batch_size, max_conv_len, num_asv).normal_()
        return (
            {  # logits
                "sent": w_logit,
                "speaker": p_logit,
                "goal": g_logit,
                "state": s_logit
            }, {  # posterior
                "conv": (MultiGaussian.unit(batch_size, 1)
                         .to(w_logit.device)),
                "sent": zsent_post,
                "speaker": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                            .to(w_logit.device)),
                "goal": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                         .to(w_logit.device)),
                "state": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                          .to(w_logit.device)),
            }, {  # prior
                "conv": (MultiGaussian.unit(batch_size, 1)
                         .to(w_logit.device)),
                "sent": zsent_prior,
                "speaker": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                            .to(w_logit.device)),
                "goal": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                         .to(w_logit.device)),
                "state": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                          .to(w_logit.device)),
            }
        )

    def _genconv_prior_impl(self, data, max_conv_len=20,
                            beam_size=4, max_sent_len=30,
                            conv_scale=1.0, spkr_scale=1.0,
                            goal_scale=1.0, state_scale=1.0, sent_scale=1.0):
        raise NotImplementedError

    def _encode_impl(self, data, *args, **kwargs):
        raise NotImplementedError

    def _decode_optimal_impl(self, data, *args, **kwargs):
        raise NotImplementedError

    def _genconv_post_impl(self, data, beam_size=4, max_sent_len=30,
                           sent_scale=1.0, **kwargs):
        conv_lens, w, w_lens, p, g, g_lens, s, s_lens, asv, asv_lens = \
            self.tuplize_data(data)
        batch_size, max_conv_len, max_sent_len = w.size()
        conv_mask = utils.mask(conv_lens)
        w_emb = self.word_encoder(w, w_lens)
        u = self.sent_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = u.masked_fill(~conv_mask.unsqueeze(-1), 0)
        ctx, _ = self.conv_encoder(u, conv_lens)
        zsent_post: MultiGaussian = self.zsent_post(
            torch.cat([utils.shift(ctx, n=1, dim=1), ctx], -1)
        )
        zsent = zsent_post.sample()
        w, w_lens, w_prob = self.sent_decoder.generate(
            h=torch.cat([zsent.view(-1, self.zsent_dim),
                         ctx.view(-1, self.conv_dim), ], 1),
            beam_size=beam_size,
            max_len=max_sent_len
        )
        sent_logprob = w_prob.log().view(batch_size, max_conv_len).sum(-1)
        return BatchData(
            sent=DoublyStacked1DTensor(
                value=w.view(batch_size, max_conv_len, -1),
                lens=conv_lens,
                lens1=w_lens.view(batch_size, max_conv_len)
            ),
            speaker=Stacked1DTensor(
                value=w.new(batch_size, max_conv_len).zero_().long(),
                lens=conv_lens
            ),
            goal=DoublyStacked1DTensor(
                value=w.new(batch_size, max_conv_len, 0).zero_().long(),
                lens=conv_lens,
                lens1=w.new(batch_size, max_conv_len).zero_().long()
            ),
            state=DoublyStacked1DTensor(
                value=w.new(batch_size, max_conv_len, 0).zero_().long(),
                lens=conv_lens,
                lens1=w.new(batch_size, max_conv_len).zero_().long()
            ),
        ), dict(
            logprob=sent_logprob,
            sent_logprob=sent_logprob,
            goal_logprob=sent_logprob.clone().fill_(float("nan")),
            state_logprob=sent_logprob.clone().fill_(float("nan")),
            spkr_logprob=sent_logprob.clone().fill_(float("nan")),
        )


class VHDA(AbstractTDA):
    name = "vhda"

    def __init__(self, *args,
                 conv_dim=256, word_dim=100, sent_dim=128, asv_dim=64,
                 state_dim=32, goal_dim=32, spkr_dim=8, ctx_dim=256,
                 zsent_dim=16, zstate_dim=8, zgoal_dim=8,
                 zspkr_dim=2, zconv_dim=16,
                 spkr_dropout=0.0, asv_dropout=0.0, goal_dropout=0.0,
                 state_dropout=0.0, sent_dropout=0.0, word_dropout=0.0,
                 word_encoder=AbstractWordEncoder,
                 seq_encoder=AbstractSequenceEncoder,
                 sent_decoder=AbstractSentDecoder,
                 state_decoder=AbstractStateDecoder,
                 ctx_encoder=AbstractContextEncoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_dim = word_dim
        self.sent_dim = sent_dim
        self.asv_dim = asv_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.spkr_dim = spkr_dim
        self.conv_dim = conv_dim
        self.ctx_dim = ctx_dim
        self.zconv_dim = zconv_dim
        self.zspkr_dim = zspkr_dim
        self.zgoal_dim = zgoal_dim
        self.zstate_dim = zstate_dim
        self.zsent_dim = zsent_dim
        self.word_encoder_cls = word_encoder
        self.seq_encoder_cls = seq_encoder
        self.state_decoder_cls = state_decoder
        self.sent_decoder_cls = sent_decoder
        self.ctx_encoder_cls = ctx_encoder
        self.asv_dropout = asv_dropout
        self.spkr_dropout = spkr_dropout
        self.goal_dropout = goal_dropout
        self.state_dropout = state_dropout
        self.sent_dropout = sent_dropout
        self.word_dropout = word_dropout

        self.word_encoder = self.word_encoder_cls(
            vocab=self.vocabs.word,
            word_dim=self.word_dim
        )
        self.asv_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=0,
            hidden_dim=self.asv_dim
        )
        self.post_query = nn.Parameter(torch.randn(self.sent_dim))
        self.sent_post_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=self.sent_dim,
            hidden_dim=self.sent_dim
        )
        self.conv_encoder = self.seq_encoder_cls(
            input_dim=self.sent_dim,
            query_dim=self.sent_dim,
            hidden_dim=self.conv_dim
        )
        self.ctx_encoder = self.ctx_encoder_cls(
            input_dim=(self.zconv_dim + self.spkr_dim +
                       self.goal_dim + self.state_dim + self.sent_dim),
            ctx_dim=self.ctx_dim
        )
        self.spkr_encoder = torchmodels.Linear(
            in_features=len(self.vocabs.speaker),
            out_features=self.spkr_dim
        )
        self.spkr_decoder = torchmodels.Linear(
            # conv_rnn + z_conv + z_speaker
            in_features=self.ctx_dim + self.zconv_dim + self.zspkr_dim,
            out_features=len(self.vocabs.speaker)
        )
        self.goal_encoder = self.seq_encoder_cls(
            input_dim=self.asv_dim,
            query_dim=self.zconv_dim,
            hidden_dim=self.goal_dim
        )
        self.goal_decoder = self.state_decoder_cls(
            vocabs=self.vocabs, state_type="goal",
            # conv_rnn + z_conv + z_speaker + z_goal
            input_dim=(self.ctx_dim + self.zconv_dim +
                       self.zspkr_dim + self.zgoal_dim),
            asv_dim=self.asv_dim
        )
        self.state_encoder = self.seq_encoder_cls(
            input_dim=self.asv_dim,
            query_dim=self.zconv_dim,
            hidden_dim=self.state_dim
        )
        self.state_decoder = self.state_decoder_cls(
            vocabs=self.vocabs, state_type="state",
            # conv_rnn + z_conv + z_speaker + z_goal + z_turn
            input_dim=(self.ctx_dim + self.zconv_dim +
                       self.zspkr_dim + self.zgoal_dim + self.zstate_dim),
            asv_dim=self.asv_dim
        )
        self.sent_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=self.zconv_dim,
            hidden_dim=self.sent_dim
        )
        self.sent_decoder = self.sent_decoder_cls(
            vocab=self.vocabs.word,
            word_encoder=self.word_encoder,
            # conv_rnn + z_conv + z_sp + z_goal + z_turn + z_utt
            hidden_dim=(self.ctx_dim + self.zconv_dim + self.zspkr_dim +
                        self.zgoal_dim + self.zstate_dim + self.zsent_dim)
        )
        self.zconv_post = MultiGaussianLayer(
            input_dim=self.conv_dim,
            hidden_dim=self.zconv_dim
        )
        self.zspkr_prior = MultiGaussianLayer(
            # conv_rnn + z_conv
            input_dim=self.ctx_dim + self.zconv_dim,
            hidden_dim=self.zspkr_dim
        )
        self.zspkr_post = MultiGaussianLayer(
            # conv_rnn + z_conv + h_speaker
            input_dim=self.ctx_dim + self.zconv_dim + self.spkr_dim,
            hidden_dim=self.zspkr_dim
        )
        self.zgoal_prior = MultiGaussianLayer(
            # conv_rnn + z_conv + z_speaker
            input_dim=self.ctx_dim + self.zconv_dim + self.zspkr_dim,
            hidden_dim=self.zgoal_dim
        )
        self.zgoal_post = MultiGaussianLayer(
            # conv_rnn + z_conv + z_speaker + h_goal
            input_dim=(self.ctx_dim + self.zconv_dim +
                       self.zspkr_dim + self.goal_dim),
            hidden_dim=self.zgoal_dim
        )
        self.zstate_prior = MultiGaussianLayer(
            # conv_rnn + z_conv + z_speaker + z_goal
            input_dim=(self.ctx_dim + self.zconv_dim +
                       self.zspkr_dim + self.zgoal_dim),
            hidden_dim=self.zstate_dim
        )
        self.zstate_post = MultiGaussianLayer(
            # conv_rnn + z_conv + z_speaker + z_goal + h_turn
            input_dim=(self.ctx_dim + self.zconv_dim +
                       self.zspkr_dim + self.zgoal_dim + self.state_dim),
            hidden_dim=self.zstate_dim
        )
        self.zsent_prior = MultiGaussianLayer(
            # conv_rnn + z_conv + z_speaker + z_goal + z_turn
            input_dim=(self.ctx_dim + self.zconv_dim + self.zspkr_dim +
                       self.zgoal_dim + self.zstate_dim),
            hidden_dim=self.zsent_dim
        )
        self.zsent_post = MultiGaussianLayer(
            # conv_rnn + z_conv + z_sp + z_goal + z_turn + h_utt
            input_dim=(self.ctx_dim + self.zconv_dim + self.zspkr_dim +
                       self.zgoal_dim + self.zstate_dim + self.sent_dim),
            hidden_dim=self.zsent_dim
        )
        self.spkr_unk = nn.Parameter(torch.zeros(self.spkr_dim))
        self.goal_unk = nn.Parameter(torch.zeros(self.goal_dim))
        self.state_unk = nn.Parameter(torch.zeros(self.state_dim))
        self.sent_unk = nn.Parameter(torch.zeros(self.sent_dim))
        self.word_unk = nn.Parameter(torch.zeros(self.word_dim))
        self.asv_unk = nn.Parameter(torch.zeros(self.asv_dim))
        self.spkr_eye = nn.Parameter(torch.eye(self.num_speakers),
                                     requires_grad=False)

    @property
    def num_speakers(self):
        return self.vocabs.num_speakers

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.spkr_unk, 0, 1 / math.sqrt(self.spkr_dim))
        nn.init.normal_(self.goal_unk, 0, 1 / math.sqrt(self.goal_dim))
        nn.init.normal_(self.state_unk, 0, 1 / math.sqrt(self.state_dim))
        nn.init.normal_(self.sent_unk, 0, 1 / math.sqrt(self.sent_dim))
        nn.init.normal_(self.word_unk, 0, 1 / math.sqrt(self.word_dim))
        nn.init.normal_(self.asv_unk, 0, 1 / math.sqrt(self.asv_dim))
        nn.init.normal_(self.post_query, 0, 1 / math.sqrt(len(self.post_query)))

    def dropout(self, x, dropout_rate, unk):
        if not self.training:
            return x
        x_size = x.size()
        x = x.view(-1, x.size(-1))
        idx = dist.Bernoulli(dropout_rate).sample((x.size(0),)).bool()
        if idx.any():
            x[idx] = unk
        return x.view(*x_size)

    def _inference_impl(self, data, sample_scale=1.0,
                        dropout_scale=1.0, **kwargs):
        conv_lens, w, w_lens, p, g, g_lens, s, s_lens, asv, asv_lens = \
            self.tuplize_data(data)
        batch_size, max_conv_len, max_sent_len = w.size()
        max_goal_len, max_state_len = g.size(-1), s.size(-1)
        num_asv, max_asv_len = asv.size()
        conv_mask = utils.mask(conv_lens)
        w_mask = utils.mask(w_lens) & conv_mask.unsqueeze(-1)
        g_mask = utils.mask(g_lens) & conv_mask.unsqueeze(-1)
        s_mask = utils.mask(s_lens) & conv_mask.unsqueeze(-1)
        w_emb = self.word_encoder(w, w_lens)
        p_emb = self.spkr_eye[p]
        asv_emb = self.word_encoder(asv, asv_lens)
        _, asv_h = self.asv_encoder(asv_emb, asv_lens)

        u = self.sent_post_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
            self.post_query.unsqueeze(0).expand(batch_size * max_conv_len, -1)
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = u.masked_fill(~conv_mask.unsqueeze(-1), 0)
        _, c = self.conv_encoder(
            u,
            conv_lens,
            self.post_query.unsqueeze(0).expand(batch_size, -1),
        )
        zconv_post: MultiGaussian = self.zconv_post(c)
        zconv = zconv_post.sample(sample_scale)
        zconv_exp = zconv.unsqueeze(1).expand(-1, max_conv_len, -1)
        zconv_exp_flat = zconv_exp.reshape(-1, self.zconv_dim)

        w_emb = self.dropout(w_emb,
                             dropout_rate=self.word_dropout * dropout_scale,
                             unk=self.word_unk)
        u = self.sent_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
            zconv_exp_flat
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        g_asv = self.dropout(
            asv_h[g.masked_fill(~g_mask, 0)],
            dropout_rate=self.asv_dropout * dropout_scale,
            unk=self.asv_unk
        )
        v = self.goal_encoder(
            g_asv.view(-1, max_goal_len, self.asv_dim),
            g_lens.view(-1),
            zconv_exp_flat,
        )[1].view(batch_size, max_conv_len, self.goal_dim)
        s_asv = self.dropout(
            asv_h[s.masked_fill(~s_mask, 0)],
            dropout_rate=self.asv_dropout * dropout_scale,
            unk=self.asv_unk
        )
        t = self.state_encoder(
            s_asv.view(-1, max_state_len, self.asv_dim),
            s_lens.view(-1),
            zconv_exp_flat
        )[1].view(batch_size, max_conv_len, self.state_dim)
        u = self.dropout(u, self.sent_dropout * dropout_scale, self.sent_unk)
        v = self.dropout(v, self.goal_dropout * dropout_scale, self.goal_unk)
        t = self.dropout(t, self.state_dropout * dropout_scale, self.state_unk)
        r = self.dropout(self.spkr_encoder(p_emb),
                         dropout_rate=self.spkr_dropout * dropout_scale,
                         unk=self.spkr_unk)
        context, _, _ = self.ctx_encoder(
            torch.cat([
                utils.shift(u, dim=1),
                utils.shift(v, dim=1),
                utils.shift(t, dim=1),
                utils.shift(r, dim=1),
                zconv.unsqueeze(1).expand(-1, max_conv_len, -1)
            ], 2),
            conv_lens
        )
        zspkr_post: MultiGaussian = self.zspkr_post(
            torch.cat([
                r.view(-1, self.spkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zspkr_dim)
        zspkr_prior: MultiGaussian = self.zspkr_prior(
            torch.cat([
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zspkr_dim)
        zspkr = (zspkr_post.sample(sample_scale)
                 .view(batch_size, max_conv_len, -1)
                 .masked_fill(~conv_mask.unsqueeze(-1), 0))
        zgoal_post: MultiGaussian = self.zgoal_post(
            torch.cat([
                v.view(-1, self.goal_dim),
                zspkr.view(-1, self.zspkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zgoal_dim)
        zgoal_prior: MultiGaussian = self.zgoal_prior(
            torch.cat([
                zspkr.view(-1, self.zspkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zgoal_dim)
        zgoal = (zgoal_post.sample(sample_scale)
                 .view(batch_size, max_conv_len, -1)
                 .masked_fill(~conv_mask.unsqueeze(-1), 0))
        zstate_post: MultiGaussian = self.zstate_post(
            torch.cat([
                t.view(-1, self.state_dim),
                zgoal.view(-1, self.zgoal_dim),
                zspkr.view(-1, self.zspkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zstate_dim)
        zstate_prior: MultiGaussian = self.zstate_prior(
            torch.cat([
                zgoal.view(-1, self.zgoal_dim),
                zspkr.view(-1, self.zspkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zstate_dim)
        zstate = (zstate_post.sample(sample_scale)
                  .view(batch_size, max_conv_len, -1)
                  .masked_fill(~conv_mask.unsqueeze(-1), 0))
        zsent_post: MultiGaussian = self.zsent_post(
            torch.cat([
                u.view(-1, self.sent_dim),
                zstate.view(-1, self.zstate_dim),
                zgoal.view(-1, self.zgoal_dim),
                zspkr.view(-1, self.zspkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zsent_dim)
        zsent_prior: MultiGaussian = self.zsent_prior(
            torch.cat([
                zstate.view(-1, self.zstate_dim),
                zgoal.view(-1, self.zgoal_dim),
                zspkr.view(-1, self.zspkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zsent_dim)
        zsent = (zsent_post.sample(sample_scale)
                 .view(batch_size, max_conv_len, -1)
                 .masked_fill(~conv_mask.unsqueeze(-1), 0))
        p_logit = self.spkr_decoder(
            (torch.cat([zspkr, context, zconv_exp], 2)
             .view(batch_size * max_conv_len, -1))
        ).view(batch_size, max_conv_len, -1)
        g_logit = self.goal_decoder(
            (torch.cat([zgoal, zspkr, context, zconv_exp], 2)
             .view(batch_size * max_conv_len, -1)),
            p.view(batch_size * max_conv_len),
            asv_h
        ).view(batch_size, max_conv_len, -1)
        s_logit = self.state_decoder(
            (torch.cat([zstate, zgoal, zspkr, context, zconv_exp], 2)
             .view(batch_size * max_conv_len, -1)),
            p.view(batch_size * max_conv_len),
            asv_h
        ).view(batch_size, max_conv_len, -1)
        w_logit = self.sent_decoder(
            (torch.cat([zsent, zstate, zgoal, zspkr, context, zconv_exp], 2)
             .view(batch_size * max_conv_len, -1)),
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1)
        ).view(batch_size, max_conv_len, max_sent_len, -1)
        return (
            {  # logits
                "sent": w_logit,
                "speaker": p_logit,
                "goal": g_logit,
                "state": s_logit
            }, {  # posterior
                "conv": zconv_post,
                "sent": zsent_post,
                "speaker": zspkr_post,
                "goal": zgoal_post,
                "state": zstate_post,
            }, {  # prior
                "conv": (MultiGaussian.unit(*zconv_post.size())
                         .to(w_logit.device)),
                "sent": zsent_prior,
                "speaker": zspkr_prior,
                "goal": zgoal_prior,
                "state": zstate_prior
            }
        )

    def _encode_impl(self, data, *args, **kwargs):
        conv_lens = data["conv_lens"]
        w, w_lens = data["sent"], data["sent_lens"]
        batch_size, max_conv_len, max_sent_len = w.size()
        conv_mask = utils.mask(conv_lens)
        w_emb = self.word_encoder(w, w_lens)
        u = self.sent_post_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
            self.post_query.unsqueeze(0).expand(batch_size * max_conv_len, -1)
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = u.masked_fill(~conv_mask.unsqueeze(-1), 0)
        _, c = self.conv_encoder(
            u,
            conv_lens,
            self.post_query.unsqueeze(0).expand(batch_size, -1),
        )
        return self.zconv_post(c)

    def _decode_optimal_impl(self, data, asv_h=None,
                             conv_state=None,
                             init_u=None, init_v=None, init_t=None, init_r=None,
                             max_conv_len=20, max_sent_len=30,
                             spkr_scale=1.0, goal_scale=1.0,
                             state_scale=1.0, sent_scale=1.0,
                             beam_size=8, **kwargs):
        def pred_prob(prob, threshold=0.5):
            """Computes the prediction prob, given the threshold."""
            prob = prob.clone()
            mask = prob > 0.5
            prob[mask] = 1 - prob[mask]
            return prob

        def state_sparse(mask):
            # pad empty state with asv_pad at the last of the sequence
            mask = mask.clone()
            mask[:, asv_pad_idx] = 0  # turn off asv_pad first
            state = utils.to_sparse(mask)
            value, lens = state.value, state.lens
            value = torch.cat([value, value.new(batch_size, 1).zero_()], 1)
            value[torch.arange(batch_size), lens] = asv_pad_idx
            return Stacked1DTensor(
                value=value,
                lens=lens + 1
            )

        zconv, asv, asv_lens = data["zconv"], data["asv"], data["asv_lens"]
        if asv_h is None:
            asv_emb = self.word_encoder(asv, asv_lens)
            _, asv_h = self.asv_encoder(asv_emb, asv_lens)
        batch_size = zconv.size(0)
        eoc_idx = self.vocabs.word["<eoc>"]
        asv_pad = ActSlotValue("<pad>", "<pad>", "<pad>")
        asv_pad_idx = self.vocabs.goal_state.asv[asv_pad]
        sents, spkrs, goals, states = list(), list(), list(), list()
        conv_lens = zconv.new(batch_size).long().zero_()
        conv_done = zconv.new(batch_size).bool().zero_()
        spkr_logprob = zconv.new(batch_size).zero_()
        sent_logprob = zconv.new(batch_size).zero_()
        goal_logprob = zconv.new(batch_size).zero_()
        state_logprob = zconv.new(batch_size).zero_()
        if init_u is None:
            u = zconv.new(batch_size, self.sent_dim).zero_()
        else:
            u = init_u
        if init_v is None:
            v = zconv.new(batch_size, self.goal_dim).zero_()
        else:
            v = init_v
        if init_t is None:
            t = zconv.new(batch_size, self.state_dim).zero_()
        else:
            t = init_t
        if init_r is None:
            r = zconv.new(batch_size, self.spkr_dim).zero_()
        else:
            r = init_r
        if conv_state is None:
            conv_state = self.ctx_encoder.init_state(batch_size)
        for i in range(max_conv_len):
            context, _, conv_state = self.ctx_encoder(
                torch.cat([u, v, t, r, zconv], 1).unsqueeze(1),
                lens=u.new(batch_size).long().fill_(1),
                h=conv_state
            )
            context = context.view(-1, self.ctx_dim)
            zspkr = self.zspkr_prior(
                torch.cat([context, zconv], 1)
            ).sample(spkr_scale)
            zgoal = self.zgoal_prior(
                torch.cat([zspkr, context, zconv], 1)
            ).sample(goal_scale)
            zstate = self.zstate_prior(
                torch.cat([zgoal, zspkr, context, zconv], 1)
            ).sample(state_scale)
            zutt = self.zsent_prior(
                torch.cat([zstate, zgoal, zspkr, context, zconv], 1)
            ).sample(sent_scale)
            p_logit = self.spkr_decoder(
                torch.cat([zspkr, context, zconv], 1)
            )
            p_prob, p = torch.softmax(p_logit, -1).max(-1)
            p_logprob = p_prob.log().sum(-1)
            r = self.spkr_encoder(self.spkr_eye[p])
            g_logit = self.goal_decoder(
                torch.cat([zgoal, zspkr, context, zconv], 1),
                p, asv_h
            )
            g_prob = utils.sigmoid_inf(g_logit)
            g = state_sparse(g_prob > 0.5)
            g_logprob = pred_prob(g_prob).log().sum(-1)
            v = self.goal_encoder(asv_h[g.value], g.lens, zconv)[1]
            s_logit = self.state_decoder(
                torch.cat([zstate, zgoal, zspkr, context, zconv], 1),
                p, asv_h
            )
            s_prob = utils.sigmoid_inf(s_logit)
            s = state_sparse(s_prob > 0.5)
            s_logprob = pred_prob(s_prob).log().sum(-1)
            t = self.state_encoder(asv_h[s.value], s.lens, zconv)[1]
            w, w_lens, w_prob = self.sent_decoder.generate(
                h=torch.cat([
                    zutt,
                    zstate,
                    zgoal,
                    zspkr,
                    context.view(batch_size, self.ctx_dim),
                    zconv
                ], 1),
                beam_size=beam_size,
                max_len=max_sent_len
            )
            w_logprob = w_prob.log()
            sents.append(Stacked1DTensor(w, w_lens))
            _, u = self.sent_encoder(self.word_encoder(w, w_lens),
                                     w_lens, zconv)
            spkrs.append(p), goals.append(g), states.append(s)
            conv_lens += 1 - conv_done.long()
            done = ((w == eoc_idx) & utils.mask(w_lens, w.size(1))).any(1)
            conv_done |= done
            sent_logprob = w_logprob.masked_fill(conv_done, 0)
            goal_logprob = g_logprob.masked_fill(conv_done, 0)
            state_logprob = s_logprob.masked_fill(conv_done, 0)
            spkr_logprob = p_logprob.masked_fill(conv_done, 0)
            if conv_done.all().item():
                break
        sents = utils.stack_stacked1dtensors(sents)
        goals = utils.stack_stacked1dtensors(goals)
        states = utils.stack_stacked1dtensors(states)
        return BatchData(
            sent=DoublyStacked1DTensor(
                value=sents.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=sents.lens1.t().contiguous()
            ),
            speaker=Stacked1DTensor(
                value=torch.stack(spkrs).transpose(1, 0).contiguous(),
                lens=conv_lens
            ),
            goal=DoublyStacked1DTensor(
                value=goals.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=goals.lens1.t().contiguous()
            ),
            state=DoublyStacked1DTensor(
                value=states.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=states.lens1.t().contiguous()
            ),
        ), dict(
            logprob=sum((sent_logprob, goal_logprob,
                         state_logprob, spkr_logprob)),
            sent_logprob=sent_logprob,
            goal_logprob=goal_logprob,
            state_logprob=state_logprob,
            spkr_logprob=spkr_logprob,
        )

    def _genconv_prior_impl(self, data, max_conv_len=20,
                            beam_size=4, max_sent_len=30,
                            conv_scale=1.0, spkr_scale=1.0,
                            goal_scale=1.0, state_scale=1.0, sent_scale=1.0):
        n = data["n"]
        zconv = MultiGaussian.unit(n.item(), self.ctx_dim).to(n.device)
        zconv_sample = zconv.sample(conv_scale)
        batch, info = self._decode_optimal_impl(
            data={"zconv": zconv_sample, "asv": data["asv"],
                  "asv_lens": data["asv_lens"]},
            eoc="<eoc>",
            max_conv_len=max_conv_len,
            spkr_scale=spkr_scale,
            goal_scale=goal_scale,
            state_scale=state_scale,
            sent_scale=sent_scale,
            beam_size=beam_size,
            max_sent_len=max_sent_len
        )
        info["zconv"] = zconv
        info["zconv-sample"] = zconv_sample
        return batch, info

    def _genconv_post_impl(self, data, max_conv_len=20,
                           beam_size=4, max_sent_len=30,
                           conv_scale=1.0, spkr_scale=1.0,
                           goal_scale=1.0, state_scale=1.0, sent_scale=1.0):
        zconv = self._encode_impl(data)
        zconv_sample = zconv.sample(conv_scale)
        batch, info = self._decode_optimal_impl(
            data={"zconv": zconv_sample, "asv": data["asv"],
                  "asv_lens": data["asv_lens"]},
            eoc="<eoc>",
            max_conv_len=max_conv_len,
            spkr_scale=spkr_scale,
            goal_scale=goal_scale,
            state_scale=state_scale,
            sent_scale=sent_scale,
            beam_size=beam_size,
            max_sent_len=max_sent_len
        )
        info["zconv"] = zconv
        info["zconv-sample"] = zconv_sample
        return batch, info


class HDA(AbstractTDA):
    name = "hda"

    def __init__(self, *args,
                 conv_dim=256, word_dim=100, sent_dim=128, asv_dim=64,
                 state_dim=32, goal_dim=32, spkr_dim=8, ctx_dim=256,
                 spkr_dropout=0.0, asv_dropout=0.0, goal_dropout=0.0,
                 state_dropout=0.0, sent_dropout=0.0, word_dropout=0.0,
                 word_encoder=AbstractWordEncoder,
                 seq_encoder=AbstractSequenceEncoder,
                 sent_decoder=AbstractSentDecoder,
                 state_decoder=AbstractStateDecoder,
                 ctx_encoder=AbstractContextEncoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_dim = word_dim
        self.sent_dim = sent_dim
        self.asv_dim = asv_dim
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.spkr_dim = spkr_dim
        self.conv_dim = conv_dim
        self.ctx_dim = ctx_dim
        self.word_encoder_cls = word_encoder
        self.seq_encoder_cls = seq_encoder
        self.state_decoder_cls = state_decoder
        self.sent_decoder_cls = sent_decoder
        self.ctx_encoder_cls = ctx_encoder
        self.asv_dropout = asv_dropout
        self.spkr_dropout = spkr_dropout
        self.goal_dropout = goal_dropout
        self.state_dropout = state_dropout
        self.sent_dropout = sent_dropout
        self.word_dropout = word_dropout

        self.word_encoder = self.word_encoder_cls(
            vocab=self.vocabs.word,
            word_dim=self.word_dim
        )
        self.asv_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=0,
            hidden_dim=self.asv_dim
        )
        self.post_query = nn.Parameter(torch.randn(self.sent_dim))
        self.sent_post_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=self.sent_dim,
            hidden_dim=self.sent_dim
        )
        self.conv_encoder = self.seq_encoder_cls(
            input_dim=self.sent_dim,
            query_dim=self.sent_dim,
            hidden_dim=self.conv_dim
        )
        self.ctx_encoder = self.ctx_encoder_cls(
            input_dim=(self.conv_dim + self.spkr_dim +
                       self.goal_dim + self.state_dim + self.sent_dim),
            ctx_dim=self.ctx_dim
        )
        self.spkr_encoder = torchmodels.Linear(
            in_features=len(self.vocabs.speaker),
            out_features=self.spkr_dim
        )
        self.spkr_decoder = torchmodels.Linear(
            # conv_rnn + z_conv + z_speaker
            in_features=self.ctx_dim + self.conv_dim + self.spkr_dim,
            out_features=len(self.vocabs.speaker)
        )
        self.goal_encoder = self.seq_encoder_cls(
            input_dim=self.asv_dim,
            query_dim=self.conv_dim,
            hidden_dim=self.goal_dim
        )
        self.goal_decoder = self.state_decoder_cls(
            vocabs=self.vocabs, state_type="goal",
            # conv_rnn + z_conv + z_speaker + z_goal
            input_dim=(self.ctx_dim + self.conv_dim +
                       self.spkr_dim + self.goal_dim),
            asv_dim=self.asv_dim
        )
        self.state_encoder = self.seq_encoder_cls(
            input_dim=self.asv_dim,
            query_dim=self.conv_dim,
            hidden_dim=self.state_dim
        )
        self.state_decoder = self.state_decoder_cls(
            vocabs=self.vocabs, state_type="state",
            # conv_rnn + z_conv + z_speaker + z_goal + z_turn
            input_dim=(self.ctx_dim + self.conv_dim +
                       self.spkr_dim + self.goal_dim + self.state_dim),
            asv_dim=self.asv_dim
        )
        self.sent_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=self.conv_dim,
            hidden_dim=self.sent_dim
        )
        self.sent_decoder = self.sent_decoder_cls(
            vocab=self.vocabs.word,
            word_encoder=self.word_encoder,
            # conv_rnn + z_conv + z_sp + z_goal + z_turn + z_utt
            hidden_dim=(self.ctx_dim + self.conv_dim + self.spkr_dim +
                        self.goal_dim + self.state_dim + self.sent_dim)
        )
        self.spkr_unk = nn.Parameter(torch.zeros(self.spkr_dim))
        self.goal_unk = nn.Parameter(torch.zeros(self.goal_dim))
        self.state_unk = nn.Parameter(torch.zeros(self.state_dim))
        self.sent_unk = nn.Parameter(torch.zeros(self.sent_dim))
        self.word_unk = nn.Parameter(torch.zeros(self.word_dim))
        self.asv_unk = nn.Parameter(torch.zeros(self.asv_dim))
        self.spkr_eye = nn.Parameter(torch.eye(self.num_speakers),
                                     requires_grad=False)

    @property
    def num_speakers(self):
        return self.vocabs.num_speakers

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.spkr_unk, 0, 1 / math.sqrt(self.spkr_dim))
        nn.init.normal_(self.goal_unk, 0, 1 / math.sqrt(self.goal_dim))
        nn.init.normal_(self.state_unk, 0, 1 / math.sqrt(self.state_dim))
        nn.init.normal_(self.sent_unk, 0, 1 / math.sqrt(self.sent_dim))
        nn.init.normal_(self.word_unk, 0, 1 / math.sqrt(self.word_dim))
        nn.init.normal_(self.asv_unk, 0, 1 / math.sqrt(self.asv_dim))
        nn.init.normal_(self.post_query, 0, 1 / math.sqrt(len(self.post_query)))

    def dropout(self, x, dropout_rate, unk):
        if not self.training:
            return x
        x_size = x.size()
        x = x.view(-1, x.size(-1))
        idx = dist.Bernoulli(dropout_rate).sample((x.size(0),)).bool()
        if idx.any():
            x[idx] = unk
        return x.view(*x_size)

    def _inference_impl(self, data, sample_scale=1.0,
                        dropout_scale=1.0, **kwargs):
        conv_lens, w, w_lens, p, g, g_lens, s, s_lens, asv, asv_lens = \
            self.tuplize_data(data)
        batch_size, max_conv_len, max_sent_len = w.size()
        max_goal_len, max_state_len = g.size(-1), s.size(-1)
        num_asv, max_asv_len = asv.size()
        conv_mask = utils.mask(conv_lens)
        w_mask = utils.mask(w_lens) & conv_mask.unsqueeze(-1)
        g_mask = utils.mask(g_lens) & conv_mask.unsqueeze(-1)
        s_mask = utils.mask(s_lens) & conv_mask.unsqueeze(-1)
        w_emb = self.word_encoder(w, w_lens)
        p_emb = self.spkr_eye[p]
        asv_emb = self.word_encoder(asv, asv_lens)
        _, asv_h = self.asv_encoder(asv_emb, asv_lens)

        u = self.sent_post_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
            self.post_query.unsqueeze(0).expand(batch_size * max_conv_len, -1)
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = u.masked_fill(~conv_mask.unsqueeze(-1), 0)
        _, c = self.conv_encoder(
            u,
            conv_lens,
            self.post_query.unsqueeze(0).expand(batch_size, -1),
        )
        c_exp = c.unsqueeze(1).expand(-1, max_conv_len, -1)
        c_exp_flat = c_exp.reshape(-1, self.conv_dim)

        w_emb = self.dropout(w_emb,
                             dropout_rate=self.word_dropout * dropout_scale,
                             unk=self.word_unk)
        u = self.sent_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
            c_exp_flat
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        g_asv = self.dropout(
            asv_h[g.masked_fill(~g_mask, 0)],
            dropout_rate=self.asv_dropout * dropout_scale,
            unk=self.asv_unk
        )
        v = self.goal_encoder(
            g_asv.view(-1, max_goal_len, self.asv_dim),
            g_lens.view(-1),
            c_exp_flat,
        )[1].view(batch_size, max_conv_len, self.goal_dim)
        s_asv = self.dropout(
            asv_h[s.masked_fill(~s_mask, 0)],
            dropout_rate=self.asv_dropout * dropout_scale,
            unk=self.asv_unk
        )
        t = self.state_encoder(
            s_asv.view(-1, max_state_len, self.asv_dim),
            s_lens.view(-1),
            c_exp_flat
        )[1].view(batch_size, max_conv_len, self.state_dim)
        u = self.dropout(u, self.sent_dropout * dropout_scale, self.sent_unk)
        v = self.dropout(v, self.goal_dropout * dropout_scale, self.goal_unk)
        t = self.dropout(t, self.state_dropout * dropout_scale, self.state_unk)
        r = self.dropout(self.spkr_encoder(p_emb),
                         dropout_rate=self.spkr_dropout * dropout_scale,
                         unk=self.spkr_unk)
        context, _, _ = self.ctx_encoder(
            torch.cat([
                utils.shift(u, dim=1),
                utils.shift(v, dim=1),
                utils.shift(t, dim=1),
                utils.shift(r, dim=1),
                c.unsqueeze(1).expand(-1, max_conv_len, -1)
            ], 2),
            conv_lens
        )
        p_logit = self.spkr_decoder(
            (torch.cat([r, context, c_exp], 2)
             .view(batch_size * max_conv_len, -1))
        ).view(batch_size, max_conv_len, -1)
        g_logit = self.goal_decoder(
            (torch.cat([v, r, context, c_exp], 2)
             .view(batch_size * max_conv_len, -1)),
            p.view(batch_size * max_conv_len),
            asv_h
        ).view(batch_size, max_conv_len, -1)
        s_logit = self.state_decoder(
            (torch.cat([t, v, r, context, c_exp], 2)
             .view(batch_size * max_conv_len, -1)),
            p.view(batch_size * max_conv_len),
            asv_h
        ).view(batch_size, max_conv_len, -1)
        w_logit = self.sent_decoder(
            (torch.cat([u, t, v, r, context, c_exp], 2)
             .view(batch_size * max_conv_len, -1)),
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1)
        ).view(batch_size, max_conv_len, max_sent_len, -1)
        conv_dist = MultiGaussian.unit(*c.size()).to(w.device)
        conv_dist.mu = c
        return (
            {  # logits
                "sent": w_logit,
                "speaker": p_logit,
                "goal": g_logit,
                "state": s_logit
            }, {  # posterior
                "conv": conv_dist,
                "sent": MultiGaussian.unit(batch_size, max_conv_len, 1).to(
                    w.device),
                "speaker": MultiGaussian.unit(batch_size, max_conv_len, 1).to(
                    w.device),
                "goal": MultiGaussian.unit(batch_size, max_conv_len, 1).to(
                    w.device),
                "state": MultiGaussian.unit(batch_size, max_conv_len, 1).to(
                    w.device),
            }, {  # prior
                "conv": MultiGaussian.unit(*conv_dist.size()).to(c.device),
                "sent": MultiGaussian.unit(batch_size, max_conv_len, 1).to(
                    w.device),
                "speaker": MultiGaussian.unit(batch_size, max_conv_len, 1).to(
                    w.device),
                "goal": MultiGaussian.unit(batch_size, max_conv_len, 1).to(
                    w.device),
                "state": MultiGaussian.unit(batch_size, max_conv_len, 1).to(
                    w.device)
            }
        )

    def _encode_impl(self, data, *args, **kwargs):
        conv_lens = data["conv_lens"]
        w, w_lens = data["sent"], data["sent_lens"]
        batch_size, max_conv_len, max_sent_len = w.size()
        conv_mask = utils.mask(conv_lens)
        w_emb = self.word_encoder(w, w_lens)
        u = self.sent_post_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
            self.post_query.unsqueeze(0).expand(batch_size * max_conv_len, -1)
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = u.masked_fill(~conv_mask.unsqueeze(-1), 0)
        _, c = self.conv_encoder(
            u,
            conv_lens,
            self.post_query.unsqueeze(0).expand(batch_size, -1),
        )
        return MultiGaussian(c, c.clone().detach().fill_(float("-inf")))

    def _decode_optimal_impl(self, data, asv_h=None,
                             conv_state=None,
                             init_u=None, init_v=None, init_t=None, init_r=None,
                             max_conv_len=20, max_sent_len=30,
                             spkr_scale=1.0, goal_scale=1.0,
                             state_scale=1.0, sent_scale=1.0,
                             beam_size=8, **kwargs):
        def pred_prob(prob, threshold=0.5):
            """Computes the prediction prob, given the threshold."""
            prob = prob.clone()
            mask = prob > 0.5
            prob[mask] = 1 - prob[mask]
            return prob

        def state_sparse(mask):
            # pad empty state with asv_pad at the last of the sequence
            mask = mask.clone()
            mask[:, asv_pad_idx] = 0  # turn off asv_pad first
            state = utils.to_sparse(mask)
            value, lens = state.value, state.lens
            value = torch.cat([value, value.new(batch_size, 1).zero_()], 1)
            value[torch.arange(batch_size), lens] = asv_pad_idx
            return Stacked1DTensor(
                value=value,
                lens=lens + 1
            )

        c, asv, asv_lens = data["zconv"], data["asv"], data["asv_lens"]
        if asv_h is None:
            asv_emb = self.word_encoder(asv, asv_lens)
            _, asv_h = self.asv_encoder(asv_emb, asv_lens)
        batch_size = c.size(0)
        eoc_idx = self.vocabs.word["<eoc>"]
        asv_pad = ActSlotValue("<pad>", "<pad>", "<pad>")
        asv_pad_idx = self.vocabs.goal_state.asv[asv_pad]
        sents, spkrs, goals, states = list(), list(), list(), list()
        conv_lens = c.new(batch_size).long().zero_()
        conv_done = c.new(batch_size).bool().zero_()
        spkr_logprob = c.new(batch_size).zero_()
        sent_logprob = c.new(batch_size).zero_()
        goal_logprob = c.new(batch_size).zero_()
        state_logprob = c.new(batch_size).zero_()
        if init_u is None:
            u = c.new(batch_size, self.sent_dim).zero_()
        else:
            u = init_u
        if init_v is None:
            v = c.new(batch_size, self.goal_dim).zero_()
        else:
            v = init_v
        if init_t is None:
            t = c.new(batch_size, self.state_dim).zero_()
        else:
            t = init_t
        if init_r is None:
            r = c.new(batch_size, self.spkr_dim).zero_()
        else:
            r = init_r
        if conv_state is None:
            conv_state = self.ctx_encoder.init_state(batch_size)
        for i in range(max_conv_len):
            context, _, conv_state = self.ctx_encoder(
                torch.cat([u, v, t, r, c], 1).unsqueeze(1),
                lens=u.new(batch_size).long().fill_(1),
                h=conv_state
            )
            context = context.view(-1, self.ctx_dim)
            r = self._perturb(r, max(spkr_scale - 1.0, 0.0))
            v = self._perturb(v, max(goal_scale - 1.0, 0.0))
            t = self._perturb(t, max(state_scale - 1.0, 0.0))
            u = self._perturb(u, max(sent_scale - 1.0, 0.0))
            p_logit = self.spkr_decoder(
                torch.cat([r, context, c], 1)
            )
            p_prob, p = torch.softmax(p_logit, -1).max(-1)
            p_logprob = p_prob.log().sum(-1)
            r = self.spkr_encoder(self.spkr_eye[p])
            g_logit = self.goal_decoder(
                torch.cat([v, r, context, c], 1),
                p, asv_h
            )
            g_prob = utils.sigmoid_inf(g_logit)
            g = state_sparse(g_prob > 0.5)
            g_logprob = pred_prob(g_prob).log().sum(-1)
            v = self.goal_encoder(asv_h[g.value], g.lens, c)[1]
            s_logit = self.state_decoder(
                torch.cat([t, v, r, context, c], 1),
                p, asv_h
            )
            s_prob = utils.sigmoid_inf(s_logit)
            s = state_sparse(s_prob > 0.5)
            s_logprob = pred_prob(s_prob).log().sum(-1)
            t = self.state_encoder(asv_h[s.value], s.lens, c)[1]
            w, w_lens, w_prob = self.sent_decoder.generate(
                h=torch.cat([
                    u,
                    t,
                    v,
                    r,
                    context.view(batch_size, self.ctx_dim),
                    c
                ], 1),
                beam_size=beam_size,
                max_len=max_sent_len
            )
            w_logprob = w_prob.log()
            sents.append(Stacked1DTensor(w, w_lens))
            _, u = self.sent_encoder(self.word_encoder(w, w_lens),
                                     w_lens, c)
            spkrs.append(p), goals.append(g), states.append(s)
            conv_lens += 1 - conv_done.long()
            done = ((w == eoc_idx) & utils.mask(w_lens, w.size(1))).any(1)
            conv_done |= done
            sent_logprob = w_logprob.masked_fill(conv_done, 0)
            goal_logprob = g_logprob.masked_fill(conv_done, 0)
            state_logprob = s_logprob.masked_fill(conv_done, 0)
            spkr_logprob = p_logprob.masked_fill(conv_done, 0)
            if conv_done.all().item():
                break
        sents = utils.stack_stacked1dtensors(sents)
        goals = utils.stack_stacked1dtensors(goals)
        states = utils.stack_stacked1dtensors(states)
        return BatchData(
            sent=DoublyStacked1DTensor(
                value=sents.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=sents.lens1.t().contiguous()
            ),
            speaker=Stacked1DTensor(
                value=torch.stack(spkrs).transpose(1, 0).contiguous(),
                lens=conv_lens
            ),
            goal=DoublyStacked1DTensor(
                value=goals.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=goals.lens1.t().contiguous()
            ),
            state=DoublyStacked1DTensor(
                value=states.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=states.lens1.t().contiguous()
            ),
        ), dict(
            logprob=sum((sent_logprob, goal_logprob,
                         state_logprob, spkr_logprob)),
            sent_logprob=sent_logprob,
            goal_logprob=goal_logprob,
            state_logprob=state_logprob,
            spkr_logprob=spkr_logprob,
        )

    def _genconv_prior_impl(self, data, max_conv_len=20,
                            beam_size=4, max_sent_len=30,
                            conv_scale=1.0, spkr_scale=1.0,
                            goal_scale=1.0, state_scale=1.0, sent_scale=1.0):
        raise NotImplementedError

    @staticmethod
    def _perturb(x, eps):
        return x + x.clone().uniform_(-eps, eps)

    def _genconv_post_impl(self, data, max_conv_len=20,
                           beam_size=4, max_sent_len=30,
                           conv_scale=1.0, spkr_scale=1.0,
                           goal_scale=1.0, state_scale=1.0, sent_scale=1.0):
        c = self._encode_impl(data).mu
        c_sample = self._perturb(c, max(conv_scale - 1.0, 0.0))
        batch, info = self._decode_optimal_impl(
            data={"zconv": c, "asv": data["asv"],
                  "asv_lens": data["asv_lens"]},
            eoc="<eoc>",
            max_conv_len=max_conv_len,
            spkr_scale=spkr_scale,
            goal_scale=goal_scale,
            state_scale=state_scale,
            sent_scale=sent_scale,
            beam_size=beam_size,
            max_sent_len=max_sent_len
        )
        conv_dist = MultiGaussian.unit(*c.size()).to(c.device)
        conv_dist.mu = c
        info["zconv"] = conv_dist
        info["zconv-sample"] = c_sample
        return batch, info


class VHDAWithoutGoal(AbstractTDA):
    name = "vhda-nogoal"

    def __init__(self, *args,
                 conv_dim=256, word_dim=100, sent_dim=128, asv_dim=64,
                 state_dim=32, spkr_dim=8, ctx_dim=256,
                 zsent_dim=16, zstate_dim=8,
                 zspkr_dim=2, zconv_dim=16,
                 spkr_dropout=0.0, asv_dropout=0.0,
                 state_dropout=0.0, sent_dropout=0.0, word_dropout=0.0,
                 word_encoder=AbstractWordEncoder,
                 seq_encoder=AbstractSequenceEncoder,
                 sent_decoder=AbstractSentDecoder,
                 state_decoder=AbstractStateDecoder,
                 ctx_encoder=AbstractContextEncoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_dim = word_dim
        self.sent_dim = sent_dim
        self.asv_dim = asv_dim
        self.state_dim = state_dim
        self.spkr_dim = spkr_dim
        self.conv_dim = conv_dim
        self.ctx_dim = ctx_dim
        self.zconv_dim = zconv_dim
        self.zspkr_dim = zspkr_dim
        self.zstate_dim = zstate_dim
        self.zsent_dim = zsent_dim
        self.word_encoder_cls = word_encoder
        self.seq_encoder_cls = seq_encoder
        self.state_decoder_cls = state_decoder
        self.sent_decoder_cls = sent_decoder
        self.ctx_encoder_cls = ctx_encoder
        self.asv_dropout = asv_dropout
        self.spkr_dropout = spkr_dropout
        self.state_dropout = state_dropout
        self.sent_dropout = sent_dropout
        self.word_dropout = word_dropout

        self.word_encoder = self.word_encoder_cls(
            vocab=self.vocabs.word,
            word_dim=self.word_dim
        )
        self.asv_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=0,
            hidden_dim=self.asv_dim
        )
        self.post_query = nn.Parameter(torch.randn(self.sent_dim))
        self.sent_post_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=self.sent_dim,
            hidden_dim=self.sent_dim
        )
        self.conv_encoder = self.seq_encoder_cls(
            input_dim=self.sent_dim,
            query_dim=self.sent_dim,
            hidden_dim=self.conv_dim
        )
        self.ctx_encoder = self.ctx_encoder_cls(
            input_dim=(self.zconv_dim + self.spkr_dim +
                       self.state_dim + self.sent_dim),
            ctx_dim=self.ctx_dim
        )
        self.spkr_encoder = torchmodels.Linear(
            in_features=len(self.vocabs.speaker),
            out_features=self.spkr_dim
        )
        self.spkr_decoder = torchmodels.Linear(
            # conv_rnn + z_conv + z_speaker
            in_features=self.ctx_dim + self.zconv_dim + self.zspkr_dim,
            out_features=len(self.vocabs.speaker)
        )
        self.state_encoder = self.seq_encoder_cls(
            input_dim=self.asv_dim,
            query_dim=self.zconv_dim,
            hidden_dim=self.state_dim
        )
        self.state_decoder = self.state_decoder_cls(
            vocabs=self.vocabs, state_type="state",
            # conv_rnn + z_conv + z_speaker + z_turn
            input_dim=(self.ctx_dim + self.zconv_dim +
                       self.zspkr_dim + self.zstate_dim),
            asv_dim=self.asv_dim
        )
        self.sent_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=self.zconv_dim,
            hidden_dim=self.sent_dim
        )
        self.sent_decoder = self.sent_decoder_cls(
            vocab=self.vocabs.word,
            word_encoder=self.word_encoder,
            # conv_rnn + z_conv + z_sp + z_turn + z_utt
            hidden_dim=(self.ctx_dim + self.zconv_dim + self.zspkr_dim +
                        self.zstate_dim + self.zsent_dim)
        )
        self.zconv_post = MultiGaussianLayer(
            input_dim=self.conv_dim,
            hidden_dim=self.zconv_dim
        )
        self.zspkr_prior = MultiGaussianLayer(
            # conv_rnn + z_conv
            input_dim=self.ctx_dim + self.zconv_dim,
            hidden_dim=self.zspkr_dim
        )
        self.zspkr_post = MultiGaussianLayer(
            # conv_rnn + z_conv + h_speaker
            input_dim=self.ctx_dim + self.zconv_dim + self.spkr_dim,
            hidden_dim=self.zspkr_dim
        )
        self.zstate_prior = MultiGaussianLayer(
            # conv_rnn + z_conv + z_speaker
            input_dim=(self.ctx_dim + self.zconv_dim +
                       self.zspkr_dim),
            hidden_dim=self.zstate_dim
        )
        self.zstate_post = MultiGaussianLayer(
            # conv_rnn + z_conv + z_speaker + h_turn
            input_dim=(self.ctx_dim + self.zconv_dim +
                       self.zspkr_dim + self.state_dim),
            hidden_dim=self.zstate_dim
        )
        self.zsent_prior = MultiGaussianLayer(
            # conv_rnn + z_conv + z_speaker + z_turn
            input_dim=(self.ctx_dim + self.zconv_dim + self.zspkr_dim +
                       self.zstate_dim),
            hidden_dim=self.zsent_dim
        )
        self.zsent_post = MultiGaussianLayer(
            # conv_rnn + z_conv + z_sp + z_turn + h_utt
            input_dim=(self.ctx_dim + self.zconv_dim + self.zspkr_dim +
                       self.zstate_dim + self.sent_dim),
            hidden_dim=self.zsent_dim
        )
        self.spkr_unk = nn.Parameter(torch.zeros(self.spkr_dim))
        self.state_unk = nn.Parameter(torch.zeros(self.state_dim))
        self.sent_unk = nn.Parameter(torch.zeros(self.sent_dim))
        self.word_unk = nn.Parameter(torch.zeros(self.word_dim))
        self.asv_unk = nn.Parameter(torch.zeros(self.asv_dim))
        self.spkr_eye = nn.Parameter(torch.eye(self.num_speakers),
                                     requires_grad=False)

    @property
    def num_speakers(self):
        return self.vocabs.num_speakers

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.spkr_unk, 0, 1 / math.sqrt(self.spkr_dim))
        nn.init.normal_(self.state_unk, 0, 1 / math.sqrt(self.state_dim))
        nn.init.normal_(self.sent_unk, 0, 1 / math.sqrt(self.sent_dim))
        nn.init.normal_(self.word_unk, 0, 1 / math.sqrt(self.word_dim))
        nn.init.normal_(self.asv_unk, 0, 1 / math.sqrt(self.asv_dim))
        nn.init.normal_(self.post_query, 0, 1 / math.sqrt(len(self.post_query)))

    def dropout(self, x, dropout_rate, unk):
        if not self.training:
            return x
        x_size = x.size()
        x = x.view(-1, x.size(-1))
        idx = dist.Bernoulli(dropout_rate).sample((x.size(0),)).bool()
        if idx.any():
            x[idx] = unk
        return x.view(*x_size)

    def _inference_impl(self, data, sample_scale=1.0,
                        dropout_scale=1.0, **kwargs):
        conv_lens, w, w_lens, p, g, g_lens, s, s_lens, asv, asv_lens = \
            self.tuplize_data(data)
        batch_size, max_conv_len, max_sent_len = w.size()
        max_state_len = s.size(-1)
        num_asv, max_asv_len = asv.size()
        conv_mask = utils.mask(conv_lens)
        w_mask = utils.mask(w_lens) & conv_mask.unsqueeze(-1)
        g_mask = utils.mask(g_lens) & conv_mask.unsqueeze(-1)
        s_mask = utils.mask(s_lens) & conv_mask.unsqueeze(-1)
        w_emb = self.word_encoder(w, w_lens)
        p_emb = self.spkr_eye[p]
        asv_emb = self.word_encoder(asv, asv_lens)
        _, asv_h = self.asv_encoder(asv_emb, asv_lens)

        u = self.sent_post_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
            self.post_query.unsqueeze(0).expand(batch_size * max_conv_len, -1)
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = u.masked_fill(~conv_mask.unsqueeze(-1), 0)
        _, c = self.conv_encoder(
            u,
            conv_lens,
            self.post_query.unsqueeze(0).expand(batch_size, -1),
        )
        zconv_post: MultiGaussian = self.zconv_post(c)
        zconv = zconv_post.sample(sample_scale)
        zconv_exp = zconv.unsqueeze(1).expand(-1, max_conv_len, -1)
        zconv_exp_flat = zconv_exp.reshape(-1, self.zconv_dim)

        w_emb = self.dropout(w_emb,
                             dropout_rate=self.word_dropout * dropout_scale,
                             unk=self.word_unk)
        u = self.sent_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
            zconv_exp_flat
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        s_asv = self.dropout(
            asv_h[s.masked_fill(~s_mask, 0)],
            dropout_rate=self.asv_dropout * dropout_scale,
            unk=self.asv_unk
        )
        t = self.state_encoder(
            s_asv.view(-1, max_state_len, self.asv_dim),
            s_lens.view(-1),
            zconv_exp_flat
        )[1].view(batch_size, max_conv_len, self.state_dim)
        u = self.dropout(u, self.sent_dropout * dropout_scale, self.sent_unk)
        t = self.dropout(t, self.state_dropout * dropout_scale, self.state_unk)
        r = self.dropout(self.spkr_encoder(p_emb),
                         dropout_rate=self.spkr_dropout * dropout_scale,
                         unk=self.spkr_unk)
        context, _, _ = self.ctx_encoder(
            torch.cat([
                utils.shift(u, dim=1),
                utils.shift(t, dim=1),
                utils.shift(r, dim=1),
                zconv.unsqueeze(1).expand(-1, max_conv_len, -1)
            ], 2),
            conv_lens
        )
        zspkr_post: MultiGaussian = self.zspkr_post(
            torch.cat([
                r.view(-1, self.spkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zspkr_dim)
        zspkr_prior: MultiGaussian = self.zspkr_prior(
            torch.cat([
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zspkr_dim)
        zspkr = (zspkr_post.sample(sample_scale)
                 .view(batch_size, max_conv_len, -1)
                 .masked_fill(~conv_mask.unsqueeze(-1), 0))
        zstate_post: MultiGaussian = self.zstate_post(
            torch.cat([
                t.view(-1, self.state_dim),
                zspkr.view(-1, self.zspkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zstate_dim)
        zstate_prior: MultiGaussian = self.zstate_prior(
            torch.cat([
                zspkr.view(-1, self.zspkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zstate_dim)
        zstate = (zstate_post.sample(sample_scale)
                  .view(batch_size, max_conv_len, -1)
                  .masked_fill(~conv_mask.unsqueeze(-1), 0))
        zsent_post: MultiGaussian = self.zsent_post(
            torch.cat([
                u.view(-1, self.sent_dim),
                zstate.view(-1, self.zstate_dim),
                zspkr.view(-1, self.zspkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zsent_dim)
        zsent_prior: MultiGaussian = self.zsent_prior(
            torch.cat([
                zstate.view(-1, self.zstate_dim),
                zspkr.view(-1, self.zspkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zsent_dim)
        zsent = (zsent_post.sample(sample_scale)
                 .view(batch_size, max_conv_len, -1)
                 .masked_fill(~conv_mask.unsqueeze(-1), 0))
        p_logit = self.spkr_decoder(
            (torch.cat([zspkr, context, zconv_exp], 2)
             .view(batch_size * max_conv_len, -1))
        ).view(batch_size, max_conv_len, -1)
        s_logit = self.state_decoder(
            (torch.cat([zstate, zspkr, context, zconv_exp], 2)
             .view(batch_size * max_conv_len, -1)),
            p.view(batch_size * max_conv_len),
            asv_h
        ).view(batch_size, max_conv_len, -1)
        w_logit = self.sent_decoder(
            (torch.cat([zsent, zstate, zspkr, context, zconv_exp], 2)
             .view(batch_size * max_conv_len, -1)),
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1)
        ).view(batch_size, max_conv_len, max_sent_len, -1)
        return (
            {  # logits
                "sent": w_logit,
                "speaker": p_logit,
                "goal": s_logit.detach().clone(),
                "state": s_logit
            }, {  # posterior
                "conv": zconv_post,
                "sent": zsent_post,
                "speaker": zspkr_post,
                "goal": MultiGaussian.unit(*zstate_post.size()).to(w.device),
                "state": zstate_post,
            }, {  # prior
                "conv": (MultiGaussian.unit(*zconv_post.size())
                         .to(w_logit.device)),
                "sent": zsent_prior,
                "speaker": zspkr_prior,
                "goal": MultiGaussian.unit(*zstate_prior.size()).to(w.device),
                "state": zstate_prior
            }
        )

    def _encode_impl(self, data, *args, **kwargs):
        conv_lens = data["conv_lens"]
        w, w_lens = data["sent"], data["sent_lens"]
        batch_size, max_conv_len, max_sent_len = w.size()
        conv_mask = utils.mask(conv_lens)
        w_emb = self.word_encoder(w, w_lens)
        u = self.sent_post_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
            self.post_query.unsqueeze(0).expand(batch_size * max_conv_len, -1)
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = u.masked_fill(~conv_mask.unsqueeze(-1), 0)
        _, c = self.conv_encoder(
            u,
            conv_lens,
            self.post_query.unsqueeze(0).expand(batch_size, -1),
        )
        return self.zconv_post(c)

    def _decode_optimal_impl(self, data, asv_h=None,
                             conv_state=None,
                             init_u=None, init_v=None, init_t=None, init_r=None,
                             max_conv_len=20, max_sent_len=30,
                             spkr_scale=1.0, goal_scale=1.0,
                             state_scale=1.0, sent_scale=1.0,
                             beam_size=8, **kwargs):
        def pred_prob(prob, threshold=0.5):
            """Computes the prediction prob, given the threshold."""
            prob = prob.clone()
            mask = prob > 0.5
            prob[mask] = 1 - prob[mask]
            return prob

        def state_sparse(mask):
            # pad empty state with asv_pad at the last of the sequence
            mask = mask.clone()
            mask[:, asv_pad_idx] = 0  # turn off asv_pad first
            state = utils.to_sparse(mask)
            value, lens = state.value, state.lens
            value = torch.cat([value, value.new(batch_size, 1).zero_()], 1)
            value[torch.arange(batch_size), lens] = asv_pad_idx
            return Stacked1DTensor(
                value=value,
                lens=lens + 1
            )

        zconv, asv, asv_lens = data["zconv"], data["asv"], data["asv_lens"]
        if asv_h is None:
            asv_emb = self.word_encoder(asv, asv_lens)
            _, asv_h = self.asv_encoder(asv_emb, asv_lens)
        batch_size = zconv.size(0)
        eoc_idx = self.vocabs.word["<eoc>"]
        asv_pad = ActSlotValue("<pad>", "<pad>", "<pad>")
        asv_pad_idx = self.vocabs.goal_state.asv[asv_pad]
        sents, spkrs, states = list(), list(), list()
        conv_lens = zconv.new(batch_size).long().zero_()
        conv_done = zconv.new(batch_size).bool().zero_()
        spkr_logprob = zconv.new(batch_size).zero_()
        sent_logprob = zconv.new(batch_size).zero_()
        state_logprob = zconv.new(batch_size).zero_()
        if init_u is None:
            u = zconv.new(batch_size, self.sent_dim).zero_()
        else:
            u = init_u
        if init_t is None:
            t = zconv.new(batch_size, self.state_dim).zero_()
        else:
            t = init_t
        if init_r is None:
            r = zconv.new(batch_size, self.spkr_dim).zero_()
        else:
            r = init_r
        if conv_state is None:
            conv_state = self.ctx_encoder.init_state(batch_size)
        for i in range(max_conv_len):
            context, _, conv_state = self.ctx_encoder(
                torch.cat([u, t, r, zconv], 1).unsqueeze(1),
                lens=u.new(batch_size).long().fill_(1),
                h=conv_state
            )
            context = context.view(-1, self.ctx_dim)
            zspkr = self.zspkr_prior(
                torch.cat([context, zconv], 1)
            ).sample(spkr_scale)
            zstate = self.zstate_prior(
                torch.cat([zspkr, context, zconv], 1)
            ).sample(state_scale)
            zutt = self.zsent_prior(
                torch.cat([zstate, zspkr, context, zconv], 1)
            ).sample(sent_scale)
            p_logit = self.spkr_decoder(
                torch.cat([zspkr, context, zconv], 1)
            )
            p_prob, p = torch.softmax(p_logit, -1).max(-1)
            p_logprob = p_prob.log().sum(-1)
            r = self.spkr_encoder(self.spkr_eye[p])
            s_logit = self.state_decoder(
                torch.cat([zstate, zspkr, context, zconv], 1),
                p, asv_h
            )
            s_prob = utils.sigmoid_inf(s_logit)
            s = state_sparse(s_prob > 0.5)
            s_logprob = pred_prob(s_prob).log().sum(-1)
            t = self.state_encoder(asv_h[s.value], s.lens, zconv)[1]
            w, w_lens, w_prob = self.sent_decoder.generate(
                h=torch.cat([
                    zutt,
                    zstate,
                    zspkr,
                    context.view(batch_size, self.ctx_dim),
                    zconv
                ], 1),
                beam_size=beam_size,
                max_len=max_sent_len
            )
            w_logprob = w_prob.log()
            sents.append(Stacked1DTensor(w, w_lens))
            _, u = self.sent_encoder(self.word_encoder(w, w_lens),
                                     w_lens, zconv)
            spkrs.append(p), states.append(s)
            conv_lens += 1 - conv_done.long()
            done = ((w == eoc_idx) & utils.mask(w_lens, w.size(1))).any(1)
            conv_done |= done
            sent_logprob = w_logprob.masked_fill(conv_done, 0)
            state_logprob = s_logprob.masked_fill(conv_done, 0)
            spkr_logprob = p_logprob.masked_fill(conv_done, 0)
            if conv_done.all().item():
                break
        sents = utils.stack_stacked1dtensors(sents)
        states = utils.stack_stacked1dtensors(states)
        return BatchData(
            sent=DoublyStacked1DTensor(
                value=sents.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=sents.lens1.t().contiguous()
            ),
            speaker=Stacked1DTensor(
                value=torch.stack(spkrs).transpose(1, 0).contiguous(),
                lens=conv_lens
            ),
            goal=DoublyStacked1DTensor(
                value=zconv.new(batch_size, max_conv_len, 0).zero_().long(),
                lens=conv_lens,
                lens1=zconv.new(batch_size, max_conv_len).zero_().long()
            ),
            state=DoublyStacked1DTensor(
                value=states.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=states.lens1.t().contiguous()
            ),
        ), dict(
            logprob=sum((sent_logprob, state_logprob, spkr_logprob)),
            sent_logprob=sent_logprob,
            goal_logprob=sent_logprob.clone().fill_(float("nan")),
            state_logprob=state_logprob,
            spkr_logprob=spkr_logprob,
        )

    def _genconv_prior_impl(self, data, max_conv_len=20,
                            beam_size=4, max_sent_len=30,
                            conv_scale=1.0, spkr_scale=1.0,
                            goal_scale=1.0, state_scale=1.0, sent_scale=1.0):
        n = data["n"]
        zconv = MultiGaussian.unit(n.item(), self.ctx_dim).to(n.device)
        zconv_sample = zconv.sample(conv_scale)
        batch, info = self._decode_optimal_impl(
            data={"zconv": zconv_sample, "asv": data["asv"],
                  "asv_lens": data["asv_lens"]},
            eoc="<eoc>",
            max_conv_len=max_conv_len,
            spkr_scale=spkr_scale,
            goal_scale=goal_scale,
            state_scale=state_scale,
            sent_scale=sent_scale,
            beam_size=beam_size,
            max_sent_len=max_sent_len
        )
        info["zconv"] = zconv
        info["zconv-sample"] = zconv_sample
        return batch, info

    def _genconv_post_impl(self, data, max_conv_len=20,
                           beam_size=4, max_sent_len=30,
                           conv_scale=1.0, spkr_scale=1.0,
                           goal_scale=1.0, state_scale=1.0, sent_scale=1.0):
        zconv = self._encode_impl(data)
        zconv_sample = zconv.sample(conv_scale)
        batch, info = self._decode_optimal_impl(
            data={"zconv": zconv_sample, "asv": data["asv"],
                  "asv_lens": data["asv_lens"]},
            eoc="<eoc>",
            max_conv_len=max_conv_len,
            spkr_scale=spkr_scale,
            goal_scale=goal_scale,
            state_scale=state_scale,
            sent_scale=sent_scale,
            beam_size=beam_size,
            max_sent_len=max_sent_len
        )
        info["zconv"] = zconv
        info["zconv-sample"] = zconv_sample
        return batch, info


class VHDAWithoutGoalAct(AbstractTDA):
    name = "vhda-nogoalact"

    def __init__(self, *args,
                 conv_dim=256, word_dim=100, sent_dim=128,
                 spkr_dim=8, ctx_dim=256,
                 zsent_dim=16,
                 zspkr_dim=2, zconv_dim=16,
                 spkr_dropout=0.0, sent_dropout=0.0, word_dropout=0.0,
                 word_encoder=AbstractWordEncoder,
                 seq_encoder=AbstractSequenceEncoder,
                 sent_decoder=AbstractSentDecoder,
                 ctx_encoder=AbstractContextEncoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_dim = word_dim
        self.sent_dim = sent_dim
        self.spkr_dim = spkr_dim
        self.conv_dim = conv_dim
        self.ctx_dim = ctx_dim
        self.zconv_dim = zconv_dim
        self.zspkr_dim = zspkr_dim
        self.zsent_dim = zsent_dim
        self.word_encoder_cls = word_encoder
        self.seq_encoder_cls = seq_encoder
        self.sent_decoder_cls = sent_decoder
        self.ctx_encoder_cls = ctx_encoder
        self.spkr_dropout = spkr_dropout
        self.sent_dropout = sent_dropout
        self.word_dropout = word_dropout

        self.word_encoder = self.word_encoder_cls(
            vocab=self.vocabs.word,
            word_dim=self.word_dim
        )
        self.post_query = nn.Parameter(torch.randn(self.sent_dim))
        self.sent_post_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=self.sent_dim,
            hidden_dim=self.sent_dim
        )
        self.conv_encoder = self.seq_encoder_cls(
            input_dim=self.sent_dim,
            query_dim=self.sent_dim,
            hidden_dim=self.conv_dim
        )
        self.ctx_encoder = self.ctx_encoder_cls(
            input_dim=(self.zconv_dim + self.spkr_dim +
                       self.goal_dim + self.state_dim + self.sent_dim),
            ctx_dim=self.ctx_dim
        )
        self.spkr_encoder = torchmodels.Linear(
            in_features=len(self.vocabs.speaker),
            out_features=self.spkr_dim
        )
        self.spkr_decoder = torchmodels.Linear(
            # conv_rnn + z_conv + z_speaker
            in_features=self.ctx_dim + self.zconv_dim + self.zspkr_dim,
            out_features=len(self.vocabs.speaker)
        )
        self.sent_encoder = self.seq_encoder_cls(
            input_dim=self.word_dim,
            query_dim=self.zconv_dim,
            hidden_dim=self.sent_dim
        )
        self.sent_decoder = self.sent_decoder_cls(
            vocab=self.vocabs.word,
            word_encoder=self.word_encoder,
            hidden_dim=(self.ctx_dim + self.zconv_dim + self.zspkr_dim +
                        self.zsent_dim)
        )
        self.zconv_post = MultiGaussianLayer(
            input_dim=self.conv_dim,
            hidden_dim=self.zconv_dim
        )
        self.zspkr_prior = MultiGaussianLayer(
            # conv_rnn + z_conv
            input_dim=self.ctx_dim + self.zconv_dim,
            hidden_dim=self.zspkr_dim
        )
        self.zspkr_post = MultiGaussianLayer(
            # conv_rnn + z_conv + h_speaker
            input_dim=self.ctx_dim + self.zconv_dim + self.spkr_dim,
            hidden_dim=self.zspkr_dim
        )
        self.zsent_prior = MultiGaussianLayer(
            input_dim=(self.ctx_dim + self.zconv_dim + self.zspkr_dim),
            hidden_dim=self.zsent_dim
        )
        self.zsent_post = MultiGaussianLayer(
            input_dim=(self.ctx_dim + self.zconv_dim + self.zspkr_dim +
                       self.sent_dim),
            hidden_dim=self.zsent_dim
        )
        self.spkr_unk = nn.Parameter(torch.zeros(self.spkr_dim))
        self.sent_unk = nn.Parameter(torch.zeros(self.sent_dim))
        self.word_unk = nn.Parameter(torch.zeros(self.word_dim))
        self.spkr_eye = nn.Parameter(torch.eye(self.num_speakers),
                                     requires_grad=False)

    @property
    def num_speakers(self):
        return self.vocabs.num_speakers

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.spkr_unk, 0, 1 / math.sqrt(self.spkr_dim))
        nn.init.normal_(self.sent_unk, 0, 1 / math.sqrt(self.sent_dim))
        nn.init.normal_(self.word_unk, 0, 1 / math.sqrt(self.word_dim))
        nn.init.normal_(self.post_query, 0, 1 / math.sqrt(len(self.post_query)))

    def dropout(self, x, dropout_rate, unk):
        if not self.training:
            return x
        x_size = x.size()
        x = x.view(-1, x.size(-1))
        idx = dist.Bernoulli(dropout_rate).sample((x.size(0),)).bool()
        if idx.any():
            x[idx] = unk
        return x.view(*x_size)

    def _inference_impl(self, data, sample_scale=1.0,
                        dropout_scale=1.0, **kwargs):
        conv_lens, w, w_lens, p, g, g_lens, s, s_lens, asv, asv_lens = \
            self.tuplize_data(data)
        batch_size, max_conv_len, max_sent_len = w.size()
        num_asv, max_asv_len = asv.size()
        conv_mask = utils.mask(conv_lens)
        w_emb = self.word_encoder(w, w_lens)
        p_emb = self.spkr_eye[p]

        u = self.sent_post_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
            self.post_query.unsqueeze(0).expand(batch_size * max_conv_len, -1)
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = u.masked_fill(~conv_mask.unsqueeze(-1), 0)
        _, c = self.conv_encoder(
            u,
            conv_lens,
            self.post_query.unsqueeze(0).expand(batch_size, -1),
        )
        zconv_post: MultiGaussian = self.zconv_post(c)
        zconv = zconv_post.sample(sample_scale)
        zconv_exp = zconv.unsqueeze(1).expand(-1, max_conv_len, -1)
        zconv_exp_flat = zconv_exp.reshape(-1, self.zconv_dim)

        w_emb = self.dropout(w_emb,
                             dropout_rate=self.word_dropout * dropout_scale,
                             unk=self.word_unk)
        u = self.sent_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
            zconv_exp_flat
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = self.dropout(u, self.sent_dropout * dropout_scale, self.sent_unk)
        r = self.dropout(self.spkr_encoder(p_emb),
                         dropout_rate=self.spkr_dropout * dropout_scale,
                         unk=self.spkr_unk)
        context, _, _ = self.ctx_encoder(
            torch.cat([
                utils.shift(u, dim=1),
                utils.shift(r, dim=1),
                zconv.unsqueeze(1).expand(-1, max_conv_len, -1)
            ], 2),
            conv_lens
        )
        zspkr_post: MultiGaussian = self.zspkr_post(
            torch.cat([
                r.view(-1, self.spkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zspkr_dim)
        zspkr_prior: MultiGaussian = self.zspkr_prior(
            torch.cat([
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zspkr_dim)
        zspkr = (zspkr_post.sample(sample_scale)
                 .view(batch_size, max_conv_len, -1)
                 .masked_fill(~conv_mask.unsqueeze(-1), 0))
        zsent_post: MultiGaussian = self.zsent_post(
            torch.cat([
                u.view(-1, self.sent_dim),
                zspkr.view(-1, self.zspkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zsent_dim)
        zsent_prior: MultiGaussian = self.zsent_prior(
            torch.cat([
                zspkr.view(-1, self.zspkr_dim),
                context.view(-1, self.ctx_dim),
                zconv_exp_flat
            ], 1)
        ).view_(batch_size, max_conv_len, self.zsent_dim)
        zsent = (zsent_post.sample(sample_scale)
                 .view(batch_size, max_conv_len, -1)
                 .masked_fill(~conv_mask.unsqueeze(-1), 0))
        p_logit = self.spkr_decoder(
            (torch.cat([zspkr, context, zconv_exp], 2)
             .view(batch_size * max_conv_len, -1))
        ).view(batch_size, max_conv_len, -1)
        w_logit = self.sent_decoder(
            (torch.cat([zsent, zspkr, context, zconv_exp], 2)
             .view(batch_size * max_conv_len, -1)),
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1)
        ).view(batch_size, max_conv_len, max_sent_len, -1)
        # gibberish
        g_logit = w_logit.new(batch_size, max_conv_len, num_asv).normal_()
        s_logit = w_logit.new(batch_size, max_conv_len, num_asv).normal_()
        return (
            {  # logits
                "sent": w_logit,
                "speaker": p_logit,
                "goal": g_logit,
                "state": s_logit
            }, {  # posterior
                "conv": zconv_post,
                "sent": zsent_post,
                "speaker": zspkr_post,
                "goal": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                         .to(w_logit.device)),
                "state": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                          .to(w_logit.device)),
            }, {  # prior
                "conv": (MultiGaussian.unit(*zconv_post.size())
                         .to(w_logit.device)),
                "sent": zsent_prior,
                "speaker": zspkr_prior,
                "goal": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                         .to(w_logit.device)),
                "state": (MultiGaussian.unit(batch_size, max_conv_len, 1)
                          .to(w_logit.device))
            }
        )

    def _encode_impl(self, data, *args, **kwargs):
        conv_lens = data["conv_lens"]
        w, w_lens = data["sent"], data["sent_lens"]
        batch_size, max_conv_len, max_sent_len = w.size()
        conv_mask = utils.mask(conv_lens)
        w_emb = self.word_encoder(w, w_lens)
        u = self.sent_post_encoder(
            w_emb.view(-1, max_sent_len, self.word_dim),
            w_lens.view(-1),
            self.post_query.unsqueeze(0).expand(batch_size * max_conv_len, -1)
        )[1].view(batch_size, max_conv_len, self.sent_dim)
        u = u.masked_fill(~conv_mask.unsqueeze(-1), 0)
        _, c = self.conv_encoder(
            u,
            conv_lens,
            self.post_query.unsqueeze(0).expand(batch_size, -1),
        )
        return self.zconv_post(c)

    def _decode_optimal_impl(self, data, conv_state=None,
                             init_u=None, init_r=None,
                             max_conv_len=20, max_sent_len=30,
                             spkr_scale=1.0, sent_scale=1.0,
                             beam_size=8, **kwargs):
        zconv = data["zconv"]
        batch_size = zconv.size(0)
        eoc_idx = self.vocabs.word["<eoc>"]
        sents, spkrs, goals, states = list(), list(), list(), list()
        conv_lens = zconv.new(batch_size).long().zero_()
        conv_done = zconv.new(batch_size).bool().zero_()
        spkr_logprob = zconv.new(batch_size).zero_()
        sent_logprob = zconv.new(batch_size).zero_()
        goal_logprob = zconv.new(batch_size).zero_()
        state_logprob = zconv.new(batch_size).zero_()
        if init_u is None:
            u = zconv.new(batch_size, self.sent_dim).zero_()
        else:
            u = init_u
        if init_r is None:
            r = zconv.new(batch_size, self.spkr_dim).zero_()
        else:
            r = init_r
        if conv_state is None:
            conv_state = self.ctx_encoder.init_state(batch_size)
        for i in range(max_conv_len):
            context, _, conv_state = self.ctx_encoder(
                torch.cat([u, r, zconv], 1).unsqueeze(1),
                lens=u.new(batch_size).long().fill_(1),
                h=conv_state
            )
            context = context.view(-1, self.ctx_dim)
            zspkr = self.zspkr_prior(
                torch.cat([context, zconv], 1)
            ).sample(spkr_scale)
            zutt = self.zsent_prior(
                torch.cat([zspkr, context, zconv], 1)
            ).sample(sent_scale)
            p_logit = self.spkr_decoder(
                torch.cat([zspkr, context, zconv], 1)
            )
            p_prob, p = torch.softmax(p_logit, -1).max(-1)
            p_logprob = p_prob.log().sum(-1)
            r = self.spkr_encoder(self.spkr_eye[p])
            w, w_lens, w_prob = self.sent_decoder.generate(
                h=torch.cat([
                    zutt,
                    zspkr,
                    context.view(batch_size, self.ctx_dim),
                    zconv
                ], 1),
                beam_size=beam_size,
                max_len=max_sent_len
            )
            w_logprob = w_prob.log()
            sents.append(Stacked1DTensor(w, w_lens))
            _, u = self.sent_encoder(self.word_encoder(w, w_lens),
                                     w_lens, zconv)
            spkrs.append(p)
            goals.append(Stacked1DTensor(
                value=w.new(batch_size, 0).zero_().long(),
                lens=w.new(batch_size).zero_().long()
            ))
            states.append(Stacked1DTensor(
                value=w.new(batch_size, 0).zero_().long(),
                lens=w.new(batch_size).zero_().long()
            ))
            conv_lens += 1 - conv_done.long()
            done = ((w == eoc_idx) & utils.mask(w_lens, w.size(1))).any(1)
            conv_done |= done
            sent_logprob = w_logprob.masked_fill(conv_done, 0)
            spkr_logprob = p_logprob.masked_fill(conv_done, 0)
            if conv_done.all().item():
                break
        sents = utils.stack_stacked1dtensors(sents)
        goals = utils.stack_stacked1dtensors(goals)
        states = utils.stack_stacked1dtensors(states)
        return BatchData(
            sent=DoublyStacked1DTensor(
                value=sents.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=sents.lens1.t().contiguous()
            ),
            speaker=Stacked1DTensor(
                value=torch.stack(spkrs).transpose(1, 0).contiguous(),
                lens=conv_lens
            ),
            goal=DoublyStacked1DTensor(
                value=goals.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=goals.lens1.t().contiguous()
            ),
            state=DoublyStacked1DTensor(
                value=states.value.permute(1, 0, 2).contiguous(),
                lens=conv_lens,
                lens1=states.lens1.t().contiguous()
            ),
        ), dict(
            logprob=(sent_logprob + spkr_logprob),
            sent_logprob=sent_logprob,
            goal_logprob=goal_logprob,
            state_logprob=state_logprob,
            spkr_logprob=spkr_logprob,
        )

    def _genconv_prior_impl(self, data, max_conv_len=20,
                            beam_size=4, max_sent_len=30,
                            conv_scale=1.0, spkr_scale=1.0,
                            goal_scale=1.0, state_scale=1.0, sent_scale=1.0):
        n = data["n"]
        zconv = MultiGaussian.unit(n.item(), self.ctx_dim).to(n.device)
        zconv_sample = zconv.sample(conv_scale)
        batch, info = self._decode_optimal_impl(
            data={"zconv": zconv_sample},
            eoc="<eoc>",
            max_conv_len=max_conv_len,
            spkr_scale=spkr_scale,
            sent_scale=sent_scale,
            beam_size=beam_size,
            max_sent_len=max_sent_len
        )
        info["zconv"] = zconv
        info["zconv-sample"] = zconv_sample
        return batch, info

    def _genconv_post_impl(self, data, max_conv_len=20,
                           beam_size=4, max_sent_len=30,
                           conv_scale=1.0, spkr_scale=1.0,
                           goal_scale=1.0, state_scale=1.0, sent_scale=1.0):
        zconv = self._encode_impl(data)
        zconv_sample = zconv.sample(conv_scale)
        batch, info = self._decode_optimal_impl(
            data={"zconv": zconv_sample},
            eoc="<eoc>",
            max_conv_len=max_conv_len,
            spkr_scale=spkr_scale,
            sent_scale=sent_scale,
            beam_size=beam_size,
            max_sent_len=max_sent_len
        )
        info["zconv"] = zconv
        info["zconv-sample"] = zconv_sample
        return batch, info
