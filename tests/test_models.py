import enum
import logging
import random
import functools
from dataclasses import dataclass
from typing import Sequence, Tuple, Mapping, Set

import torch
import torch.nn as nn
import torch.optim as op
from torchmodels.modules import pooling
from torchmodels.modules import feedforward

import utils
from datasets import BatchData
from datasets import Turn
from datasets import Dialog
from datasets import DialogState
from datasets import ActSlotValue
from datasets import create_dataloader
from datasets import DialogDataset
from datasets import DialogProcessor
from datasets import SentProcessor
from models import MultiGaussian
from models import MultiGaussianLayer
from models import BeamSearcher
from models import RNNSentEncoder
from models import RNNSentDecoder
from models import LSTMContextEncoder
from models import EmbeddingWordEncoder
from models import LSTMDecodingRNN
from models import VHCR
from models import VHDA
from models import SelfAttentiveSequenceEncoder
from models import GenericStateDecoder
from models import EmbeddingLabelEncoder
from models.rnn import LSTM
from models.rnn import BidirectionalLSTM


@dataclass
class SentDataset:
    vocab: utils.Vocabulary
    data: Sequence[Tuple[int, Sequence[int]]]


def create_sent_dataset() -> SentDataset:
    paragraph = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Integer nisi justo, ultricies eget tincidunt in, condimentum "
        "et libero. Nunc malesuada sollicitudin volutpat. Class aptent "
        "taciti sociosqu ad litora torquent per conubia nostra, per "
        "inceptos himenaeos. Duis fringilla feugiat tellus ac semper. "
        "Maecenas lectus mi, elementum a libero sed, fermentum tempor "
        "nisl. Proin ut pretium tellus, nec consectetur enim. "
        "Pellentesque dui risus, blandit in ex eu, accumsan rutrum leo. "
        "Nulla facilisi. Nam turpis dui, lacinia non tristique vel, "
        "feugiat et quam. Phasellus et tellus nisi. Curabitur lacus "
        "lectus, ullamcorper a nisi a, mattis dignissim arcu."
    )
    data = [list(map(lambda x: x.strip(), sent.split()))
            for sent in paragraph.split(".")]
    factory = utils.VocabularyFactory(reserved=["<bos>", "<eos>"])
    factory.update(w for s in data for w in s)
    vocab = factory.get_vocab()
    data = [
        (i, list(map(vocab.__getitem__, ["<bos>"] + sent + ["<eos>"])))
        for i, sent in enumerate(data)
    ]
    return SentDataset(
        vocab=vocab,
        data=data
    )


def create_dummy_dataset():
    data = [Dialog([
        Turn(
            text="chinese restaurant please",
            speaker="user1",
            goal=DialogState.from_asvs([
                ActSlotValue("inform", "food", "chinese")
            ]),
            state=DialogState.from_asvs([
                ActSlotValue("inform", "food", "chinese")
            ])
        ),
        Turn(
            text="found two restaurant",
            speaker="user2",
            goal=DialogState.from_asvs([
                ActSlotValue("offer", "found", "one"),
                ActSlotValue("inform", "address", "someplace")
            ]),
            state=DialogState.from_asvs([
                ActSlotValue("offer", "found", "two")
            ])
        )
    ]), Dialog([
        Turn(
            text="what about on the east",
            speaker="user2",
            goal=DialogState.from_asvs([
                ActSlotValue("inform", "food", "chinese"),
                ActSlotValue("inform", "area", "east")
            ]),
            state=DialogState.from_asvs([
                ActSlotValue("inform", "area", "east")
            ])
        ),
        Turn(
            text="found one and the address is someplace",
            speaker="user1",
            goal=DialogState.from_asvs([
                ActSlotValue("offer", "found", "one"),
                ActSlotValue("inform", "address", "someplace")
            ]),
            state=DialogState.from_asvs([
                ActSlotValue("offer", "found", "one"),
                ActSlotValue("inform", "address", "someplace")
            ])
        )
    ])]
    processor = DialogProcessor(
        sent_processor=SentProcessor(
            bos=True,
            eos=True,
            lowercase=True,
            tokenizer="space"
        ),
        boc=True,
        eoc=True
    )
    processor.prepare_vocabs(data)
    return DialogDataset(data, processor)


def create_dummy_conv_dataset2():
    return DialogDataset(
        [Dialog([
            Turn("foo bar", speaker=Speaker.USER,
                 states={"foo": {"bar"}}),
            Turn("bar foo", speaker=Speaker.WIZARD,
                 states={"foo": {"foo"}})
        ]), Dialog([
            Turn("foo bar foo", speaker=Speaker.USER,
                 states={"foo": {"bar", "foo"}}),
            Turn("bar foo", speaker=Speaker.WIZARD,
                 states={"foo": {"foo"}}),
            Turn("bar foo foo", speaker=Speaker.USER,
                 states={"foo": {"foo"}})
        ])]
    )


def create_conv_dataset() -> DialogDataset:
    return DialogDataset(load_woz("examples/train.json"))


def pad_stack(xs, fill=0):
    seq_len = list(map(len, xs))
    max_len = max(seq_len)
    return torch.stack([torch.cat([x, x.new(max_len - len(x)).fill_(fill)])
                        if len(x) < max_len else x for x in xs])


def test_sent_encoder_decoder():
    from models.seq_encoder import SelfAttentiveSequenceEncoder
    from models.word import EmbeddingWordEncoder
    dataset = create_sent_dataset()
    word_emb = EmbeddingWordEncoder(dataset.vocab, word_dim=100)
    encoder = SelfAttentiveSequenceEncoder(
        input_dim=100,
        hidden_dim=100,
        query_dim=0,
        rnn=LSTM
    )
    decoder = RNNSentDecoder(
        vocab=dataset.vocab,
        word_encoder=word_emb,
        rnn_dim=100,
        hidden_dim=100,
        decoding_rnn=LSTMDecodingRNN
    )
    word_emb.reset_parameters()
    encoder.reset_parameters()
    decoder.reset_parameters()
    optim = op.Adam(list(encoder.parameters()) + list(word_emb.parameters()) +
                    list(decoder.parameters()))
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    for i in range(2000):
        word_emb.zero_grad()
        encoder.zero_grad()
        decoder.zero_grad()
        random.shuffle(dataset.data)
        y = [torch.tensor(a) for _, a in dataset.data]
        lens = torch.LongTensor(list(map(len, y)))
        y_in = pad_stack([a[:-1] for a in y])
        y_out = pad_stack([a[1:] for a in y], fill=-1)
        y = pad_stack(y)
        y_emb = word_emb(y, lens)
        _, h = encoder(y_emb, lens)
        logit = decoder(h, word_emb(y_in, lens - 1), lens - 1)
        loss = ce(logit.view(-1, logit.size(-1)), y_out.view(-1))
        loss.backward()
        optim.step()
        print(i, loss.item())
    vocab = dataset.vocab
    bs_cls = functools.partial(
        BeamSearcher,
        initial_logit=torch.eye(len(vocab))[vocab["<bos>"]].log(),
        end_idx=dataset.vocab["<eos>"]
    )
    random.shuffle(dataset.data)
    y = [torch.tensor(a) for _, a in dataset.data]
    lens = torch.LongTensor(list(map(len, y)))
    pred, lens, prob = decoder.generate(encoder(pad_stack(y), lens)[1], bs_cls)
    for i, (pd, l, gold) in enumerate(zip(pred, lens, y)):
        print(i)
        print(pd[:l], gold)
        assert l.item() == len(gold)
        assert (pd[:l.item()] == gold).all()


def test_sent_decoder():
    dataset = create_sent_dataset()
    random.shuffle(dataset.data)
    x, y = zip(*((i, torch.tensor(x)) for i, x in dataset.data))
    x = torch.tensor(x)
    model = RNNSentDecoder(
        vocab=dataset.vocab,
        hidden_dim=len(x),
        word_dim=50,
        word_encoder=EmbeddingWordEncoder,
        decoding_rnn=lambda *args, **kwargs: LSTMDecodingRNN(
            *args, init_layer=feedforward.MultiLayerFeedForward, **kwargs
        ),
        output_layer=feedforward.MultiLayerFeedForward
    )
    model.reset_parameters()
    print(dataset)
    print(model)
    optim = op.Adam(model.parameters())
    ce = nn.CrossEntropyLoss(ignore_index=-1)
    for i in range(2000):
        model.zero_grad()
        random.shuffle(dataset.data)
        x, y = zip(*((i, torch.tensor(a)) for i, a in dataset.data))
        x = torch.tensor(x)
        y_lens = torch.tensor(list(map(len, y))).long()
        y_in = pad_stack([a[:-1] for a in y])
        y_out = pad_stack([a[1:] for a in y], fill=-1)
        logit = model(torch.eye(len(x))[x], y_in, y_lens - 1)
        loss = ce(logit.view(-1, logit.size(-1)), y_out.view(-1))
        loss.backward()
        optim.step()
        print(i, loss.item())
    bs_cls = functools.partial(
        BeamSearcher,
        initial_logit=torch.eye(len(x))[dataset.vocab["<bos>"]].log(),
        end_idx=dataset.vocab["<eos>"]
    )
    pred, lens, prob = model.generate(torch.eye(len(x))[x], bs_cls)
    for i, (pd, l, gold) in enumerate(zip(pred, lens, y)):
        print(i)
        print(pd[:l], gold)
        assert l.item() == len(gold)
        assert (pd[:l.item()] == gold).all()


def test_gaussian():
    batch_size = 32
    layer = MultiGaussianLayer(10, 20)
    layer.reset_parameters()
    optim = op.Adam(layer.parameters())
    gaus: MultiGaussian = layer(torch.randn(batch_size, 10))
    print(f"layer random mu: {gaus.mu.mean()}, random std: {gaus.std.mean()}")
    for i in range(10000):
        optim.zero_grad()
        loss = layer(torch.randn(batch_size, 10)).kl_div().mean()
        loss.backward()
        optim.step()
        if i % 100 == 0:
            print(i, loss.item())
    gaus: MultiGaussian = layer(torch.randn(batch_size, 10))
    print(f"layer random mu: {gaus.mu.mean()}, random std: {gaus.std.mean()}")
    x: MultiGaussian = layer(torch.randn(32, 10))
    y: MultiGaussian = layer(torch.randn(32, 10))
    samples = torch.stack([x.sample() for _ in range(1000)])
    print(f"layer random sample mu: {samples.mean()}, "
          f"sample var: {samples.std()}")
    print(f"average KLD(random1, random2): {x.kl_div(y).mean()}")
    print(f"average KLD(random1, unit): {x.kl_div().mean()}")
    unit = MultiGaussian(torch.zeros(32, 20), torch.zeros(32, 20))
    print(f"average KLD(unit, unit): {unit.kl_div().mean()}")
    assert (x.kl_div(y) >= 0).all()
    assert (x.kl_div() >= 0).all()
    assert (unit.kl_div() == 0).all()
    assert (x.kl_div() == x.kl_div(unit)).all()


def has_nan(p):
    if p is None:
        return False
    return bool((p != p).any().item())


def create_vhcr(dataset: DialogDataset):
    return VHCR(
        vocabs=dataset.vocabs,
        sent_encoder=functools.partial(
            RNNSentEncoder,
            word_encoder=functools.partial(
                EmbeddingWordEncoder,
                pad=True
            ),
            rnn=rnn.GRU,
            pooling=pooling.MaxPooling
        ),
        conv_encoder=functools.partial(
            LSTMContextEncoder,
            pack_sequence=True,
            pooling=pooling.MaxPooling
        ),
        conv_post_encoder=functools.partial(
            LSTMContextEncoder,
            bidirectional=True,
            pack_sequence=True,
            pooling=pooling.MaxPooling
        ),
        sent_decoder=functools.partial(
            RNNSentDecoder,
            word_encoder=EmbeddingWordEncoder,
            decoding_rnn=functools.partial(
                LSTMDecodingRNN,
                init_layer=feedforward.MultiLayerFeedForward
            ),
            output_layer=feedforward.MultiLayerFeedForward
        )
    )


def create_vhda(dataset: DialogDataset):
    model = VHDA(
        vocabs=dataset.processor.vocabs,
        word_encoder=EmbeddingWordEncoder,
        seq_encoder=functools.partial(
            SelfAttentiveSequenceEncoder,
            rnn=LSTM
        ),
        state_decoder=GenericStateDecoder,
        ctx_encoder=LSTMContextEncoder,
        sent_decoder=functools.partial(
            RNNSentDecoder,
            decoding_rnn=LSTMDecodingRNN,
        )
    )
    logging.info(str(model))
    logging.info(f"num params: {utils.count_parameters(model):,d}")
    return model


def vhda_gen(model):
    return functools.partial(
        model,
        conv_scale=0.0,
        speaker_scale=0.0,
        goal_scale=0.0,
        turn_scale=0.0,
        sent_scale=0.0
    )


def test_jda(create_fn=create_vhda, gen_fn=vhda_gen):
    dataset = create_dummy_dataset()
    dataloader = create_dataloader(
        dataset,
        batch_size=2,
    )
    model = create_fn(dataset)
    optimizer = op.Adam(p for p in model.parameters() if p.requires_grad)
    ce = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    model.reset_parameters()
    for eidx in range(300):
        model.train()
        for i, batch in enumerate(dataloader):
            batch: BatchData = batch
            optimizer.zero_grad()
            model.inference()
            w, p = batch.word, batch.speaker
            g, g_lens = batch.goal, batch.goal_lens
            s, s_lens = batch.turn, batch.turn_lens
            sent_lens, conv_lens = batch.sent_lens, batch.conv_lens
            batch_size, max_conv_len, max_sent_len = w.size()
            w_logit, p_logit, g_logit, s_logit, info = model(batch.to_dict())
            w_target = w.masked_fill(~utils.mask(conv_lens).unsqueeze(-1), -1)
            w_target = w_target.view(-1, max_sent_len).masked_fill(
                ~utils.mask(sent_lens.view(-1)), -1
            ).view(batch_size, max_conv_len, -1)
            recon_loss = ce(
                w_logit[:, :, :-1].contiguous().view(-1, w_logit.size(-1)),
                w_target[:, :, 1:].contiguous().view(-1)
            ).view(batch_size, max_conv_len, max_sent_len - 1).sum(-1).sum(-1)
            goal_loss = bce(
                g_logit,
                utils.to_dense(g, g_lens, g_logit.size(-1)).float()
            )
            goal_loss = (goal_loss.masked_fill(~utils.mask(conv_lens)
                                               .unsqueeze(-1).unsqueeze(-1), 0)
                         .sum(-1).sum(-1).sum(-1))
            turn_loss = bce(
                s_logit,
                utils.to_dense(s, s_lens, s_logit.size(-1)).float()
            )
            turn_loss = (turn_loss.masked_fill(~utils.mask(conv_lens)
                                               .unsqueeze(-1).unsqueeze(-1), 0)
                         .sum(-1).sum(-1).sum(-1))
            speaker_loss = ce(
                p_logit.view(-1, p_logit.size(-1)),
                p.masked_fill(~utils.mask(conv_lens), -1).view(-1)
            ).view(batch_size, max_conv_len).sum(-1)
            kld_loss = sum(v for k, v in info.items()
                           if k in {"sent", "conv", "speaker", "goal", "turn"})
            loss = (recon_loss + goal_loss + turn_loss + speaker_loss +
                    kld_loss * min(0.3, max(0.01, i / 500)))
            print(f"[e{eidx + 1}] "
                  f"loss={loss.mean().item(): 4.4f} "
                  f"recon={recon_loss.mean().item(): 4.4f} "
                  f"goal={goal_loss.mean().item(): 4.4f} "
                  f"turn={turn_loss.mean().item(): 4.4f} "
                  f"speaker={speaker_loss.mean().item(): 4.4f} "
                  f"kld={kld_loss.mean().item(): 4.4f}")
            loss.mean().backward()
            optimizer.step()

            model.eval()
            model.genconv_post()
            batch_gen, info = gen_fn(model)(batch.to_dict())
            print("Input: ")
            print(f"{dataset.processor.lexicalize(batch[0])}")
            print()
            print(f"Predicted (prob={info['logprob'][0].exp().item():.4f}): ")
            print(f"{dataset.processor.lexicalize(batch_gen[0])}")
    model.eval()
    model.genconv_post()
    for batch in dataloader:
        batch_gen, logprobs = gen_fn(model)(batch.to_dict())
        for x, y in zip(map(dataset.processor.lexicalize, batch),
                        map(dataset.processor.lexicalize, batch_gen)):
            assert x == y, f"{x}\n!=\n{y}"


def test_rnn():
    x = torch.randn(3, 4, 6)
    lens = torch.tensor([2, 4, 1])
    lstm = BidirectionalLSTM(input_dim=6, hidden_dim=6, packed=True)
    o, c, h = lstm(x, lens)
    x2, lens2 = x[:1, :lens[0]], lens[:1]
    o2, c2, h2 = lstm(x2, lens2)

    assert utils.compare_tensors(x[:1, :2], x2)
    assert ((o[:1, :2] - o2) < 0.0001).all().item()
    assert ((h[:1] - h2) < 0.0001).all().item()


def test_turn_state_encoder_decoder():
    dataset = create_dummy_dataset()
    vocabs = list(dataset.vocabs.turn.slot_values.values())
    encoder = GenericStateEncoder(
        vocabs=vocabs,
        output_dim=100,
        label_encoder=functools.partial(
            EmbeddingLabelEncoder
        ),
        label_layer=feedforward.MultiLayerFeedForward,
        label_pooling=pooling.SumPooling,
        state_pooling=pooling.MaxPooling,
        output_layer=feedforward.MultiLayerFeedForward
    )
    decoder = GenericStateDecoder(
        input_dim=100,
        vocabs=vocabs,
        input_layer=feedforward.MultiLayerFeedForward,
        output_layer=feedforward.MultiLayerFeedForward,
        label_emb=EmbeddingLabelEncoder
    )
    encoder.reset_parameters()
    decoder.reset_parameters()
    encoder.train(), decoder.train()
    params = [p for p in encoder.parameters() if p.requires_grad]
    params += [p for p in decoder.parameters() if p.requires_grad]
    optimizer = op.Adam(params)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    vocab_lens = torch.LongTensor(list(map(len, vocabs)))
    x_sparse = torch.randint(0, 2, (4, len(vocab_lens), max(vocab_lens))).byte()
    x_sparse = x_sparse.masked_fill(~utils.mask(vocab_lens), 0)
    x, lens = utils.to_sparse(x_sparse)
    x_sparse = x_sparse.masked_fill(~utils.mask(vocab_lens), -1)
    lens = torch.randint(0, 3, (4, len(encoder.vocabs))) + 1
    for i in range(100):
        logits = decoder(encoder(x, lens))
        loss = bce(logits, x_sparse.float())
        loss = loss.masked_fill(~utils.mask(vocab_lens), 0).sum()
        print(i, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    encoder.eval(), decoder.eval()
    logits = decoder(encoder(x, lens))
    x_pred = torch.sigmoid(logits) > 0.5
    x_pred = x_pred.masked_fill(~utils.mask(vocab_lens), -1)
    assert (x_pred == x_sparse).all().item()


def test_tda_inference(create_fn=create_vhda):
    dataset = create_dummy_dataset()
    dataloader = create_dataloader(dataset, batch_size=3)
    model = create_fn(dataset)
    model.reset_parameters()
    batch: BatchData = next(iter(dataloader))
    asv = dataset.processor.tensorize_state_vocab("goal_state")
    logit, post, prior = model({
        "conv_lens": batch.conv_lens,
        "sent": batch.sent.value,
        "sent_lens": batch.sent.lens1,
        "speaker": batch.speaker.value,
        "state": batch.state.value,
        "state_lens": batch.state.lens1,
        "goal": batch.goal.value,
        "goal_lens": batch.goal.lens1,
        "asv": asv.value,
        "asv_lens": asv.lens
    })
    print(logit, post, prior)
    for k, v in logit.items():
        assert not utils.has_nan(v), f"nan detected in {k} logit"
    for k, gaus in post.items():
        assert not utils.has_nan(gaus.mu) and not utils.has_nan(gaus.logvar), \
            f"nan detected in {k} posterior distribution"
    for k, gaus in prior.items():
        assert not utils.has_nan(gaus.mu) and not utils.has_nan(gaus.logvar), \
            f"nan detected in {k} prior distribution"


def main():
    test_sent_encoder_decoder()


if __name__ == "__main__":
    main()
