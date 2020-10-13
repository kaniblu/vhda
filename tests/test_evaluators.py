import pprint

import torch

from utils import *
from datasets import *
from evaluators import *
from embeddings import *


def load_data():
    data = DialogDataset(load_woz("examples/train.json"))
    loader = create_dataloader(
        dataset=data,
        batch_size=4,
        shuffle=True
    )
    batch = next(iter(loader))
    return data, batch


def generate_input_output(data, batch):
    w, p, s, conv_lens, sent_lens, state_lens = batch.tensors
    batch_size, max_conv_len, max_sent_len = w.size()
    num_states = s.size(2)
    max_labels = max(map(len, data.vocabs.state_labels.values()))
    num_words = len(data.vocabs.word)
    outputs = (
        torch.randn(batch_size, max_conv_len, max_sent_len, num_words),
        torch.randn(batch_size, max_conv_len, 2),
        torch.randn(batch_size, max_conv_len, num_states, max_labels)
    )
    return outputs


def generate_gold(data, batch):
    w, p, s, conv_lens, sent_lens, state_lens = batch.tensors
    max_labels = max(map(len, data.vocabs.state_labels.values()))
    outputs = (
        torch.eye(len(data.vocabs.word))[w],
        torch.eye(2)[p],
        to_dense(s, state_lens, max_labels).float().add(-0.5).mul(float("inf"))
    )
    return outputs


def predict(batch: BatchData, outputs) -> BatchData:
    w_logit, p_logit, s_logit = outputs
    s_pred, s_lens = to_sparse(torch.sigmoid(s_logit) > 0.5)
    return BatchData(
        word=w_logit.max(-1)[1],
        speaker=p_logit.max(-1)[1],
        states=s_pred,
        conv_lens=batch.conv_lens,
        sent_lens=batch.sent_lens,
        state_lens=s_lens
    )


def test_dstate():
    data, batch = load_data()
    outputs = generate_input_output(data, batch)
    ev = DialogStateEvaluator(data.vocabs)
    pprint.pprint(ev(predict(batch, outputs), batch))
    gold_outputs = generate_gold(data, batch)
    gold_eval = ev(predict(batch, gold_outputs), batch)
    pprint.pprint(gold_eval)
    assert all(v.item() == 1.0 for v in gold_eval.values())


def test_speaker():
    data, batch = load_data()
    outputs = generate_input_output(data, batch)
    ev = SpeakerEvaluator(Vocabulary(
        f2i={Speaker.USER.name: 0, Speaker.WIZARD.name: 1},
        i2f={0: Speaker.USER.name, 1: Speaker.WIZARD.name}
    ))
    pprint.pprint(ev(predict(batch, outputs), batch))
    gold_outputs = generate_gold(data, batch)
    gold_eval = ev(predict(batch, gold_outputs), batch)
    pprint.pprint(gold_eval)
    assert all(v.item() == 1.0 for v in gold_eval.values())


def test_embedding():
    data, batch = load_data()
    embed = GloveFormatEmbeddings("examples/glove.840B.300d.woz.txt")
    embed.preload()
    ev = EmbeddingEvaluator(data.vocabs.word, embeds=embed)
    outputs = generate_input_output(data, batch)
    pprint.pprint(ev(predict(batch, outputs), batch))
    gold_outputs = generate_gold(data, batch)
    gold_eval = ev(predict(batch, gold_outputs), batch)
    pprint.pprint(gold_eval)
    assert all(v.item() == 1.0 for v in gold_eval.values())


if __name__ == "__main__":
    test_embedding()
