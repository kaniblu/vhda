import torch
import torch.optim as op
import torch.nn as nn
from models import BeamSearcher


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.embedder = nn.Embedding(
            num_embeddings=4,
            embedding_dim=10
        )
        self.rnn = nn.LSTM(
            input_size=10,
            hidden_size=10,
        )
        self.output_layer = nn.Linear(
            in_features=10,
            out_features=4
        )

    def cell_forward(self, x, s=None):
        if s is not None:
            s = (s[0].unsqueeze(0), s[1].unsqueeze(0))
        o, s_prime = self.rnn(x.unsqueeze(0), s)
        return o.squeeze(0), (s_prime[0].squeeze(0), s_prime[1].squeeze(0))

    def forward(self, x):
        x = self.embedder(x)
        o, s = self.rnn(x.permute(1, 0, 2))
        o = o.permute(1, 0, 2)
        return self.output_layer(o)


def test_beamsearch():
    bos_idx, eos_idx = 2, 3
    gt_seq = torch.LongTensor([bos_idx] + [0, 1, 0, 1, 1, 0] + [eos_idx])
    model = Model()
    optimizer = op.Adam(model.parameters())
    ce = nn.CrossEntropyLoss()
    for i in range(2000):
        model.zero_grad()
        x, y = gt_seq[:-1], gt_seq[1:]
        pred = model(x.unsqueeze(0).expand(16, -1))
        loss = ce(pred.view(-1, 4), y.unsqueeze(0).repeat(16, 1).view(-1))
        loss.backward()
        optimizer.step()
        print(i, loss.item())
    bs = BeamSearcher(
        embedder=lambda x, lens=None: model.embedder(x),
        cell=model.cell_forward,
        classifier=model.output_layer,
        initial_logit=torch.eye(4)[bos_idx].log(),
        end_idx=eos_idx,
        beam_size=3
    )
    pred, lens, prob = bs.search((torch.zeros(1, 10), torch.zeros(1, 10)))
    print(pred, lens)
    assert lens[0][0].item() == gt_seq.size(0)
    assert (pred[0][0][:lens[0][0].item()] == gt_seq).all()


if __name__ == "__main__":
    test_beamsearch()
