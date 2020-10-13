import logging
import logging.config
from dataclasses import dataclass

from torch.utils import tensorboard

import utils
from loopers import Inferencer
from loopers import LossInferencer
from loopers import LogInferencer
from loopers import TrainInferencer
from loopers import EvaluatingInferencer
from loopers import Generator
from loopers import LogGenerator
from loopers import BeamSearchGenerator
from loopers import EvaluatingGenerator
from losses import VHDALoss
from datasets import create_dataloader
from evaluators import SpeakerEvaluator
from evaluators import DialogStateEvaluator
from evaluators import BLEUEvaluator
from evaluators import DistinctEvaluator
from evaluators import SentLengthEvaluator
from evaluators import RougeEvaluator
from evaluators import EmbeddingEvaluator
from evaluators import WordEntropyEvaluator
from evaluators import StateEntropyEvaluator
from embeds import GloveFormatEmbeddings
from .test_models import create_dummy_dataset
from .test_models import create_vhda


@dataclass
class TestInferencer(LogInferencer, EvaluatingInferencer, TrainInferencer,
                     LossInferencer, Inferencer):
    pass


@dataclass
class TestGenerator(
    LogGenerator,
    EvaluatingGenerator,
    BeamSearchGenerator,
    Generator
):
    pass


def test_inferencer():
    dataset = create_dummy_dataset()
    dataloader = create_dataloader(dataset, batch_size=3)
    model = create_vhda(dataset)
    model.reset_parameters()
    looper = TestInferencer(
        model=model,
        report_every=1,
        loss=VHDALoss(
            vocabs=dataset.processor.vocabs
        ),
        writer=tensorboard.SummaryWriter("/tmp/test"),
        display_stats={"loss", "kld", "spkr-acc-turn",
                       "state-acc-turn", "goal-acc-turn"},
        progress_stat="loss",
        evaluators=(
            SpeakerEvaluator(dataset.processor.vocabs.speaker),
            DialogStateEvaluator(dataset.processor.vocabs)
        ),
        processor=dataset.processor
    )
    stats = None
    for _ in range(100):
        stats = looper(dataloader)
    assert stats is not None
    for k, v in stats.items():
        assert not utils.has_nan(v), f"{k} item contains NaN"


def test_generator():
    dataset = create_dummy_dataset()
    dataloader = create_dataloader(dataset, batch_size=3)
    model = create_vhda(dataset)
    model.reset_parameters()
    trainer = TestInferencer(
        model=model,
        report_every=1,
        loss=VHDALoss(
            vocabs=dataset.processor.vocabs
        ),
        writer=tensorboard.SummaryWriter("/tmp/test"),
        display_stats={"loss", "kld", "spkr-acc-turn",
                       "state-acc-turn", "goal-acc-turn"},
        progress_stat="loss",
        evaluators=(
            SpeakerEvaluator(dataset.processor.vocabs.speaker),
            DialogStateEvaluator(dataset.processor.vocabs)
        ),
        processor=dataset.processor
    )
    for _ in range(20):
        trainer(dataloader)
    generator = TestGenerator(
        model=model,
        processor=dataset.processor,
        beam_size=2,
        evaluators=(
            BLEUEvaluator(dataset.processor.vocabs),
            DistinctEvaluator(dataset.processor.vocabs),
            SentLengthEvaluator(dataset.processor.vocabs),
            RougeEvaluator(dataset.processor.vocabs),
            EmbeddingEvaluator(
                vocab=dataset.processor.vocabs.word,
                embeds=GloveFormatEmbeddings(
                    path="tests/data/glove/glove.840B.300d.woz.txt"
                ).preload()
            ),
            WordEntropyEvaluator(dataset),
            StateEntropyEvaluator(dataset)
        ),
        display_stats=({"bleu-smooth7-user1", "bleu-smooth7-user2",
                       "dist-1", "dist-2", "rouge-l-f1-user1",
                        "sent-len-user1", "emb-greedy"}),
        report_every=1
    )
    samples, stats = generator(dataset.data, 10)
    print(samples)
    print(stats)


def main():
    logging.config.dictConfig(utils.load_yaml("tests/config/logging.yml"))
    test_generator()


if __name__ == "__main__":
    main()
