__all__ = ["train", "TrainArguments", "Record", "create_loss"]

import pprint
import random
import pathlib
import functools
import itertools
import logging
import logging.config
from dataclasses import dataclass, field
from typing import Optional, Sequence, ClassVar

import yaap
import inflect
import torch
import torch.utils.data as td
import torch.optim as op
import torch.utils.tensorboard
import torchmodels

import utils
import models
import models.dialog
import datasets
import losses
from datasets import Dialog
from evaluators import SpeakerEvaluator
from evaluators import DialogStateEvaluator
from evaluators import DistinctEvaluator
from evaluators import SentLengthEvaluator
from evaluators import RougeEvaluator
from evaluators import PosteriorEvaluator
from evaluators import DialogLengthEvaluator
from evaluators import WordEntropyEvaluator
from loopers import LossInferencer
from loopers import TrainInferencer
from loopers import EvaluatingInferencer
from loopers import VHDAInferencer
from loopers import LogInferencer
from loopers import Generator
from loopers import LogGenerator
from loopers import BeamSearchGenerator
from loopers import EvaluatingGenerator


def create_loss(model, vocabs: datasets.VocabSet,
                kld_weight: utils.Scheduler = utils.ConstantScheduler(1.0),
                enable_kl=True, kl_mode="kl-mi") -> losses.Loss:
    assert kl_mode in {"kl", "kl-mi", "kl-mi+"}
    if isinstance(model, models.VHDA):
        return losses.VHDALoss(
            vocabs=vocabs,
            kld_weight=kld_weight,
            enable_kl=enable_kl,
            kl_mode=kl_mode
        )
    elif isinstance(model, models.VHCR):
        return losses.VHCRLoss(
            vocabs=vocabs,
            enable_kl=enable_kl,
            kld_weight=kld_weight,
            kl_mode=kl_mode
        )
    elif isinstance(model, models.HDA):
        return losses.HDALoss(
            vocabs=vocabs
        )
    elif isinstance(model, models.VHDAWithoutGoal):
        return losses.VHDAWithoutGoalLoss(
            vocabs=vocabs,
            kld_weight=kld_weight,
            enable_kl=enable_kl,
            kl_mode=kl_mode
        )
    elif isinstance(model, models.VHDAWithoutGoalAct):
        return losses.VHDAWithoutGoalActLoss(
            vocabs=vocabs,
            kld_weight=kld_weight,
            enable_kl=enable_kl,
            calibrate_mi="mi" in kl_mode
        )
    elif isinstance(model, models.VHRED):
        return losses.VHREDLoss(
            vocabs=vocabs,
            enable_kl=enable_kl,
            kld_weight=kld_weight
        )
    elif isinstance(model, models.VHUS):
        return losses.VHUSLoss(
            vocabs=vocabs,
            enable_kl=enable_kl,
            kld_weight=kld_weight
        )
    else:
        raise RuntimeError(f"unsupported model: {type(model)}")


@dataclass
class TrainArguments(utils.Arguments):
    model: models.AbstractTDA
    train_data: Sequence[Dialog]
    valid_data: Sequence[Dialog]
    processor: datasets.DialogProcessor
    device: torch.device = torch.device("cpu")
    save_dir: pathlib.Path = pathlib.Path("out")
    report_every: Optional[int] = None
    batch_size: int = 32
    valid_batch_size: int = 64
    optimizer: str = "adam"
    gradient_clip: Optional[float] = None
    l2norm_weight: Optional[float] = None
    learning_rate: float = 0.001
    num_epochs: int = 10
    kld_schedule: utils.Scheduler = utils.ConstantScheduler(1.0)
    dropout_schedule: utils.Scheduler = utils.ConstantScheduler(1.0)
    validate_every: int = 1
    beam_size: int = 4
    max_gen_len: int = 30
    early_stop: bool = False
    early_stop_criterion: str = "~val-loss"
    early_stop_patience: Optional[int] = None
    save_every: Optional[int] = None
    disable_kl: bool = False
    kl_mode: str = "kl-mi"


@dataclass
class FinegrainedValidator(LogInferencer, EvaluatingInferencer, LossInferencer):

    def on_run_started(self, dataloader: td.DataLoader) -> td.DataLoader:
        ret = super().on_run_started(dataloader)
        self.model.eval()
        return ret


@dataclass
class GenerativeValidator(
    LogGenerator,
    EvaluatingGenerator,
    BeamSearchGenerator,
    Generator
):
    pass


@dataclass
class Record:
    state_dict: dict = field(repr=False)
    criterion: float
    epoch_idx: int
    summary: utils.TensorMap

    def to_short_json(self):
        return {
            "criterion": self.criterion,
            "epoch": self.epoch_idx,
            "summary": {k: v.item() for k, v in
                        self.summary.items() if k.count("-") < 2}
        }

    def to_json(self):
        return {
            "criterion": self.criterion,
            "epoch": self.epoch_idx,
            "summary": {k: v.item() for k, v in self.summary.items()}
        }


@dataclass
class Trainer(LogInferencer, EvaluatingInferencer, TrainInferencer):
    save_dir: pathlib.Path = None
    num_epochs: int = 10
    fin_valid: FinegrainedValidator = None
    gen_valid: GenerativeValidator = None
    validate_every: int = 1
    early_stop: bool = False
    early_stop_criterion: str = "~val-loss"  # use 'tilde' to denote negation
    early_stop_patience: Optional[int] = None
    save_every: Optional[int] = None
    _best_record: Record = field(init=False, repr=False, default=None)
    _eidx: int = field(init=False, default=None)
    _val_dataloader: Optional[td.DataLoader] = field(init=False, default=None)
    _continue: bool = field(init=False, default=True)
    _engine: ClassVar[inflect.engine] = inflect.engine()

    def __post_init__(self):
        super().__post_init__()
        if self.save_dir is None:
            raise ValueError(f"must provide save dir")
        if self.fin_valid is None:
            raise ValueError(f"must provide a (fine-grained) validator")
        if self.gen_valid is None:
            raise ValueError(f"must provide a (dialog) validator")
        if self.gen_valid is None:
            raise ValueError(f"must provide a generator")
        self.save_dir.mkdir(exist_ok=True)

    def make_record(self, stats):
        assert "epoch" in stats
        if self._best_record is not None:
            del self._best_record.state_dict
            del self._best_record
        crit_key = self.early_stop_criterion.lstrip("~")
        if crit_key not in stats:
            raise KeyError(f"not a valid criterion: {crit_key}; "
                           f"available criteria: {tuple(stats.keys())}")
        self._best_record = Record(
            state_dict=self.state_dict(),
            criterion=stats[crit_key].item(),
            epoch_idx=stats["epoch"].item(),
            summary={k: v.cpu().detach() for k, v in stats.items()}
        )
        self._logger.info(f"new record found: "
                          f"{self._best_record.to_short_json()}")
        utils.save_json(self._best_record.to_json(),
                        self.save_dir.joinpath("best-record.json"))
        self.save_snapshot(self._best_record.state_dict, "best")

    def state_dict(self):
        return {k: v.cpu().detach().clone()
                for k, v in self.model.state_dict().items()}

    def check_early_stop(self, stats):
        assert "epoch" in stats
        if self._best_record is None:
            self.make_record(stats)
            return False
        neg = self.early_stop_criterion.startswith("~")
        crit_key = self.early_stop_criterion.lstrip("~")
        if crit_key not in stats:
            self._logger.warning(f"early stopping criterion {crit_key} not "
                                 f"found in `stats` ({stats.keys()}); falling "
                                 f"back to default value of 0")
        crit = stats.get(crit_key, torch.tensor(0.0)).item()
        if (crit > self._best_record.criterion) != neg:
            self.make_record(stats)
            return False
        return (self.early_stop_patience is not None and
                stats["epoch"] > (self._best_record.epoch_idx +
                                  self.early_stop_patience))

    def save_snapshot(self, state_dict, tag=None):
        segments = ["checkpoint"]
        if tag is not None:
            segments.append(tag)
        path = self.save_dir.joinpath("-".join(segments) + ".pth")
        self._logger.info(f"saving snapshot to {path}...")
        torch.save(state_dict, path)

    def train(self, dataloader: td.DataLoader, val_dataloader: td.DataLoader
              ) -> Record:
        stats = None
        for eidx in range(1, self.num_epochs + 1):
            stats = self(dataloader)
            stats["epoch"] = torch.tensor(eidx)
            if eidx % self.validate_every == 0:
                with torch.no_grad():
                    fval_stats = self.fin_valid(val_dataloader)
                with torch.no_grad():
                    samples, gval_stats = \
                        self.gen_valid(val_dataloader.dataset.data)
                sample = random.choice(samples)
                self.log_dialog("gval-input-sample", sample.input)
                self.log_dialog("gval-gen-sample", sample.output)
                with torch.no_grad():
                    samples, _ = self.gen_valid(dataloader.dataset.data, 3)
                for i, sample in enumerate(samples, 1):
                    self.log_dialog(f"gtrain-input-sample-{i}", sample.input)
                    self.log_dialog(f"gtrain-gen-sample-{i}", sample.output)
                val_stats = utils.merge_dict(fval_stats, gval_stats)
                self.log_stats(f"e{eidx}-val-summary", val_stats, prefix="val")
                stats.update({f"val-{k}": v for k, v in val_stats.items()})
                if self.early_stop and self.check_early_stop(stats):
                    break
            if self.save_every is not None and eidx % self.save_every == 0:
                self.save_snapshot(self.state_dict(), f"e{eidx}")
        if stats is not None and self._best_record is None:
            self.make_record(stats)
        if self.early_stop and self._best_record is not None:
            self.model.load_state_dict(self._best_record.state_dict)
        if self._best_record is not None:
            self._logger.info(f"final summary: "
                              f"{pprint.pformat(self._best_record.to_json())}")
        self.save_snapshot(self.state_dict(), "final")
        return self._best_record


def train(args: TrainArguments) -> Record:
    model, device = args.model, args.device
    save_dir = args.save_dir
    shell = utils.ShellUtils()
    shell.mkdir(save_dir, silent=True)
    utils.save_json(args.to_json(), str(save_dir.joinpath("args.json")))
    logger = logging.getLogger("train")
    processor = args.processor
    vocabs: datasets.VocabSet = processor.vocabs
    train_dataset = datasets.DialogDataset(
        data=args.train_data,
        processor=processor
    )
    valid_dataset = datasets.DialogDataset(
        data=args.valid_data,
        processor=processor
    )
    logger.info("preparing training environment...")
    loss = create_loss(
        model=model,
        vocabs=vocabs,
        kld_weight=args.kld_schedule,
        enable_kl=not args.disable_kl,
        kl_mode=args.kl_mode
    )
    if args.optimizer == "adam":
        op_cls = op.Adam
    else:
        raise ValueError(f"unsupported optimizer: {args.optimizer}")
    fval_cls = FinegrainedValidator
    fval_kwg = dict(
        model=model,
        processor=processor,
        device=device,
        evaluators=list(filter(None, (
            SpeakerEvaluator(vocabs.speaker),
            DialogStateEvaluator(vocabs),
            PosteriorEvaluator()
        ))),
        report_every=None,
        run_end_report=False,
        progress_stat="loss",
        loss=loss
    )
    if isinstance(model, models.VHDA):
        @dataclass
        class VHDAValidator(VHDAInferencer, fval_cls):
            pass

        fval_cls = VHDAValidator
        fval_kwg.update(dict(
            sample_scale=1.0
        ))
    fval = fval_cls(**fval_kwg)

    gval_cls = GenerativeValidator
    gval_kwg = dict(
        model=model,
        processor=processor,
        batch_size=args.valid_batch_size,
        device=device,
        evaluators=list(filter(None, [
            DistinctEvaluator(vocabs),
            SentLengthEvaluator(vocabs),
            RougeEvaluator(vocabs),
            DialogLengthEvaluator(),
            WordEntropyEvaluator(train_dataset)
        ])),
        report_every=None,
        run_end_report=False,
        beam_size=args.beam_size,
        max_sent_len=args.max_gen_len
    )
    gval = gval_cls(**gval_kwg)

    trainer_cls = Trainer
    trainer_kwargs = dict(
        model=model,
        processor=processor,
        device=device,
        writer=torch.utils.tensorboard.SummaryWriter(
            log_dir=str(args.save_dir)
        ),
        evaluators=list(filter(None, (
            SpeakerEvaluator(vocabs.speaker),
            DialogStateEvaluator(vocabs)
        ))),
        progress_stat="loss",
        display_stats={"loss", "kld", "goal-acc-turn-user",
                       "rouge-l-f1", "conv-mi", "nll", "conv-len"},
        report_every=args.report_every,
        stats_formatter=utils.StatsFormatter(num_cols=3),
        dialog_formatter=utils.DialogTableFormatter(
            max_col_len=50
        ),
        loss=loss,
        optimizer_cls=functools.partial(
            op_cls,
            lr=args.learning_rate
        ),
        grad_clip=args.gradient_clip,
        l2norm=args.l2norm_weight,
        save_dir=pathlib.Path(args.save_dir),
        num_epochs=args.num_epochs,
        fin_valid=fval,
        gen_valid=gval,
        validate_every=args.validate_every,
        early_stop=args.early_stop,
        early_stop_criterion=args.early_stop_criterion,
        early_stop_patience=args.early_stop_patience,
        save_every=args.save_every
    )
    if isinstance(model, models.VHDA):
        @dataclass
        class VHDATrainer(VHDAInferencer, trainer_cls):
            pass

        trainer_cls = VHDATrainer
        trainer_kwargs.update(dict(
            dropout_scale=args.dropout_schedule
        ))
    trainer = trainer_cls(**trainer_kwargs)
    train_dataloader = datasets.create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )
    valid_dataloader = datasets.create_dataloader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    logger.info("commencing training...")
    record = trainer.train(train_dataloader, valid_dataloader)
    logger.info(f"final summary: {pprint.pformat(record.to_short_json())}")
    logger.info("done!")
    return record


def create_parser():
    parser = yaap.Yaap()
    # data options
    parser.add_pth("data-dir", is_dir=True, must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("tests/data/json")),
                   help="Path to the data dir. Must contain 'train.json' and "
                        "'dev.json'.")
    # model options
    parser.add_pth("model-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/vhda-mini.yml")),
                   help="Path to the model configuration file.")
    parser.add_int("gpu", min_bound=0,
                   help="GPU device to use. (e.g. 0, 1, etc.)")
    # display options
    parser.add_pth("logging-config", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/logging.yml")),
                   help="Path to a logging config file (yaml/json).")
    parser.add_pth("save-dir", default="out", is_dir=True,
                   help="Directory to save output files.")
    parser.add_bol("overwrite", help="Whether to overwrite save dir.")
    parser.add_int("report-every",
                   help="Report training statistics every N steps.")
    # training options
    parser.add_int("batch-size", default=32,
                   help="Mini-batch size.")
    parser.add_int("valid-batch-size", default=32,
                   help="Mini-batch sizes for validation inference.")
    parser.add_str("optimizer", default="adam", choices=("adam",),
                   help="Optimizer to use.")
    parser.add_flt("gradient-clip",
                   help="Clip gradients by norm size.")
    parser.add_flt("l2norm-weight",
                   help="Weight of l2-norm regularization.")
    parser.add_flt("learning-rate", default=0.001, min_bound=0,
                   help="Optimizer learning rate.")
    parser.add_int("epochs", default=10, min_bound=1,
                   help="Number of epochs to train for.")
    parser.add_str("kld-schedule",
                   help="KLD w schedule given as a list of data points. Each "
                        "data point is a pair of training step and target "
                        "dropout scale. Steps in-between data points will be "
                        "interpolated. e.g. '[(0, 1.0), (10000, 0.1)]'")
    parser.add_str("dropout-schedule",
                   help="Dropout schedule given as a list of data points. Each "
                        "data point is a pair of training step and target "
                        "dropout scale. Steps in-between data points will be "
                        "interpolated. e.g. '[(0, 1.0), (10000, 0.1)]'")
    parser.add_flt("validate-every", default=1,
                   help="Number of epochs in-between validations.")
    parser.add_bol("early-stop",
                   help="Whether to enable early-stopping.")
    parser.add_str("early-stop-criterion", default="~loss",
                   help="The training statistics key to use as criterion "
                        "for early-stopping. Prefix with '~' to denote "
                        "negation during comparison.")
    parser.add_int("early-stop-patience",
                   help="Number of epochs to wait without breaking "
                        "records until executing early-stopping. "
                        "defaults to infinity.")
    parser.add_int("save-every",
                   help="Number of epochs to wait until saving a model "
                        "checkpoint.")
    # model specific settings
    parser.add_bol("disable-kl",
                   help="Whether to disable kl-divergence term.")
    parser.add_str("kl-mode", default="kl-mi",
                   help="KL mode: one of kl, kl-mi, kl-mi+.")
    parser.add_int("seed", help="Random seed.")
    return parser


def main():
    args = utils.parse_args(create_parser())
    if args.logging_config is not None:
        logging.config.dictConfig(utils.load_yaml(args.logging_config))
    save_dir = pathlib.Path(args.save_dir)
    if (not args.overwrite and
            save_dir.exists() and utils.has_element(save_dir.glob("*.json"))):
        raise FileExistsError(f"save directory ({save_dir}) is not empty")
    shell = utils.ShellUtils()
    shell.mkdir(save_dir, silent=True)
    logger = logging.getLogger("train")
    utils.seed(args.seed)
    logger.info("loading data...")
    load_fn = utils.chain_func(
        lambda data: list(map(Dialog.from_json, data)),
        utils.load_json
    )
    data_dir = pathlib.Path(args.data_dir)
    train_data = load_fn(str(data_dir.joinpath("train.json")))
    valid_data = load_fn(str(data_dir.joinpath("dev.json")))
    processor = datasets.DialogProcessor(
        sent_processor=datasets.SentProcessor(
            bos=True,
            eos=True,
            lowercase=True,
            tokenizer="space",
            max_len=30
        ),
        boc=True,
        eoc=True,
        state_order="randomized",
        max_len=30
    )
    processor.prepare_vocabs(list(itertools.chain(train_data, valid_data)))
    utils.save_pickle(processor, save_dir.joinpath("processor.pkl"))
    logger.info("preparing model...")
    utils.save_json(utils.load_yaml(args.model_path),
                    save_dir.joinpath("model.json"))
    torchmodels.register_packages(models)
    model_cls = torchmodels.create_model_cls(models, args.model_path)
    model: models.AbstractTDA = model_cls(processor.vocabs)
    model.reset_parameters()
    utils.report_model(logger, model)
    device = torch.device("cpu")
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    model = model.to(device)

    def create_scheduler(s):
        return utils.PiecewiseScheduler([utils.Coordinate(*t) for t in eval(s)])

    train_args = TrainArguments(
        model=model,
        train_data=tuple(train_data),
        valid_data=tuple(valid_data),
        processor=processor,
        device=device,
        save_dir=save_dir,
        report_every=args.report_every,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        optimizer=args.optimizer,
        gradient_clip=args.gradient_clip,
        l2norm_weight=args.l2norm_weight,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        kld_schedule=(utils.ConstantScheduler(1.0)
                      if args.kld_schedule is None else
                      create_scheduler(args.kld_schedule)),
        dropout_schedule=(utils.ConstantScheduler(1.0)
                          if args.dropout_schedule is None else
                          create_scheduler(args.dropout_schedule)),
        validate_every=args.validate_every,
        early_stop=args.early_stop,
        early_stop_criterion=args.early_stop_criterion,
        early_stop_patience=args.early_stop_patience,
        disable_kl=args.disable_kl,
        kl_mode=args.kl_mode,
        save_every=args.save_every
    )
    utils.save_json(train_args.to_json(), save_dir.joinpath("args.json"))
    record = train(train_args)
    utils.save_json(record.to_json(), save_dir.joinpath("final-summary.json"))


if __name__ == "__main__":
    with torch.autograd.detect_anomaly():
        main()
