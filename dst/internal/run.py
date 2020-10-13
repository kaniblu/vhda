__all__ = ["Runner", "TestDataloader", "create_parser"]

import copy
import json
import functools
import random
import logging
import logging.config
import pathlib
import pprint
import itertools
import collections
from dataclasses import dataclass
from typing import (Sequence, Tuple, Mapping, Dict,
                    Optional, Iterable, List, Callable)

import numpy as np
import tqdm
import yaap
import torch
import torch.nn as nn
import torch.optim as op
import torch.utils.tensorboard as tb
import torchmodels

import utils
import models
import datasets
from datasets import DSTDialog
from datasets import DSTTurn
from datasets import DialogAct
from datasets import ActSlotValue
from datasets import DialogState
from datasets import PaddingCollator
from . import datasets as dst_datasets
from . import models as dst_models
from .datasets import DSTBatchData
from .datasets import DSTTestBatchData
from .models import dst


@dataclass
class TestDataloader:
    dialogs: Sequence[DSTDialog]
    processor: dst_datasets.DSTDialogProcessor
    max_batch_size: int = 32
    _collator: PaddingCollator = utils.private_field()

    def __post_init__(self):
        self._collator = PaddingCollator(
            frozenset(("sent", "system_acts", "belief_state",
                       "slot", "asr_score")))

    @property
    def total_items(self):
        return sum(len(dialog.dst_turns) for dialog in self.dialogs)

    def __len__(self):
        return len(self.dialogs)

    def create_batch(self, dialogs: Iterable[DSTDialog]):
        items = []
        for dialog in dialogs:
            for turn in dialog.dst_turns:
                items.append(self.processor.tensorize_dst_test_turn(turn))
        return DSTTestBatchData.from_dict(self._collator(items))

    def __iter__(self):
        bucket: List[DSTDialog] = []
        for dialog in self.dialogs:
            if (sum(len(d.dst_turns) for d in bucket) + len(dialog.dst_turns)) \
                    > self.max_batch_size:
                if bucket:
                    yield bucket, self.create_batch(bucket)
                bucket = []
            bucket.append(dialog)
        if bucket:
            yield bucket, self.create_batch(bucket)


@dataclass
class Record:
    epoch: int
    value: float
    stats: Mapping[str, float]
    params: Dict[str, torch.Tensor]

    def to_json(self):
        return {
            "epoch": self.epoch,
            "value": self.value,
            "stats": self.stats
        }


def bucket_items(items: Sequence, lens: Sequence[int]):
    assert len(items) == sum(lens)
    ret = []
    cum_lens = np.cumsum(lens).tolist()
    for i, j in zip([0] + cum_lens, cum_lens):
        ret.append(items[i:j])
    return ret


def harmonic_mean(arr: np.ndarray):
    if (arr == 0).any():
        return 0.0
    return 1 / (1 / arr).mean()


@dataclass
class Runner:
    model: dst.AbstractDialogStateTracker
    processor: dst_datasets.DSTDialogProcessor
    save_dir: pathlib.Path = pathlib.Path("out")
    device: torch.device = torch.device("cpu")
    epochs: int = 30
    scheduler: Callable[[op.Optimizer], op.lr_scheduler._LRScheduler] = None
    loss: str = "sum"
    gradient_clip: Optional[float] = None
    l2norm: Optional[float] = None
    train_validate: bool = False
    inepoch_report_chance: float = 0.1
    early_stop: bool = False
    early_stop_criterion: str = "joint-goal"
    early_stop_patience: Optional[int] = None
    asr_method: str = "score"
    asr_sigmoid_sum_order: str = "sum-sigmoid"
    asr_topk: Optional[int] = None
    _logger: logging.Logger = utils.private_field(default=None)
    _user_tensor: utils.Stacked1DTensor = utils.private_field(default=None)
    _wizard_tensor: utils.Stacked1DTensor = utils.private_field(default=None)
    _bce: nn.BCEWithLogitsLoss = utils.private_field(default=None)
    _record: Record = utils.private_field(default=None)

    def __post_init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._user_tensor = self.processor.tensorize_state_vocab(
            speaker="user",
            # tensorizer=self.processor.tensorize_turn_label_asv
        )
        self._user_tensor = self._user_tensor.to(self.device)
        self._wizard_tensor = self.processor.tensorize_state_vocab(
            speaker="wizard"
        )
        self._wizard_tensor = self._wizard_tensor.to(self.device)
        self._bce = nn.BCEWithLogitsLoss(reduction="none")
        utils.ShellUtils().mkdir(self.save_dir, True)
        assert self.asr_sigmoid_sum_order in {"sigmoid-sum", "sum-sigmoid"}

    @property
    def vocabs(self):
        return self.processor.vocabs

    @property
    def criterion(self):
        return (self.early_stop_criterion.lstrip("~"),
                self.early_stop_criterion.startswith("~"))

    def prepare_system_acts(self, s: utils.Stacked1DTensor):
        s, s_lens = s.value, s.lens
        if s_lens.max().item() == 0:
            return (self._wizard_tensor.value[s], s_lens,
                    self._wizard_tensor.lens[s])
        s_lats = s_lens
        s, s_lens = self._wizard_tensor.value[s], self._wizard_tensor.lens[s]
        return s[..., :s_lens.max()], s_lats, s_lens

    def compute_loss(self, batch: DSTBatchData, ontology):
        loss = None
        for act_slot, (ont_idx, ont) in ontology.items():
            as_idx = torch.tensor(self.vocabs.speaker_state["user"]
                                  .act_slot[act_slot]).to(self.device)
            ont_idx, ont = ont_idx.to(self.device), ont.to(self.device)
            logit = self.model(
                as_idx,
                *batch.sent.tensors,
                *self.prepare_system_acts(batch.system_acts),
                *ont.tensors
            )
            # ont_idx: [num_ont] -> [batch_size x num_ont x state_lat]
            # s: [batch_size x state_lat] ->
            #    [batch_size x num_ont x state_lat]
            # target: [batch_size x num_ont]
            s = batch.belief_state
            target = \
                ((ont_idx.unsqueeze(0).unsqueeze(-1) == s.value.unsqueeze(1))
                 .masked_fill(~utils.mask(s.lens).unsqueeze(1), 0).any(-1))
            current_loss = self._bce(logit, target.float())
            if self.loss == "mean":
                current_loss = current_loss.mean(-1)
            elif self.loss == "sum":
                current_loss = current_loss.sum(-1)
            else:
                raise ValueError(f"unsupported loss method: {self.loss}")
            if loss is None:
                loss = current_loss
            else:
                loss += current_loss
        return loss

    def make_record(self, epoch, stats):
        self._record = Record(
            epoch=epoch,
            value=stats.get(self.criterion[0], None),
            stats=stats,
            params={k: v.cpu().detach()
                    for k, v in self.model.state_dict().items()}
        )

    def predict(self, batch, ontology):
        pred = [list() for _ in range(batch.batch_size)]
        loss = None
        for act_slot, (ont_idx, ont) in ontology.items():
            as_idx = torch.tensor(self.vocabs.speaker_state["user"]
                                  .act_slot[act_slot]).to(self.device)
            ont_idx, ont = ont_idx.to(self.device), ont.to(self.device)
            logit = self.model(
                as_idx,
                *batch.sent.tensors,
                *self.prepare_system_acts(batch.system_acts),
                *ont.tensors
            )
            # ont_idx: [num_ont] -> [batch_size x num_ont x state_lat]
            # s: [batch_size x state_lat] ->
            #    [batch_size x num_ont x state_lat]
            # target: [batch_size x num_ont]
            s = batch.belief_state
            target = \
                ((ont_idx.unsqueeze(0).unsqueeze(-1) == s.value.unsqueeze(1))
                 .masked_fill(~utils.mask(s.lens).unsqueeze(1), 0).any(-1))
            current_loss = self._bce(logit, target.float())
            if self.loss == "mean":
                current_loss = current_loss.mean(-1)
            elif self.loss == "sum":
                current_loss = current_loss.sum(-1)
            else:
                raise ValueError(f"unsupported loss method: {self.loss}")
            if loss is None:
                loss = current_loss
            else:
                loss += current_loss
            for batch_idx, val_idx in \
                    (torch.sigmoid(logit) > 0.5).nonzero().tolist():
                pred[batch_idx].append(
                    (ont_idx[val_idx].item(), logit[batch_idx, val_idx]))

        def to_dialog_state(data: Sequence[Tuple[ActSlotValue, float]]):
            state = DialogState()
            as_map = collections.defaultdict(list)
            for asv, score in data:
                as_map[(asv.act, asv.slot)].append((asv, score))
            for (act, slt), data in as_map.items():
                if act == "request" and slt == "slot":
                    state.update(asv for asv, _ in data)
                elif act == "inform":
                    state.add(max(data, key=lambda x: x[1])[0])
            return state

        pred = [[(self.processor.vocabs.speaker_state["user"].asv[idx], score)
                 for idx, score in v] for v in pred]
        pred = list(map(to_dialog_state, pred))
        pred_inform = [{sv.slot: sv.value for sv in p.get("inform")}
                       for p in pred]
        pred_request = [{sv.value for sv in p.get("request")} for p in pred]
        # DSTC2: 'this' resolution
        pred = [
            (DSTTurn(turn.wizard, turn.user.clone(inform=pi, request=pr))
             .resolve_this().user.state)
            for turn, pi, pr in zip(batch.raw, pred_inform, pred_request)
        ]
        return loss, pred

    def predict_asr(self, batch, ontology):
        pred = [list() for _ in range(batch.batch_size)]
        batch_loss = []
        for batch_idx, (batch_asr, score) in enumerate(self.iter_asr(batch)):
            loss = None
            if self.asr_topk is not None:
                score_list = sorted(enumerate(score.tolist()),
                                    key=lambda x: x[1],
                                    reverse=True)
                score_list = list(utils.bucket(
                    score_list,
                    compare_fn=lambda x, y: x[1] == y[1]))
                score_list = list(itertools.chain(*score_list[:self.asr_topk]))
                score_idx = [idx for idx, _ in score_list]
                score = score[score_idx]
                batch_asr = batch_asr[score_idx]
            if self.asr_method == "uniform":
                score.fill_(1 / len(score))
            elif self.asr_method == "ones":
                score.fill_(1)
            elif self.asr_method == "scaled":
                max_score = score.max()
                if max_score.item() == 0:
                    score.fill_(1 / len(score))
                else:
                    score = score * (1 / max_score.item())
            elif self.asr_method == "score":
                if score.sum().item() == 0:
                    score.fill_(1 / len(score))
                else:
                    score = score / score.sum()
            else:
                raise ValueError(f"unsupported method: {self.asr_method}")
            for act_slot, (ont_idx, ont) in ontology.items():
                as_idx = torch.tensor(self.vocabs.speaker_state["user"]
                                      .act_slot[act_slot]).to(self.device)
                ont_idx, ont = ont_idx.to(self.device), ont.to(self.device)
                logit = logit_raw = self.model(
                    as_idx,
                    *batch_asr.sent.tensors,
                    *self.prepare_system_acts(batch_asr.system_acts),
                    *ont.tensors
                )
                logit = torch.mm(score.unsqueeze(0), logit).squeeze(0)
                # ont_idx: [num_ont] -> [num_ont x state_lat]
                # s: [state_lat] -> [num_ont x state_lat]
                # target: [num_ont]
                s = batch_asr.belief_state[0]
                target = ((ont_idx.unsqueeze(-1) == s.unsqueeze(0)).any(-1))
                current_loss = self._bce(logit, target.float())
                if self.loss == "mean":
                    current_loss = current_loss.mean(-1)
                elif self.loss == "sum":
                    current_loss = current_loss.sum(-1)
                else:
                    raise ValueError(f"unsupported loss method: {self.loss}")
                if loss is None:
                    loss = current_loss
                else:
                    loss += current_loss
                if self.asr_sigmoid_sum_order == "sum-sigmoid":
                    current_pred = torch.sigmoid(logit) > 0.5
                elif self.asr_sigmoid_sum_order == "sigmoid-sum":
                    sigmoid = (torch.mm(score.unsqueeze(0),
                                        torch.sigmoid(logit_raw))
                               .squeeze(0).clamp_(0.0, 1.0))
                    current_pred = sigmoid > 0.5
                    logit = (sigmoid / (1 - sigmoid)).log()
                else:
                    raise ValueError(f"unsupported order: "
                                     f"{self.asr_sigmoid_sum_order}")
                for (val_idx,) in current_pred.nonzero().tolist():
                    pred[batch_idx] \
                        .append((ont_idx[val_idx].item(), logit[val_idx]))
            batch_loss.append(loss)

        def to_dialog_state(data: Sequence[Tuple[ActSlotValue, float]]):
            state = DialogState()
            as_map = collections.defaultdict(list)
            for asv, score in data:
                as_map[(asv.act, asv.slot)].append((asv, score))
            for (act, slt), data in as_map.items():
                if act == "request" and slt == "slot":
                    state.update(asv for asv, _ in data)
                elif act == "inform":
                    state.add(max(data, key=lambda x: x[1])[0])
            return state

        pred = [[(self.processor.vocabs.speaker_state["user"].asv[idx], score)
                 for idx, score in v] for v in pred]
        pred = list(map(to_dialog_state, pred))
        pred_inform = [{sv.slot: sv.value for sv in p.get("inform")}
                       for p in pred]
        pred_request = [{sv.value for sv in p.get("request")} for p in pred]
        # DSTC2: 'this' resolution
        pred = [
            (DSTTurn(turn.wizard, turn.user.clone(inform=pi, request=pr))
             .resolve_this().user.state)
            for turn, pi, pr in zip(batch.raw, pred_inform, pred_request)
        ]
        return torch.stack(batch_loss), pred

    def train(self, train_dataloader, dev_dataloader, test_fn=None):
        test_fn = test_fn or self.test
        writer = tb.SummaryWriter(log_dir=str(self.save_dir))
        ont = self.processor.tensorize_state_dict(self._user_tensor, "user")
        optimizer = op.Adam(p for p in self.model.parameters()
                            if p.requires_grad)
        scheduler = None
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
        global_step = 0
        final_stats = None
        for eidx in range(1, self.epochs + 1):
            progress_local = tqdm.tqdm(
                total=len(train_dataloader.dataset),
                dynamic_ncols=True,
                desc=f"training epoch-{eidx}"
            )
            cum_stats = collections.defaultdict(float)
            local_step = 0
            self.model.train()
            for batch in train_dataloader:
                batch = batch.to(self.device)
                batch_size = batch.batch_size
                global_step += batch_size
                local_step += batch_size
                progress_local.update(batch_size)
                optimizer.zero_grad()
                loss = self.compute_loss(batch, ont)
                loss_mean = loss.mean()
                stats = {"train-loss": loss_mean.item()}
                if self.l2norm is not None:
                    l2norm = sum(p.norm(2) for p in
                                 self.model.parameters() if p.requires_grad)
                    loss_mean += self.l2norm * l2norm
                if self.gradient_clip is not None:
                    nn.utils.clip_grad_norm_(
                        parameters=itertools.chain(
                            *(d["params"] for d in optimizer.param_groups)),
                        max_norm=self.gradient_clip
                    )
                loss_mean.backward()
                optimizer.step()
                for k, v in stats.items():
                    cum_stats[k] += v * batch_size
                if self.inepoch_report_chance >= random.random():
                    for k, v in stats.items():
                        writer.add_scalar(k, v, global_step)
                    progress_local.set_postfix({"loss": loss_mean.item()})
            stats = {f"{k}-epoch": v / local_step for k, v in cum_stats.items()}
            progress_local.close()
            self._logger.info(f"epoch {eidx} train summary:")
            self._logger.info(
                f"\n"
                f"  * train-loss: {stats['train-loss-epoch']:.4f}"
            )
            if self.train_validate:
                with torch.no_grad():
                    train_stats = test_fn(TestDataloader(
                        dialogs=train_dataloader.dataset.dialogs,
                        processor=self.processor,
                        max_batch_size=train_dataloader.batch_size
                    ), ont, "train-validate")
                stats.update({f"val-train-{k}": v
                              for k, v in train_stats.items()})
                self._logger.info(f"epoch {eidx} train-validation summary:")
                self._logger.info(
                    f"\n"
                    f"  * val-train-loss: {train_stats['loss']:.4f}\n"
                    f"  * val-hmean: {train_stats['hmean-inform']:.4f}\n"
                    f"  * val-train-joint: {train_stats['joint-goal']:.4f}\n"
                    f"  * val-train-inform: {train_stats['turn-inform']:.4f}\n"
                    f"  * val-train-request: {train_stats['turn-request']:.4f}"
                )
            with torch.no_grad():
                val_stats = test_fn(dev_dataloader, ont, "validate")
            self._logger.info(f"epoch {eidx} validation summary:")
            self._logger.info(
                f"\n"
                f"  * val-loss: {val_stats['loss']:.4f}\n"
                f"  * val-hmean: {val_stats['hmean-inform']:.4f}\n"
                f"  * val-joint: {val_stats['joint-goal']:.4f}\n"
                f"  * val-inform: {val_stats['turn-inform']:.4f}\n"
                f"  * val-request: {val_stats['turn-request']:.4f}"
            )
            stats.update({f"val-{k}": v for k, v in val_stats.items()})
            if self.early_stop:
                if self._record is None:
                    self.make_record(eidx, val_stats)
                elif (self.early_stop_patience is not None and eidx >
                      self._record.epoch + self.early_stop_patience):
                    break
                else:
                    crit, neg = self.criterion
                    crit_value = val_stats[crit]
                    if (crit_value > self._record.value) != neg:
                        self._logger.info(f"new record made! "
                                          f"{crit}={crit_value:.3f}")
                        self.make_record(eidx, val_stats)
            for k, v in stats.items():
                writer.add_scalar(k, v, global_step)
            final_stats = stats
            if scheduler is not None:
                scheduler.step()
        if self._record is None and final_stats is not None:
            self.make_record(self.epochs, final_stats)
        if self._record is not None:
            self.model.load_state_dict(self._record.params)
        return self._record

    @staticmethod
    def evaluate_batch(pred: Sequence[DialogState],
                       gold: Sequence[DialogState]):
        pred_inform = [s.get("inform") for s in pred]
        gold_inform = [s.get("inform") for s in gold]
        pred_request = [s.get("request") for s in pred]
        gold_request = [s.get("request") for s in gold]

        def all_eq(p, q):
            return [x == y for x, y in zip(p, q)]

        stats = {
            "turn-acc": np.mean(all_eq(pred, gold)),
            "turn-inform": np.mean(all_eq(pred_inform, gold_inform)),
            "turn-request": np.mean(all_eq(pred_request, gold_request))
        }
        stats["hmean-inform"] = harmonic_mean(np.array([
            stats["joint-goal"],
            stats["turn-inform"]
        ]))
        return stats

    @staticmethod
    def evaluate_dialog(pred: Sequence[DialogState],
                        gold: Sequence[DialogState]):
        pred_inform = [s.get("inform") for s in pred]
        gold_inform = [s.get("inform") for s in gold]
        pred_request = [s.get("request") for s in pred]
        gold_request = [s.get("request") for s in gold]

        def cum_sum(data: Sequence[DialogAct]):
            goal = dict()
            ret = []
            for da in data:
                for sv in da:
                    goal[sv.slot] = sv.value
                ret.append(copy.copy(goal))
            return ret

        pred_goal = cum_sum(pred_inform)
        gold_goal = cum_sum(gold_inform)

        def all_eq(p, q):
            return [x == y for x, y in zip(p, q)]

        stats = {
            "joint-goal": np.mean(all_eq(pred_goal, gold_goal)),
            "turn-acc": np.mean(all_eq(pred, gold)),
            "turn-inform": np.mean(all_eq(pred_inform, gold_inform)),
            "turn-request": np.mean(all_eq(pred_request, gold_request))
        }
        stats["hmean-inform"] = harmonic_mean(np.array([
            stats["joint-goal"],
            stats["turn-inform"]
        ]))
        return stats

    def test(self, dataloader: TestDataloader, ontology=None, mode="test"):
        self.model.eval()
        if ontology is None:
            ontology = (self.processor
                        .tensorize_state_dict(self._user_tensor, "user"))
        cum_stats = collections.defaultdict(float)
        progress = tqdm.tqdm(
            total=dataloader.total_items,
            dynamic_ncols=True,
            desc=mode
        )
        local_step = 0
        for dialogs, batch in dataloader:
            batch = batch.to(self.device)
            batch_size = batch.batch_size
            local_step += batch_size
            progress.update(batch_size)
            loss, pred = self.predict(batch, ontology)
            loss_mean = loss.mean()
            pred = bucket_items(pred, list(len(d.dst_turns) for d in dialogs))
            gold = [[turn.resolve_this().user.state
                     for turn in dialog.dst_turns] for dialog in dialogs]
            for p, g in zip(pred, gold):
                for k, v in self.evaluate_dialog(p, g).items():
                    cum_stats[k] += v * len(p)
            cum_stats["loss"] += loss_mean.item() * batch_size
        progress.close()
        return {k: v / local_step for k, v in cum_stats.items()}

    @staticmethod
    def iter_asr(batch: DSTTestBatchData) -> Iterable[DSTBatchData]:
        asr, asr_score = batch.asr, batch.asr_score
        assert (utils.compare_tensors(asr.lens, asr_score.lens)
                and asr.size(0) == asr_score.size(0))
        for i in range(asr.size(0)):
            asr_item = asr[i]
            asr_score_item = asr_score[i]
            num_asr = asr_item.size(0)
            yield DSTBatchData(
                sent=asr_item,
                system_acts=utils.pad_stack([batch.system_acts[i]] * num_asr),
                belief_state=utils.pad_stack([batch.belief_state[i]] * num_asr),
                slot=utils.pad_stack([batch.slot[i]] * num_asr),
                raw=[batch.raw[i]] * num_asr
            ), asr_score_item

    def test_asr(self, dataloader: TestDataloader, ontology=None, mode="test"):
        self.model.eval()
        if ontology is None:
            ontology = (self.processor
                        .tensorize_state_dict(self._user_tensor, "user"))
        cum_stats = collections.defaultdict(float)
        progress = tqdm.tqdm(
            total=dataloader.total_items,
            dynamic_ncols=True,
            desc=mode
        )
        local_step = 0
        for dialogs, batch in dataloader:
            batch = batch.to(self.device)
            batch_size = batch.batch_size
            local_step += batch_size
            progress.update(batch_size)
            loss, pred = self.predict_asr(batch, ontology)
            loss_mean = loss.mean()
            pred = bucket_items(pred, list(len(d.dst_turns) for d in dialogs))
            gold = [[turn.resolve_this().user.state
                     for turn in dialog.dst_turns] for dialog in dialogs]
            for p, g in zip(pred, gold):
                for k, v in self.evaluate_dialog(p, g).items():
                    cum_stats[k] += v * len(p)
            cum_stats["loss"] += loss_mean.item() * batch_size
            progress.set_postfix({
                "joint-goal": cum_stats["joint-goal"] / local_step})
        progress.close()
        return {k: v / local_step for k, v in cum_stats.items()}


def create_parser():
    parser = yaap.Yaap()
    parser.add_pth("logging-config", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("../../tests/config/logging.yml")),
                   help="Path to a logging configuration file.")
    parser.add_pth("data-dir", is_dir=True, must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("../../tests/data/json")),
                   help="Path to a json-format dialogue dataset.")
    parser.add_pth("save-dir", is_dir=True, default="out",
                   help="Saving directory.")
    parser.add_bol("overwrite", help="Whether to overwrite.")
    parser.add_pth("model-path", must_exist=True,
                   default=(pathlib.Path(__file__).absolute().parent
                            .joinpath("configs/glad-mini.yml")))
    parser.add_int("epochs", default=10, help="Number of epochs")
    parser.add_int("batch-size", default=32, help="Batch size.")
    parser.add_bol("scheduled-lr", help="Enable scheduled learning rate.")
    parser.add_str("scheduler-cls", default="StepLR",
                   help="Name of the scheduler class under "
                        "`torch.optim.lr_scheduler` package.")
    parser.add_str("scheduler-kwargs",
                   default="{\"step\": 10, \"gamma\": 0.8}",
                   help="Keyword arguments for the scheduler class, given "
                        "as a serialized json dictionary.")
    parser.add_str("loss", default="sum", choices=("sum", "mean"),
                   help="Type of loss aggregation ('sum' or 'mean').")
    parser.add_flt("l2norm",
                   help="Weight of l2norm regularization.")
    parser.add_flt("gradient-clip",
                   help="Clipping bounds for gradients.")
    parser.add_bol("train-validate",
                   help="Whether to validate on the training set as well.")
    parser.add_bol("early-stop", help="Whether to early stop.")
    parser.add_str("early-stop-criterion", default="joint-goal",
                   help="Early stopping criterion.")
    parser.add_int("early-stop-patience",
                   help="Number of epochs to wait until early stopped.")
    parser.add_bol("validate-asr",
                   help="Whether to use asr information during validation.")
    parser.add_bol("test-asr",
                   help="Whether to use asr information during testing.")
    parser.add_str("asr-method", default="scaled",
                   choices=("score", "uniform", "ones", "scaled"),
                   help="Type of aggregation method to use when summing output "
                        "scores during asr-enabled evaluation.")
    parser.add_str("asr-sigmoid-sum-order", default="sigmoid-sum",
                   help="The order of sum and sigmoid operations in ASR mode.")
    parser.add_int("asr-topk", min_bound=1,
                   help="Number of top-k candidates.")
    parser.add_bol("save-ckpt",
                   help="Whether to save the final checkpoint. "
                        "If enabled, it will be saved as 'ckpt.pth'"
                        "under save_dir.")
    parser.add_int("gpu", help="gpu device id.")
    return parser


def main():
    parser = create_parser()
    args = utils.parse_args(parser)
    if args.logging_config is not None:
        logging.config.dictConfig(utils.load_yaml(args.logging_config))
    logger = logging.getLogger("train")
    save_dir = pathlib.Path(args.save_dir)
    if (not args.overwrite and save_dir.exists() and
            utils.has_element(save_dir.glob("*"))):
        raise FileExistsError(f"save directory ({save_dir}) is not empty")
    save_dir.mkdir(exist_ok=True, parents=True)
    utils.save_yaml(vars(args), save_dir.joinpath("args.yml"))
    logger.info("preparing dataset...")
    data_dir = pathlib.Path(args.data_dir)
    data = {split: utils.load_json(data_dir.joinpath(f"{split}.json"))
            for split in ("train", "dev", "test")}
    data = {split: [datasets.DSTDialog.from_dialog(datasets.Dialog.from_json(d))
                    for d in dialogs]
            for split, dialogs in data.items()}
    logger.info("verifying dataset...")
    for split, dialogs in data.items():
        for dialog in dialogs:
            dialog.validate()
    processor = dst_datasets.DSTDialogProcessor(
        sent_processor=datasets.SentProcessor(
            bos=True,
            eos=True,
            lowercase=True,
            max_len=30
        )
    )
    processor.prepare_vocabs(
        list(itertools.chain(*(data["train"], data["dev"], data["test"]))))
    logger.info("saving processor object...")
    utils.save_pickle(processor, save_dir.joinpath("processor.pkl"))
    train_dataset = dst_datasets.DSTDialogDataset(
        dialogs=data["train"],
        processor=processor
    )
    train_dataloader = dst_datasets.create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    dev_dataloader = TestDataloader(
        dialogs=data["dev"],
        processor=processor,
        max_batch_size=args.batch_size
    )
    test_dataloader = TestDataloader(
        dialogs=data["test"],
        processor=processor,
        max_batch_size=args.batch_size
    )
    logger.info("preparing model...")
    torchmodels.register_packages(models)
    torchmodels.register_packages(dst_models)
    model_cls = torchmodels.create_model_cls(dst, args.model_path)
    model: dst.AbstractDialogStateTracker = model_cls(processor.vocabs)
    if args.gpu is None:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")
    model.reset_parameters()
    model = model.to(device)
    logger.info(str(model))
    logger.info(f"number of parameters: {utils.count_parameters(model):,d}")
    logger.info("preparing trainer...")
    runner = Runner(
        model=model,
        processor=processor,
        device=device,
        save_dir=save_dir,
        epochs=args.epochs,
        scheduler=(None if not args.scheduled_lr else
                   functools.partial(
                       getattr(op.lr_scheduler, args.scheduler_cls),
                       **json.loads(args.scheduler_kwargs)
                   )),
        loss=args.loss,
        l2norm=args.l2norm,
        gradient_clip=args.gradient_clip,
        train_validate=args.train_validate,
        early_stop=args.early_stop,
        early_stop_criterion=args.early_stop_criterion,
        early_stop_patience=args.early_stop_patience,
        asr_method=args.asr_method,
        asr_sigmoid_sum_order=args.asr_sigmoid_sum_order,
        asr_topk=args.asr_topk
    )
    logger.info("commencing training...")
    record = runner.train(
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        test_fn=runner.test_asr if args.validate_asr else None
    )
    logger.info("final summary: ")
    logger.info(pprint.pformat(record.to_json()))
    utils.save_json(record.to_json(), save_dir.joinpath("summary-final.json"))
    logger.info("commencing testing...")
    with torch.no_grad():
        eval_results = runner.test(test_dataloader)
    logger.info("test results: ")
    logger.info(pprint.pformat(eval_results))
    if args.test_asr:
        logger.info("commencing testing (asr)...")
        with torch.no_grad():
            eval_results = runner.test_asr(test_dataloader)
        logger.info("test(asr) results: ")
        logger.info(pprint.pformat(eval_results))
    eval_results["epoch"] = int(record.epoch)
    eval_results["criterion"] = record.value
    if args.save_ckpt:
        logger.info("saving checkpoint...")
        torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                   save_dir.joinpath("ckpt.pth"))
    logger.info("done!")
    utils.save_json(eval_results, save_dir.joinpath("eval.json"))


if __name__ == "__main__":
    main()
