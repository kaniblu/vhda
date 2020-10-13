__all__ = ["DialogStateEvaluator"]

from dataclasses import dataclass
from typing import List, Optional

import torch

import utils
from datasets import VocabSet
from datasets import BatchData
from utils import TensorMap
from utils import Stacked1DTensor
from utils import DoublyStacked1DTensor
from ..evaluator import FinegrainedEvaluator


def f1_score(prec, rec):
    if prec == 0 and rec == 0:
        return prec.new(1).zero_().squeeze()
    return 2 * prec * rec / (prec + rec)


@dataclass
class DialogStateEvaluator(FinegrainedEvaluator):
    vocabs: VocabSet
    return_update: bool = False
    _pred_goal: List[Stacked1DTensor] = \
        utils.private_field(default_factory=list)
    _gold_goal: List[Stacked1DTensor] = \
        utils.private_field(default_factory=list)
    _pred_state: List[Stacked1DTensor] = \
        utils.private_field(default_factory=list)
    _gold_state: List[Stacked1DTensor] = \
        utils.private_field(default_factory=list)
    _spkr: List[torch.Tensor] = utils.private_field(default_factory=list)

    def compute_accuracy(self, pred: DoublyStacked1DTensor,
                         gold: DoublyStacked1DTensor,
                         turn_mask=None) -> utils.TensorMap:
        batch_size = pred.size(0)
        pred_dense = utils.to_dense(pred.value, pred.lens1,
                                    max_size=len(self.vocabs.goal_state.asv))
        gold_dense = utils.to_dense(gold.value, gold.lens1,
                                    max_size=len(self.vocabs.goal_state.asv))
        crt = (pred_dense == gold_dense).all(-1)
        conv_mask = utils.mask(pred.lens, pred.size(1))
        if turn_mask is None:
            turn_mask = torch.ones_like(conv_mask).bool()
        turn_mask = turn_mask & conv_mask
        crt = crt & turn_mask
        num_turns = turn_mask.sum()
        stats = {
            "acc": (crt | ~turn_mask).all(-1).sum().float() / batch_size,
            "acc-turn": crt.sum().float() / num_turns,
        }
        return stats

    def reset(self):
        self._pred_goal, self._gold_goal = list(), list()
        self._pred_state, self._gold_state = list(), list()
        self._spkr = list()

    def update(self, batch: BatchData, pred: BatchData, outputs
               ) -> Optional[TensorMap]:
        self._pred_goal.extend(pred.goal)
        self._gold_goal.extend(batch.goal)
        self._pred_state.extend(pred.state)
        self._gold_state.extend(batch.state)
        self._spkr.extend(batch.speaker)
        if not self.return_update:
            return
        stats = dict()
        stats.update({f"goal-{k}": v for k, v in
                      self.compute_accuracy(pred.goal, batch.goal).items()})
        stats.update({f"state-{k}": v for k, v in
                      self.compute_accuracy(pred.state, batch.state).items()})
        for spkr_idx, spkr in self.vocabs.speaker.i2f.items():
            if spkr == "<unk>":
                continue
            spkr_value = batch.speaker.value
            stats.update({f"goal-{k}-{spkr}": v for k, v in
                          self.compute_accuracy(
                              pred.goal, batch.goal,
                              spkr_value == spkr_idx
                          ).items()})
            stats.update({f"state-{k}-{spkr}": v for k, v in
                          self.compute_accuracy(
                              pred.state, batch.state,
                              spkr_value == spkr_idx
                          ).items()})
        return stats

    def get(self) -> TensorMap:
        pred_goal = utils.stack_stacked1dtensors(self._pred_goal)
        pred_state = utils.stack_stacked1dtensors(self._pred_state)
        gold_goal = utils.stack_stacked1dtensors(self._gold_goal)
        gold_state = utils.stack_stacked1dtensors(self._gold_state)
        spkr_value = utils.pad_stack(self._spkr).value
        stats = dict()
        stats.update({
            f"goal-{k}": v for k, v in
            self.compute_accuracy(pred_goal, gold_goal).items()
        })
        stats.update({
            f"state-{k}": v for k, v in
            self.compute_accuracy(pred_state, gold_state).items()
        })
        for spkr_idx, spkr in self.vocabs.speaker.i2f.items():
            if spkr == "<unk>":
                continue
            stats.update({f"goal-{k}-{spkr}": v for k, v in
                          self.compute_accuracy(
                              pred_goal, gold_goal,
                              spkr_value == spkr_idx
                          ).items()})
            stats.update({f"state-{k}-{spkr}": v for k, v in
                          self.compute_accuracy(
                              pred_state, gold_state,
                              spkr_value == spkr_idx
                          ).items()})
        return stats
