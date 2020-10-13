__all__ = ["RougeEvaluator"]

from dataclasses import dataclass
from typing import Sequence, Optional, List, ClassVar, Mapping

import torch
import rouge

import utils
from utils import TensorMap
from datasets import VocabSet
from ...evaluator import DialogEvaluator


@dataclass
class RougeEvaluator(DialogEvaluator):
    vocabs: VocabSet
    _hyp: List[str] = utils.private_field(default_factory=list)
    _ref: List[str] = utils.private_field(default_factory=list)
    _hyp_spkr: dict = utils.private_field(default_factory=dict)
    _ref_spkr: dict = utils.private_field(default_factory=dict)
    _rouge: rouge.Rouge = utils.private_field(default_factory=rouge.Rouge)
    _key_map: ClassVar[Mapping[str, str]] = {
        "f": "f1",
        "p": "prec",
        "r": "rec"
    }

    def reset(self):
        self._hyp.clear()
        self._ref.clear()
        self._hyp_spkr.clear()
        self._ref_spkr.clear()

    def try_rouge(self, hyp: Sequence[str], ref: Sequence[str]):
        try:
            return {f"{k}-{self._key_map.get(k2, k2)}": v2 for k, v in
                    self._rouge.get_scores(hyp, ref, avg=True).items()
                    for k2, v2 in v.items()}
        except Exception as e:
            return {f"rouge-{k}-{k2}": 0.0
                    for k in ("1", "2", "l") for k2 in ("f1", "prec", "rec")}

    def update(self, samples: Sequence) -> Optional[TensorMap]:
        hyps, refs = list(), list()
        for sample in samples:
            hyp, ref = "", ""
            for hyp_turn in sample.output.turns:
                hyp += " " + hyp_turn.text.strip()
            for ref_turn in sample.input.turns:
                ref += " " + ref_turn.text.strip()
            hyp, ref = hyp.strip(), ref.strip()
            if not ref:
                continue
            if not hyp:
                hyp = "<pad>"
            hyps.append(hyp), refs.append(ref)
        self._hyp.extend(hyps)
        self._ref.extend(refs)
        for spkr in self.vocabs.speaker.f2i:
            if spkr == "<unk>":
                continue
            if spkr not in self._hyp_spkr:
                self._hyp_spkr[spkr] = list()
            if spkr not in self._ref_spkr:
                self._ref_spkr[spkr] = list()
            hyps, refs = list(), list()
            for sample in samples:
                hyp, ref = "", ""
                for hyp_turn in sample.output.turns:
                    if hyp_turn.speaker != spkr:
                        continue
                    hyp += " " + hyp_turn.text.strip()
                for ref_turn in sample.input.turns:
                    if ref_turn.speaker != spkr:
                        continue
                    ref += " " + ref_turn.text.strip()
                hyp, ref = hyp.strip(), ref.strip()
                if not ref:
                    continue
                if not hyp:
                    hyp = "<pad>"
                hyps.append(hyp), refs.append(ref)
            self._hyp_spkr[spkr].extend(hyps)
            self._ref_spkr[spkr].extend(refs)
        return

    def get(self) -> TensorMap:
        stats = self.try_rouge(self._hyp, self._ref)
        for spkr in self.vocabs.speaker.f2i:
            if spkr == "<unk>":
                continue
            stats.update({
                f"{k}-{spkr}": v for k, v in
                self.try_rouge(self._hyp_spkr.get(spkr, list()),
                               self._ref_spkr.get(spkr, list())).items()
            })
        return {k: torch.tensor(v) for k, v in stats.items()}
