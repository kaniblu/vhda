import pathlib
import logging
from dataclasses import dataclass
from typing import Sequence, Mapping

import utils
from ..common import Dialog
from ..common import DialogState
from ..common import ActSlotValue


@dataclass
class DataAdapter:
    _logger: logging.Logger = utils.private_field(default=None)

    def __post_init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def serialize_semantics(state: DialogState):
        def serialize_asv(asv: ActSlotValue):
            return {
                "slots": [[asv.slot, asv.value]],
                "act": asv.act
            }

        return list(map(serialize_asv, state))

    @staticmethod
    def parse_semantics(data) -> DialogState:
        state = DialogState()
        for asv_data in data:
            for s, v in asv_data["slots"]:
                state.add(ActSlotValue(
                    act=str(asv_data["act"]).strip(),
                    slot=str(s).strip(),
                    value=str(v).strip()
                ))
        return state

    def load(self, path: pathlib.Path, split: str = None
             ) -> Mapping[str, Sequence[Dialog]]:
        """Use `split` option to specify specific split to be loaded.
        The underlying implementation might not choose to support this."""
        raise NotImplementedError

    def save_imp(self, dat: Mapping[str, Sequence[Dialog]], path: pathlib.Path):
        """Save data into the directory/path.
        May assume that the path is clean."""
        raise NotImplementedError

    def save(self, data: Mapping[str, Sequence[Dialog]],
             path: pathlib.Path, overwrite: bool = False):
        if ((path.is_file() and path.exists() or
             path.is_dir() and utils.has_element(path.glob("*")))
                and not overwrite):
            raise FileExistsError(f"file exists or directory is "
                                  f"not empty: {path}")
        shell = utils.ShellUtils()
        shell.remove(path, recursive=True, silent=True)
        return self.save_imp(data, path)
