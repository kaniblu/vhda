__all__ = ["BeamSearchGenerator"]

from dataclasses import dataclass

from .generator import Generator


@dataclass
class BeamSearchGenerator(Generator):
    beam_size: int = 8
    max_sent_len: int = 30

    def generate_kwargs(self) -> dict:
        kwargs = dict(super().generate_kwargs())
        kwargs["beam_size"] = self.beam_size
        kwargs["max_sent_len"] = self.max_sent_len
        return kwargs
