__all__ = ["TDAGenerator"]

from dataclasses import dataclass

from .beamsearch import BeamSearchGenerator


@dataclass
class TDAGenerator(BeamSearchGenerator):
    conv_scale: float = 1.0
    spkr_scale: float = 1.0
    goal_scale: float = 1.0
    state_scale: float = 1.0
    sent_scale: float = 1.0
    max_conv_len: int = 20

    def generate_kwargs(self) -> dict:
        kwargs = dict(super().generate_kwargs())
        kwargs.update(dict(
            conv_scale=self.conv_scale,
            spkr_scale=self.spkr_scale,
            goal_scale=self.goal_scale,
            state_scale=self.state_scale,
            sent_scale=self.sent_scale,
            max_conv_len=self.max_conv_len,
            max_sent_len=self.max_sent_len,
            beam_size=self.beam_size
        ))
        return kwargs
