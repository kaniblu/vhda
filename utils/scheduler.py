__all__ = ["Coordinate", "Scheduler",
           "ConstantScheduler", "LinearScheduler", "PiecewiseScheduler"]

import bisect
from dataclasses import dataclass
from typing import Sequence


@dataclass
class Coordinate:
    step: int
    value: float

    def __lt__(self, other):
        if not isinstance(other, Coordinate):
            raise TypeError(f"unsupported comparison type: {type(other)}")
        return self.step.__lt__(other.step)

    def __gt__(self, other):
        if not isinstance(other, Coordinate):
            raise TypeError(f"unsupported comparison type: {type(other)}")
        return self.step.__gt__(other.step)

    def __ge__(self, other):
        if not isinstance(other, Coordinate):
            raise TypeError(f"unsupported comparison type: {type(other)}")
        return self.step.__ge__(other.step)

    def __le__(self, other):
        if not isinstance(other, Coordinate):
            raise TypeError(f"unsupported comparison type: {type(other)}")
        return self.step.__le__(other.step)

    def __repr__(self):
        return f"({self.step:,d}, {self.value:.3f})"


class Scheduler:

    def get(self, step: int) -> float:
        raise NotImplementedError


@dataclass
class ConstantScheduler(Scheduler):
    constant: float

    def get(self, step: int) -> float:
        return self.constant


@dataclass
class LinearScheduler(Scheduler):
    start: Coordinate
    end: Coordinate

    def __post_init__(self):
        if self.end.step <= self.start.step:
            raise ValueError(f"ending coordinate must be latter than the "
                             f"starting coordinate: {self.end} <= {self.start}")

    def __repr__(self):
        return f"{self.start} => {self.end}"

    def get(self, step: int):
        if step <= self.start.step:
            return self.start.value
        elif step >= self.end.step:
            return self.end.value
        else:
            return ((step - self.start.step) / (self.end.step - self.start.step)
                    * (self.end.value - self.start.value) + self.start.value)


@dataclass
class PiecewiseScheduler(Scheduler):
    coords: Sequence[Coordinate]

    def __post_init__(self):
        if not self.coords:
            raise ValueError(f"empty sequence")
        self.coords = list(sorted(self.coords, key=lambda x: x.step))

    def get(self, step: int):
        idx = bisect.bisect_left(self.coords, Coordinate(step, 0))
        if idx == 0:
            return self.coords[0].value
        elif idx == len(self.coords):
            return self.coords[-1].value
        return LinearScheduler(self.coords[idx - 1], self.coords[idx]).get(step)

    def __repr__(self):
        return " => ".join(map(repr, self.coords))
