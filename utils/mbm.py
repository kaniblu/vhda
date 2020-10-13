__all__ = ["MaximumBipartiteMatching"]

from dataclasses import dataclass, field
from typing import Sequence, Tuple, Set, Callable

import numpy as np


@dataclass
class MBMResult:
    num_left: int
    num_right: int
    matches: Sequence[Tuple[int, int]]
    matched_left: Set[int] = field(init=False, default=None)
    unmatched_left: Set[int] = field(init=False, default=None)
    matched_right: Set[int] = field(init=False, default=None)
    unmatched_right: Set[int] = field(init=False, default=None)

    def __post_init__(self):
        left, right = set(range(self.num_left)), set(range(self.num_right))
        if self.matches:
            self.matched_left, self.matched_right = map(set, zip(*self.matches))
        else:
            self.matched_left, self.matched_right = set(), set()
        self.unmatched_left = left - self.matched_left
        self.unmatched_right = right - self.matched_right

    def __len__(self):
        return self.matches

    @property
    def is_all_matched(self):
        return not self.unmatched_right and not self.unmatched_left


@dataclass
class MaximumBipartiteMatching:
    num_left: int
    num_right: int
    query_fn: Callable[[int, int], bool]

    def compute(self) -> MBMResult:
        match = np.array([-1] * self.num_right)
        seen = np.array([False] * self.num_right)

        def find_match(i):
            for j in range(self.num_right):
                if not self.query_fn(i, j) or seen[j]:
                    continue
                seen[j] = True
                if match[j] < 0 or find_match(match[j]):
                    match[j] = i
                    return True
            return False

        num_matches = 0
        for i in range(self.num_left):
            seen[:] = False
            if find_match(i):
                num_matches += 1
        return MBMResult(
            num_left=self.num_left,
            num_right=self.num_right,
            matches=[(j, i) for i, j in enumerate(match) if j != -1]
        )
