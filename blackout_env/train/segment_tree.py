"""
Array-based segment trees for Prioritized Experience Replay (Schaul et al., 2016):
SumSegmentTree for O(log n) priority-weighted sampling, MinSegmentTree for tracking the
global minimum priority (needed to normalize importance-sampling weights so the max weight
stays <= 1, per the original PER paper).
"""

import operator
from typing import Callable


class SegmentTree:
    def __init__(self, capacity: int, operation: Callable[[float, float], float], neutral_element: float) -> None:
        assert capacity > 0 and (capacity & (capacity - 1)) == 0, "capacity must be a power of 2"
        self._capacity = capacity
        self._operation = operation
        self._value = [neutral_element] * (2 * capacity)

    def __setitem__(self, idx: int, val: float) -> None:
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2 * idx], self._value[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        return self._value[self._capacity + idx]

    def reduce(self, start: int = 0, end: int | None = None) -> float:
        """Applies self._operation over the half-open range [start, end)."""
        if end is None:
            end = self._capacity
        result = None
        start += self._capacity
        end += self._capacity
        while start < end:
            if start % 2 == 1:
                result = self._value[start] if result is None else self._operation(result, self._value[start])
                start += 1
            if end % 2 == 1:
                end -= 1
                result = self._value[end] if result is None else self._operation(result, self._value[end])
            start //= 2
            end //= 2
        return result


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity, operator.add, 0.0)

    def sum(self, start: int = 0, end: int | None = None) -> float:
        return self.reduce(start, end)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """Smallest index i such that sum(0, i+1) > prefixsum. Assumes 0 <= prefixsum < sum()."""
        idx = 1
        while idx < self._capacity:
            left = 2 * idx
            if self._value[left] > prefixsum:
                idx = left
            else:
                prefixsum -= self._value[left]
                idx = left + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity, min, float("inf"))

    def min(self, start: int = 0, end: int | None = None) -> float:
        return self.reduce(start, end)
