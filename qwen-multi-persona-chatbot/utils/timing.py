"""
Lightweight timing utilities for measuring model response latency.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class TimerResult:
    """Container for an elapsed time measurement."""

    elapsed_seconds: float

    @property
    def elapsed_milliseconds(self) -> float:
        return self.elapsed_seconds * 1000.0


@contextmanager
def stopwatch() -> Iterator[TimerResult]:
    """
    Context manager that records wall-clock execution time.

    Example
    -------
    with stopwatch() as t:
        model.generate(...)
    print(t.elapsed_seconds)
    """
    start = time.perf_counter()
    result = TimerResult(elapsed_seconds=0.0)
    try:
        yield result
    finally:
        end = time.perf_counter()
        result.elapsed_seconds = end - start
