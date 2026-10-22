"""Per-stage timing.

Every turn carries its own breakdown rather than a single total. A turn that
takes two seconds is only actionable if you know which stage spent them, and
the inspector panel in the frontend shows exactly this.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LatencyBreakdown:
    """Milliseconds spent in each stage, plus the total."""

    stages: dict[str, float]
    total_ms: float

    def slowest(self) -> tuple[str, float] | None:
        """The stage that cost the most, for logging and for the UI."""
        if not self.stages:
            return None
        name = max(self.stages, key=lambda key: self.stages[key])
        return name, self.stages[name]

    def as_dict(self) -> dict[str, float]:
        return {**self.stages, "total": self.total_ms}


@dataclass
class LatencyTracker:
    """Collects stage timings for one turn.

    Timed with `perf_counter`, which is monotonic: a clock adjustment part way
    through a turn cannot produce a negative duration.
    """

    stages: dict[str, float] = field(default_factory=dict)
    _started: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one stage. Records even when the body raises."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = (time.perf_counter() - start) * 1000.0

    def breakdown(self) -> LatencyBreakdown:
        """Snapshot the timings so far.

        The total is wall-clock from construction, not the sum of the stages:
        anything spent between them belongs in the number the user feels.
        """
        total = (time.perf_counter() - self._started) * 1000.0
        return LatencyBreakdown(stages=dict(self.stages), total_ms=total)
