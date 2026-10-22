"""Typed data contracts passed between pipeline stages.

No bare dicts cross a stage boundary. Every stage consumes and produces
objects defined here, so a change to one stage's output is a type error
rather than a runtime surprise.

Only audio and transcripts live here now. The dialogue types -- intents,
slots, actions, conversation state -- went with the trained model and the
state machine that used them; the agent's own contracts are in
`voicebot.agent.loop` and `voicebot.agent.executor`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# eq=False because NumPy arrays have no scalar __eq__; identity comparison is correct here.
@dataclass(frozen=True, eq=False)
class AudioBuffer:
    """Mono PCM audio as float32 samples in [-1, 1]."""

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise ValueError(f"AudioBuffer must be mono (1-D), got shape {self.samples.shape}")
        if self.sample_rate <= 0:
            raise ValueError(f"AudioBuffer sample_rate must be positive, got {self.sample_rate}")

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True)
class Segment:
    """A timestamped span of recognised speech."""

    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class Transcript:
    """Output of the ASR stage."""

    text: str
    confidence: float
    duration_s: float
    segments: tuple[Segment, ...] = ()
