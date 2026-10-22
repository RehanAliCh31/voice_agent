"""Typed data contracts passed between pipeline stages.

No bare dicts cross a stage boundary. Every stage consumes and produces
objects defined here, so a change to one stage's output is a type error
rather than a runtime surprise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

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


@dataclass(frozen=True)
class Slot:
    """A parameter extracted from the user's utterance, with its character span."""

    name: str
    value: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass(frozen=True)
class Understanding:
    """Output of the NLU stage."""

    intent: str
    intent_confidence: float
    slots: tuple[Slot, ...] = ()
    oos_score: float = 0.0
    is_out_of_scope: bool = False


@dataclass
class Turn:
    """One exchange, kept for conversation history."""

    user_text: str
    bot_text: str
    intent: str | None = None


class ActionType(StrEnum):
    """What the bot is doing this turn.

    The frontend renders each of these differently -- CONFIRM draws the
    yes/no buttons, ASK_SLOT shows a prompt, OUT_OF_SCOPE explains the
    bot's limits -- so the type is part of the contract, not a detail.
    """

    ANSWER = "answer"
    ASK_SLOT = "ask_slot"
    CONFIRM = "confirm"
    CLARIFY = "clarify"
    OUT_OF_SCOPE = "out_of_scope"
    CANCELLED = "cancelled"
    GIVE_UP = "give_up"
    FAILED = "failed"


@dataclass(frozen=True)
class BotAction:
    """Output of the dialogue manager: what to say and why."""

    type: ActionType
    text: str
    intent: str | None = None
    filled_slots: dict[str, str] = field(default_factory=dict)
    missing_slots: tuple[str, ...] = ()


@dataclass
class DialogueState:
    """Per-session conversation state owned by the dialogue manager.

    The FSM phase is derived from these fields rather than stored alongside
    them, so the two can never disagree. See `dialogue.states.phase_of`.
    """

    session_id: str
    turn_index: int = 0
    active_intent: str | None = None
    filled_slots: dict[str, str] = field(default_factory=dict)
    pending_confirmation: bool = False
    slot_attempts: int = 0
    history: list[Turn] = field(default_factory=list)
