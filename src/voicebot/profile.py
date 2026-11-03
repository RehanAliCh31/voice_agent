"""Who the bot is: its name and its gender, for every voice and every sentence.

One profile for the whole process. Its gender chooses the voice in both
languages and the way the bot speaks of itself in Urdu, where the verb says
who is speaking: "میں کر سکتی ہوں" is a woman, "میں کر سکتا ہوں" a man.
English barely marks it, so English sentences are the same for both.

The model is told who it is; it never decides. Nothing it writes can change
the voice, and what it writes about itself in Urdu is checked and put right
before it is spoken (voicebot.reply_check).

Set once at startup from `agent:` in configs/assistant.yaml. Until then --
and in tests -- the default stands: female, matching the voices shipped.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

Gender = Literal["female", "male"]
GENDERS: tuple[Gender, ...] = ("female", "male")

# A name for each gender, so changing the gender never leaves a man called
# Layla. Arabic names, for the Arabic-accented voices the profile asks for.
DEFAULT_NAMES: Mapping[Gender, Mapping[str, str]] = {
    "female": {"en": "Layla", "ur": "لیلیٰ"},
    "male": {"en": "Omar", "ur": "عمر"},
}


@dataclass(frozen=True)
class AgentProfile:
    gender: Gender = "female"
    names: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_NAMES["female"]))

    def name(self, language: str | None) -> str:
        return self.names["ur" if language == "ur" else "en"]


_lock = threading.Lock()
_current = AgentProfile()


def current() -> AgentProfile:
    """The profile in force."""
    return _current


def use(profile: AgentProfile) -> AgentProfile:
    """Put `profile` in force for the whole process. Returns the one it replaced."""
    global _current
    with _lock:
        previous, _current = _current, profile
    return previous
