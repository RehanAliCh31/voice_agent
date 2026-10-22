"""The conversation's finite-state machine.

The phase is DERIVED from DialogueState rather than stored in it. Storing a
phase alongside the fields that imply it gives you two sources of truth and,
eventually, a bug where they disagree.
"""

from __future__ import annotations

from enum import StrEnum

from voicebot.schemas import DialogueState


class Phase(StrEnum):
    """Where the conversation is.

    IDLE        nothing in progress; any in-scope intent starts a task
    COLLECTING  an intent is active and still missing required slots
    CONFIRMING  every slot is filled and the bot is awaiting a yes or no
    """

    IDLE = "idle"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"


def phase_of(state: DialogueState) -> Phase:
    """Read the phase off the state."""
    if state.pending_confirmation:
        return Phase.CONFIRMING
    if state.active_intent is not None:
        return Phase.COLLECTING
    return Phase.IDLE


def reset(state: DialogueState) -> None:
    """Return the conversation to IDLE, clearing any half-finished task."""
    state.active_intent = None
    state.filled_slots = {}
    state.pending_confirmation = False
    state.slot_attempts = 0
