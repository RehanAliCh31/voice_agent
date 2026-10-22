"""The orchestrator and its timing, driven entirely by the fake engines.

This is what the fakes were built for in L0: a full turn, audio in to audio
out, with no model, no GPU and no network.
"""

import numpy as np
import pytest

from voicebot.asr.base import FakeASREngine
from voicebot.dialogue.manager import DialogueManager
from voicebot.dialogue.mock_backend import MockBank
from voicebot.nlu.base import NLUEngine
from voicebot.pipeline.latency import LatencyTracker
from voicebot.pipeline.orchestrator import VoicePipeline
from voicebot.schemas import (
    ActionType,
    AudioBuffer,
    DialogueState,
    Slot,
    Understanding,
)
from voicebot.tts.base import FakeTTSEngine


class ScriptedNLU(NLUEngine):
    """Returns a prepared Understanding, and records what it was asked."""

    def __init__(self, understanding: Understanding):
        self.understanding = understanding
        self.seen: list[str] = []

    def understand(self, text: str) -> Understanding:
        self.seen.append(text)
        return self.understanding


def _speech(seconds: float = 1.0, rate: int = 16_000) -> AudioBuffer:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
    return AudioBuffer(
        samples=(0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sample_rate=rate
    )


def _silence(seconds: float = 1.0, rate: int = 16_000) -> AudioBuffer:
    return AudioBuffer(samples=np.zeros(int(rate * seconds), dtype=np.float32), sample_rate=rate)


def _understanding(intent: str, confidence: float = 0.95, **slots) -> Understanding:
    return Understanding(
        intent=intent,
        intent_confidence=confidence,
        slots=tuple(Slot(name=k, value=v, start=0, end=0) for k, v in slots.items()),
    )


def _pipeline(heard: str, understanding: Understanding, speak: bool = True) -> VoicePipeline:
    return VoicePipeline(
        asr=FakeASREngine(fixed_text=heard),
        nlu=ScriptedNLU(understanding),
        tts=FakeTTSEngine(),
        manager=DialogueManager(bank=MockBank()),
        speak=speak,
    )


# --- a full turn -------------------------------------------------------------


def test_a_complete_turn_produces_text_understanding_action_and_audio():
    pipe = _pipeline(
        "What is my checking balance?", _understanding("balance", account_type="checking")
    )
    result = pipe.converse(_speech(), DialogueState(session_id="t"))

    assert result.text == "what is my checking balance"
    assert result.understanding.intent == "balance"
    assert result.action.type is ActionType.ANSWER
    assert "2,483.50" in result.action.text
    assert result.audio.duration_s > 0


def test_the_nlu_receives_normalised_text_not_the_raw_transcript():
    """The whole reason the normalizer sits between them."""
    nlu = ScriptedNLU(_understanding("transfer", amount="50 dollars", payee="sarah"))
    pipe = VoicePipeline(
        asr=FakeASREngine(fixed_text="Transfer $50 to Sarah!"),
        nlu=nlu,
        tts=FakeTTSEngine(),
        manager=DialogueManager(bank=MockBank()),
    )
    pipe.converse(_speech(), DialogueState(session_id="t"))
    assert nlu.seen == ["transfer 50 dollars to sarah"]


def test_state_carries_across_turns():
    pipe = _pipeline("transfer money", _understanding("transfer"))
    state = DialogueState(session_id="t")

    first = pipe.converse(_speech(), state)
    assert first.action.type is ActionType.ASK_SLOT
    assert state.active_intent == "transfer"

    second = pipe.converse(_speech(), state)
    assert state.turn_index == 2
    assert len(state.history) == 2
    assert second.action.type is ActionType.ASK_SLOT


# --- nothing was said --------------------------------------------------------


def test_silence_never_reaches_the_nlu():
    nlu = ScriptedNLU(_understanding("transfer"))
    pipe = VoicePipeline(asr=FakeASREngine(fixed_text="anything"), nlu=nlu, tts=FakeTTSEngine())
    result = pipe.converse(_silence(), DialogueState(session_id="t"))

    assert result.action.type is ActionType.CLARIFY
    assert nlu.seen == [], "the NLU was asked to classify silence"


def test_a_transcript_that_normalises_to_nothing_is_treated_as_silence():
    """Whisper sometimes returns only punctuation for a cough or a click."""
    nlu = ScriptedNLU(_understanding("transfer"))
    pipe = VoicePipeline(asr=FakeASREngine(fixed_text="..."), nlu=nlu, tts=FakeTTSEngine())
    result = pipe.converse(_speech(), DialogueState(session_id="t"))

    assert result.action.type is ActionType.CLARIFY
    assert result.text == ""
    assert nlu.seen == []


def test_the_clarify_reply_is_still_spoken():
    pipe = _pipeline("", _understanding("transfer"))
    result = pipe.converse(_silence(), DialogueState(session_id="t"))
    assert result.audio.duration_s > 0


def test_silence_still_reports_a_latency_breakdown():
    pipe = _pipeline("", _understanding("transfer"))
    result = pipe.converse(_silence(), DialogueState(session_id="t"))
    assert result.latency.total_ms >= 0
    assert "tts" in result.latency.stages


# --- the text path -----------------------------------------------------------


def test_respond_to_text_takes_the_same_route():
    pipe = _pipeline("unused", _understanding("balance", account_type="savings"))
    result = pipe.respond_to_text("What is my savings balance?", DialogueState(session_id="t"))
    assert result.action.type is ActionType.ANSWER
    assert "19,020.00" in result.action.text


def test_respond_to_text_also_normalises():
    nlu = ScriptedNLU(_understanding("transfer", amount="50 dollars", payee="sarah"))
    pipe = VoicePipeline(asr=FakeASREngine(), nlu=nlu, tts=FakeTTSEngine())
    pipe.respond_to_text("Transfer $50 to Sarah.", DialogueState(session_id="t"))
    assert nlu.seen == ["transfer 50 dollars to sarah"]


def test_respond_to_empty_text_clarifies():
    pipe = _pipeline("unused", _understanding("balance"))
    result = pipe.respond_to_text("   ", DialogueState(session_id="t"))
    assert result.action.type is ActionType.CLARIFY


# --- speech can be switched off ----------------------------------------------


def test_speak_false_skips_synthesis():
    """The API's text endpoint wants the words without the waveform."""
    pipe = _pipeline("what is my balance", _understanding("balance"), speak=False)
    result = pipe.converse(_speech(), DialogueState(session_id="t"))
    assert len(result.audio.samples) == 0
    assert result.action.text != ""


# --- timing ------------------------------------------------------------------


def test_every_stage_is_timed():
    pipe = _pipeline("what is my balance", _understanding("balance", account_type="checking"))
    result = pipe.converse(_speech(), DialogueState(session_id="t"))
    assert set(result.latency.stages) == {"vad", "asr", "normalize", "nlu", "dialogue", "tts"}


def test_the_total_is_at_least_the_sum_of_its_parts():
    pipe = _pipeline("what is my balance", _understanding("balance", account_type="checking"))
    result = pipe.converse(_speech(), DialogueState(session_id="t"))
    assert result.latency.total_ms >= sum(result.latency.stages.values()) - 1e-6


def test_slowest_identifies_the_worst_stage():
    tracker = LatencyTracker()
    with tracker.stage("quick"):
        pass
    with tracker.stage("slow"):
        sum(range(200_000))
    slowest = tracker.breakdown().slowest()
    assert slowest is not None
    assert slowest[0] == "slow"


def test_a_stage_that_raises_is_still_timed():
    tracker = LatencyTracker()
    with pytest.raises(ValueError):
        with tracker.stage("broken"):
            raise ValueError("boom")
    assert "broken" in tracker.stages


def test_slowest_of_nothing_is_none():
    assert LatencyTracker().breakdown().slowest() is None


def test_as_dict_includes_the_total():
    tracker = LatencyTracker()
    with tracker.stage("asr"):
        pass
    assert set(tracker.breakdown().as_dict()) == {"asr", "total"}
