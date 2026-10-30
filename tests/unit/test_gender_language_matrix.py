"""Every gender with every language, the fallbacks included: one voice, one gender, one language.

For each combination a turn is run through the real pipeline and the real
office-voice engine, with stand-in voice clients that record what they were
asked to say and whose voice they are. Whatever the path -- an answer, a tool
that failed, a model that slipped into the other language, a voice that went
down -- the sentence is in the conversation's language, speaks of the bot in
its gender, and is spoken by that gender's voice for that language.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from voicebot import profile
from voicebot.agent.executor import ToolExecutor
from voicebot.agent.loop import Agent, Conversation
from voicebot.asr.base import FakeASREngine
from voicebot.assistant.config import TtsEndpoint
from voicebot.assistant.tts_client import Speech, TtsError
from voicebot.bank.base import Bank
from voicebot.llm.base import FakeLLMClient, Reply, ToolCall
from voicebot.llm.prompts import persona, system_prompt
from voicebot.messages import sentence
from voicebot.pipeline.orchestrator import VoicePipeline
from voicebot.profile import DEFAULT_NAMES, AgentProfile, Gender
from voicebot.reply_check import wrong_gender_forms
from voicebot.tts.endpoints import EndpointsTTSEngine

MATRIX = [(g, lang) for g in ("female", "male") for lang in ("en", "ur")]


class _Voice:
    """A voice client of one gender and language that records what it said."""

    def __init__(self, gender: Gender, language: str) -> None:
        self.endpoint = TtsEndpoint(
            url="http://t/x", model="m", voice=f"{gender}-{language}", gender=gender
        )
        self.said: list[str] = []
        self.down = False

    def synthesize(self, text: str) -> Speech:
        if self.down:
            raise TtsError("down")
        self.said.append(text)
        return Speech(np.ones(240, np.int16), 24_000, text, 1, self.endpoint.identity())


@pytest.fixture(params=MATRIX, ids=[f"{g}-{lang}" for g, lang in MATRIX])
def case(request: pytest.FixtureRequest) -> Iterator[tuple[Gender, str]]:
    gender, language = request.param
    previous = profile.use(AgentProfile(gender=gender, names=dict(DEFAULT_NAMES[gender])))
    yield gender, language
    profile.use(previous)


def _turn(bank: Bank, replies: list[Reply], language: str) -> tuple[object, dict[str, _Voice]]:
    gender = profile.current().gender
    voices = {lang: _Voice(gender, lang) for lang in ("en", "ur")}
    tts = EndpointsTTSEngine(voices)  # type: ignore[arg-type]
    agent = Agent(FakeLLMClient(replies), ToolExecutor(bank))
    pipeline = VoicePipeline(asr=FakeASREngine(fixed_text="hello"), agent=agent, tts=tts)
    conversation = Conversation("s1")
    conversation.language = language
    said = "میرا بیلنس" if language == "ur" else "my balance"
    return pipeline.respond_to_text(said, conversation), voices


def _assert_one_voice(voices: dict[str, _Voice], language: str) -> None:
    other = "en" if language == "ur" else "ur"
    assert voices[language].said, "nothing was spoken in the conversation's language"
    assert not voices[other].said, "the other language's voice spoke"


def test_the_model_is_told_the_profiles_gender_and_name(case: tuple[Gender, str]) -> None:
    gender, _ = case
    who = "a woman" if gender == "female" else "a man"
    assert who in persona() and DEFAULT_NAMES[gender]["en"] in system_prompt()


def test_every_fixed_sentence_is_in_the_profiles_gender(case: tuple[Gender, str]) -> None:
    gender, language = case
    for key in ("nothing_said", "busy", "reply_failed", "voice_unavailable", "tool_failed"):
        text = sentence(key, language, gender)
        assert not wrong_gender_forms(text, gender), key


def test_an_answer(bank: Bank, case: tuple[Gender, str]) -> None:
    gender, language = case
    reply = "میں آپ کی مدد کر سکتی ہوں۔" if language == "ur" else "I can help with that."
    result, voices = _turn(bank, [Reply(text=reply)], language)
    _assert_one_voice(voices, language)
    assert not wrong_gender_forms(result.agent.text, gender)  # type: ignore[attr-defined]


def test_a_tool_that_failed(bank: Bank, case: tuple[Gender, str], monkeypatch) -> None:  # type: ignore[no-untyped-def]
    gender, language = case

    def broken(*args: object, **kwargs: object) -> object:
        raise ConnectionError("the bank is down")

    monkeypatch.setattr(bank, "accounts", broken)
    calls = (ToolCall("c1", "list_accounts", {}),)
    result, voices = _turn(bank, [Reply(tool_calls=calls)], language)
    assert result.agent.fallback == "tool_failed"  # type: ignore[attr-defined]
    assert result.agent.text == sentence("tool_failed", language, gender)  # type: ignore[attr-defined]
    _assert_one_voice(voices, language)


def test_a_model_that_slipped_twice(bank: Bank, case: tuple[Gender, str]) -> None:
    gender, language = case
    slip = "Aap ka balance theek hai." if language == "ur" else "آپ کا بیلنس ٹھیک ہے۔"
    result, voices = _turn(bank, [Reply(text=slip), Reply(text=slip)], language)
    assert result.agent.text == sentence("reply_failed", language, gender)  # type: ignore[attr-defined]
    _assert_one_voice(voices, language)


def test_a_voice_that_went_down_is_covered_in_the_same_voice(
    bank: Bank, case: tuple[Gender, str]
) -> None:
    gender, language = case
    voices = {lang: _Voice(gender, lang) for lang in ("en", "ur")}
    tts = EndpointsTTSEngine(voices)  # type: ignore[arg-type]
    tts.warm(("voice_unavailable",))
    recorded = tts.kept("voice_unavailable", language)
    assert recorded is not None
    voices[language].down = True
    agent = Agent(
        FakeLLMClient([Reply(text="Fine." if language == "en" else "ٹھیک ہے۔")]), ToolExecutor(bank)
    )
    pipeline = VoicePipeline(asr=FakeASREngine(fixed_text="hi"), agent=agent, tts=tts)
    conversation = Conversation("s1")
    conversation.language = language
    result = pipeline.respond_to_text("hi" if language == "en" else "ہیلو", conversation)
    # The recording made earlier by this very voice -- never another gender's.
    assert result.audio is recorded
    assert voices[language].said == [sentence("voice_unavailable", language, gender)]
