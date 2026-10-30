"""The orchestrator: VAD, speech to text, the agent, speech out.

Driven entirely by fakes — no model, no GPU, no network, no endpoint. The one
thing these tests guard most carefully is what the agent is *never* asked:
calling a remote model on room tone is a wasted round trip and an invitation
to answer a question nobody asked.
"""

from __future__ import annotations

import numpy as np
import pytest

from voicebot.agent.executor import ToolExecutor
from voicebot.agent.loop import Agent, Conversation
from voicebot.asr.base import ASREngine, ASRUnavailable, FakeASREngine
from voicebot.bank.base import Bank
from voicebot.llm.base import FakeLLMClient, Reply, ToolCall
from voicebot.llm.prompts import NOT_CAUGHT, in_language, language_note
from voicebot.messages import say
from voicebot.pipeline.orchestrator import VoicePipeline
from voicebot.schemas import AudioBuffer, Transcript
from voicebot.tts.base import FakeTTSEngine
from voicebot.tts.browser import BrowserTTSEngine


def _speech(seconds: float = 1.0, rate: int = 16_000) -> AudioBuffer:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
    samples = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return AudioBuffer(samples=samples, sample_rate=rate)


def _silence(seconds: float = 1.0, rate: int = 16_000) -> AudioBuffer:
    return AudioBuffer(samples=np.zeros(int(rate * seconds), dtype=np.float32), sample_rate=rate)


def sent(said: str, language: str = "en") -> str:
    """The customer's words as the model is sent them: with the language to answer in."""
    return f"{said}\n\n({language_note(language)})"


def _pipeline(
    bank: Bank, replies: list[Reply], heard: str = "what is my balance", speak: bool = True
) -> tuple[VoicePipeline, FakeLLMClient]:
    client = FakeLLMClient(replies)
    agent = Agent(client, ToolExecutor(bank))
    pipeline = VoicePipeline(
        asr=FakeASREngine(fixed_text=heard), agent=agent, tts=FakeTTSEngine(), speak=speak
    )
    return pipeline, client


# --- a whole turn ------------------------------------------------------------


def test_a_spoken_turn_runs_end_to_end(bank: Bank) -> None:
    pipeline, _ = _pipeline(bank, [Reply(text="Hello there.")])
    result = pipeline.converse(_speech(), Conversation("s1"))

    assert result.text == "what is my balance"
    assert result.agent.text == "Hello there."
    assert len(result.audio.samples) > 0


def test_a_typed_turn_takes_the_same_path(bank: Bank) -> None:
    pipeline, _ = _pipeline(bank, [Reply(text="Hello there.")])
    result = pipeline.respond_to_text("hello", Conversation("s1"))
    assert result.agent.text == "Hello there."


def test_latency_names_the_stages_that_exist(bank: Bank) -> None:
    pipeline, _ = _pipeline(bank, [Reply(text="ok")])
    stages = set(pipeline.converse(_speech(), Conversation("s1")).latency.stages)

    assert stages == {"vad", "asr", "agent", "tts"}
    # The NLU and template-dialogue stages are gone, and so is the normaliser
    # that existed only to feed a model trained on lowercased text.
    assert not stages & {"nlu", "dialogue", "normalize"}


# --- nothing is sent to the model that should not be ------------------------


def test_silence_never_reaches_the_model(bank: Bank) -> None:
    # FakeLLMClient with an empty script raises if it is called at all.
    pipeline, client = _pipeline(bank, [])
    result = pipeline.converse(_silence(), Conversation("s1"))

    assert result.text == ""
    assert client.calls == []
    assert result.agent.text


@pytest.mark.parametrize("heard", ["", "   ", "...", ".", "?!"])
def test_an_empty_transcript_never_reaches_the_model(bank: Bank, heard: str) -> None:
    # Whisper returns punctuation-only strings on room tone. Classifying that
    # is a round trip spent to learn nothing.
    pipeline, client = _pipeline(bank, [], heard=heard)
    result = pipeline.converse(_speech(), Conversation("s1"))

    assert result.text == ""
    assert client.calls == []


def test_whisper_text_reaches_the_model_unmodified(bank: Bank) -> None:
    # Casing and punctuation are signal to an LLM. v1 stripped both for a
    # model trained on lowercased text; doing that now is deliberate damage.
    pipeline, client = _pipeline(bank, [Reply(text="ok")], heard="Transfer $50 to Sarah.")
    pipeline.converse(_speech(), Conversation("s1"))

    # Exactly as heard, then the language to answer in.
    assert client.calls[0][0][-1]["content"] == sent("Transfer $50 to Sarah.")


# --- speech out --------------------------------------------------------------


def test_the_browser_engine_returns_no_audio() -> None:
    assert len(BrowserTTSEngine().synthesize("hello").samples) == 0


def test_the_browser_engine_declares_that_the_page_speaks() -> None:
    assert BrowserTTSEngine().speaks_client_side is True
    assert FakeTTSEngine().speaks_client_side is False


def test_speak_false_skips_synthesis(bank: Bank) -> None:
    pipeline, _ = _pipeline(bank, [Reply(text="ok")], speak=False)
    assert len(pipeline.converse(_speech(), Conversation("s1")).audio.samples) == 0


# --- state carries across turns ----------------------------------------------


def test_the_conversation_accumulates_across_turns(bank: Bank) -> None:
    pipeline, client = _pipeline(bank, [Reply(text="one"), Reply(text="two")])
    conversation = Conversation("s1")

    pipeline.respond_to_text("first", conversation)
    pipeline.respond_to_text("second", conversation)

    assert conversation.turn_index == 2
    assert len(client.calls[1][0]) > len(client.calls[0][0])


def test_a_tool_call_inside_a_spoken_turn_is_reported(bank: Bank) -> None:
    client = FakeLLMClient(
        [
            Reply(tool_calls=(ToolCall("c1", "get_balance", {"account": "savings"}),)),
            Reply(text="Your savings balance is $19,020.00."),
        ]
    )
    pipeline = VoicePipeline(
        asr=FakeASREngine(fixed_text="what is my savings balance"),
        agent=Agent(client, ToolExecutor(bank)),
        tts=FakeTTSEngine(),
    )
    result = pipeline.converse(_speech(), Conversation("s1"))

    assert result.agent.tool_calls[0].name == "get_balance"
    assert result.agent.guard_hit is False


# --- the customer's language ---------------------------------------------------


class _HeardIn(ASREngine):
    """A recogniser that names the language it heard, as the office ones do."""

    def __init__(self, text: str, language: str) -> None:
        self.text = text
        self.language = language

    def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcript:
        return Transcript(
            text=self.text, confidence=1.0, duration_s=audio.duration_s, language=self.language
        )


class _Down(ASREngine):
    def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcript:
        raise ASRUnavailable("both recognisers are down")


def _hearing(
    bank: Bank, asr: ASREngine, replies: list[Reply]
) -> tuple[VoicePipeline, FakeLLMClient]:
    client = FakeLLMClient(replies)
    agent = Agent(client, ToolExecutor(bank))
    return VoicePipeline(asr=asr, agent=agent, tts=FakeTTSEngine()), client


def in_urdu() -> Conversation:
    conversation = Conversation("s1")
    conversation.language = "ur"
    return conversation


def test_the_turn_takes_the_language_the_recogniser_heard(bank: Bank) -> None:
    pipeline, client = _hearing(
        bank, _HeardIn("میرا بیلنس کیا ہے", "ur"), [Reply(text="کون سا اکاؤنٹ؟")]
    )
    conversation = Conversation("s1")

    pipeline.converse(_speech(), conversation)

    assert conversation.language == "ur"
    assert client.calls[0][0][-1]["content"] == sent("میرا بیلنس کیا ہے", "ur")


def test_typing_follows_the_script_both_ways(bank: Bank) -> None:
    pipeline, _ = _pipeline(bank, [Reply(text="جی؟"), Reply(text="Yes?")])
    conversation = Conversation("s1")

    pipeline.respond_to_text("میرا بیلنس کیا ہے؟", conversation)
    assert conversation.language == "ur"
    pipeline.respond_to_text("what is my balance?", conversation)
    assert conversation.language == "en"


def test_urdu_typed_in_latin_letters_is_answered_as_urdu(bank: Bank) -> None:
    # Measured: taken for English, it was answered in Latin-letter Urdu five
    # times in eight, for the English voice to read.
    pipeline, client = _pipeline(bank, [Reply(text="جی ہاں۔")])
    conversation = Conversation("s1")

    pipeline.respond_to_text("kia ap muje sun skte hain?", conversation)

    assert conversation.language == "ur"
    # What was typed reaches the model as typed, with the note after it.
    assert client.calls[0][0][-1]["content"] == sent("kia ap muje sun skte hain?", "ur")


def test_a_recogniser_outage_is_said_and_not_blamed_on_the_customer(bank: Bank) -> None:
    pipeline, client = _hearing(bank, _Down(), [])

    result = pipeline.converse(_speech(), in_urdu())

    # Not "I didn't catch that": they spoke, and the service failed them --
    # and they are told they can type instead.
    assert result.agent.text == say("speech_unavailable", "ur")
    assert result.agent.fallback == "speech_unavailable"
    assert client.calls == []


def test_nothing_heard_in_an_urdu_conversation_is_said_in_urdu(bank: Bank) -> None:
    pipeline, client = _hearing(bank, _HeardIn("", "ur"), [])

    result = pipeline.converse(_speech(), in_urdu())

    assert result.agent.text == in_language(NOT_CAUGHT, "ur") != NOT_CAUGHT
    assert client.calls == []


def test_a_word_of_urdu_is_something_said(bank: Bank) -> None:
    # "ہاں" -- yes -- has no letter from A to Z in it, and is a whole answer.
    pipeline, client = _hearing(bank, _HeardIn("ہاں", "ur"), [Reply(text="ٹھیک ہے۔")])

    result = pipeline.converse(_speech(), Conversation("s1"))

    assert result.text == "ہاں"
    assert client.calls[0][0][-1]["content"] == sent("ہاں", "ur")


@pytest.mark.parametrize("heard", ["Yes.", "Savings", "OK"])
def test_a_word_or_two_of_english_leaves_an_urdu_conversation_in_urdu(
    bank: Bank, heard: str
) -> None:
    pipeline, _ = _hearing(bank, _HeardIn(heard, "en"), [Reply(text="ٹھیک ہے۔")])
    conversation = in_urdu()

    pipeline.converse(_speech(), conversation)

    assert conversation.language == "ur"


def test_an_english_sentence_takes_an_urdu_conversation_back_to_english(bank: Bank) -> None:
    pipeline, client = _hearing(
        bank, _HeardIn("What is my checking balance?", "en"), [Reply(text="Which one?")]
    )
    conversation = in_urdu()

    pipeline.converse(_speech(), conversation)

    assert conversation.language == "en"
    assert client.calls[0][0][-1]["content"] == sent("What is my checking balance?")
