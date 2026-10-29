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
from voicebot.asr.base import FakeASREngine
from voicebot.bank.base import Bank
from voicebot.llm.base import FakeLLMClient, Reply, ToolCall
from voicebot.pipeline.orchestrator import VoicePipeline
from voicebot.schemas import AudioBuffer
from voicebot.tts.base import FakeTTSEngine
from voicebot.tts.browser import BrowserTTSEngine


def _speech(seconds: float = 1.0, rate: int = 16_000) -> AudioBuffer:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
    samples = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return AudioBuffer(samples=samples, sample_rate=rate)


def _silence(seconds: float = 1.0, rate: int = 16_000) -> AudioBuffer:
    return AudioBuffer(samples=np.zeros(int(rate * seconds), dtype=np.float32), sample_rate=rate)


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

    assert client.calls[0][0][-1]["content"] == "Transfer $50 to Sarah."


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
