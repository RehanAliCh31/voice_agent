"""Every engine honours its interface, and every fake is resolvable.

The fakes are what make the rest of the suite run offline, so their contracts
are worth asserting directly rather than only through the code that uses them.
"""

import numpy as np
import pytest

from voicebot.asr.base import ASREngine, FakeASREngine
from voicebot.llm.base import FakeLLMClient, LLMClient, Reply, ToolCall
from voicebot.schemas import AudioBuffer, Transcript
from voicebot.tts.base import FakeTTSEngine, TTSEngine
from voicebot.utils.registry import create


def _audio(seconds: float = 1.0, sample_rate: int = 16000) -> AudioBuffer:
    return AudioBuffer(
        samples=np.zeros(int(seconds * sample_rate), dtype=np.float32),
        sample_rate=sample_rate,
    )


def test_fake_asr_satisfies_the_contract():
    engine = FakeASREngine(fixed_text="transfer fifty dollars to sarah")
    result = engine.transcribe(_audio())
    assert isinstance(result, Transcript)
    assert result.text == "transfer fifty dollars to sarah"
    assert result.duration_s == pytest.approx(1.0)


def test_fake_llm_satisfies_the_contract():
    client = FakeLLMClient([Reply(text="hello", tool_calls=(ToolCall("c1", "get_balance"),))])
    reply = client.complete([], [])
    assert isinstance(reply, Reply)
    assert reply.text == "hello"
    assert reply.tool_calls[0].name == "get_balance"


def test_fake_tts_returns_audio_of_nonzero_length():
    engine = FakeTTSEngine(sample_rate=16000)
    buf = engine.synthesize("confirm the transfer")
    assert isinstance(buf, AudioBuffer)
    assert buf.sample_rate == 16000
    assert buf.duration_s > 0


def test_abstract_engines_cannot_be_instantiated():
    for cls in (ASREngine, TTSEngine, LLMClient):
        with pytest.raises(TypeError):
            cls()


@pytest.mark.parametrize("kind", ["asr", "tts"])
def test_fakes_are_resolvable_through_the_registry(kind):
    engine = create(kind, "fake")
    assert engine is not None
