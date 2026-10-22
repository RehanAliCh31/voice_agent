import numpy as np
import pytest

from voicebot.asr.base import ASREngine, FakeASREngine
from voicebot.nlu.base import FakeNLUEngine, NLUEngine
from voicebot.schemas import AudioBuffer, Transcript, Understanding
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


def test_fake_nlu_satisfies_the_contract():
    engine = FakeNLUEngine(fixed_intent="transfer")
    result = engine.understand("transfer fifty dollars to sarah")
    assert isinstance(result, Understanding)
    assert result.intent == "transfer"
    assert 0.0 <= result.intent_confidence <= 1.0


def test_fake_nlu_treats_blank_input_as_out_of_scope():
    engine = FakeNLUEngine(fixed_intent="transfer")
    result = engine.understand("   ")
    assert result.is_out_of_scope is True
    assert result.intent_confidence == 0.0


def test_fake_tts_returns_audio_of_nonzero_length():
    engine = FakeTTSEngine(sample_rate=16000)
    buf = engine.synthesize("confirm the transfer")
    assert isinstance(buf, AudioBuffer)
    assert buf.sample_rate == 16000
    assert buf.duration_s > 0


def test_abstract_engines_cannot_be_instantiated():
    for cls in (ASREngine, NLUEngine, TTSEngine):
        with pytest.raises(TypeError):
            cls()


@pytest.mark.parametrize("kind", ["asr", "nlu", "tts"])
def test_fakes_are_resolvable_through_the_registry(kind):
    engine = create(kind, "fake")
    assert engine is not None
