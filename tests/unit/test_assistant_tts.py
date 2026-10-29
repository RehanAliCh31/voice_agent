"""The voices' audio: read correctly however the WAV was written, trimmed, retried."""

from __future__ import annotations

import io
import json
import wave

import httpx
import numpy as np
import pytest

from voicebot.assistant.config import TtsEndpoint
from voicebot.assistant.tts_client import TtsClient, TtsError, parse_wav, trim_silence


def _wav(samples: np.ndarray, rate: int = 24000, channels: int = 1, width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buffer.getvalue()


def test_a_wav_is_read_as_mono_int16_and_its_rate() -> None:
    samples = np.arange(-500, 500, dtype=np.int16)
    out, rate = parse_wav(_wav(samples))
    assert rate == 24000 and out.dtype == np.int16 and np.array_equal(out, samples)


def test_a_wav_written_while_streaming_is_still_read() -> None:
    data = bytearray(_wav(np.ones(100, dtype=np.int16)))
    data[40:44] = (0xFFFFFFFF).to_bytes(4, "little")  # the data size, unknown when streamed
    out, _ = parse_wav(bytes(data))
    assert len(out) == 100


def test_stereo_is_averaged_to_mono() -> None:
    stereo = np.array([[100, 300], [-100, -300]], dtype=np.int16)
    out, _ = parse_wav(_wav(stereo, channels=2))
    assert list(out) == [200, -200]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b'{"error":"response_format must be pcm or wav"}', "not a WAV"),
        (_wav(np.ones(10, dtype=np.uint8), width=1), "unsupported WAV"),
        (b"RIFF\x00\x00\x00\x00WAVE", "no data chunk"),
    ],
)
def test_what_is_not_16_bit_pcm_wav_is_refused(data: bytes, message: str) -> None:
    with pytest.raises(TtsError, match=message):
        parse_wav(data)


def test_silence_is_trimmed_from_both_ends_leaving_a_little() -> None:
    rate = 24000
    voice = (np.sin(np.arange(rate // 2) / 5) * 8000).astype(np.int16)
    clip = np.concatenate([np.zeros(rate // 2, np.int16), voice, np.zeros(rate, np.int16)])
    out = trim_silence(clip, rate)
    assert rate // 2 <= len(out) < rate // 2 + rate * 0.15


def test_a_clip_of_nothing_but_silence_is_left_alone() -> None:
    silence = np.zeros(2400, dtype=np.int16)
    assert len(trim_silence(silence, 24000)) == 2400


def _client(handler) -> httpx.Client:  # type: ignore[no-untyped-def]
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_the_request_names_the_model_voice_and_extras() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=_wav(np.full(2400, 4000, np.int16)))

    endpoint = TtsEndpoint(
        url="http://tts.test/v1/audio/speech",
        model="OmniVoice",
        voice="assistant-pk",
        extra={"seed": 7},
    )
    speech = TtsClient(endpoint, "ur", 5, 1, client=_client(handler)).synthesize("آپ کا کارڈ")
    assert seen[0] == {
        "model": "OmniVoice",
        "input": "آپ کا کارڈ",
        "response_format": "wav",
        "voice": "assistant-pk",
        "seed": 7,
    }
    assert speech.sample_rate == 24000 and speech.duration_s == pytest.approx(0.1, abs=0.01)


def test_no_voice_field_is_sent_when_none_is_configured() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=_wav(np.full(240, 4000, np.int16)))

    TtsClient(
        TtsEndpoint(url="http://t/x", model="qwen3-tts"), "en", 5, 0, client=_client(handler)
    ).synthesize("Hi")
    assert "voice" not in seen[0]


def test_a_failure_is_retried_once_then_reported() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": "response_format must be pcm or wav"})

    with pytest.raises(TtsError, match="HTTP 400"):
        TtsClient(
            TtsEndpoint(url="http://t/x", model="m"), "en", 5, 1, client=_client(handler)
        ).synthesize("Hi")
    assert len(calls) == 2


def test_a_timeout_then_success_is_a_success() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, content=_wav(np.full(240, 4000, np.int16)))

    TtsClient(
        TtsEndpoint(url="http://t/x", model="m"), "en", 5, 1, client=_client(handler)
    ).synthesize("Hi")
    assert len(calls) == 2
