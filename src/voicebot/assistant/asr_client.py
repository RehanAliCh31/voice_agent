"""The two speech recognisers, and what to do about what they hear in silence.

Both take a 16 kHz mono 16-bit WAV, which is what the microphone produces.
That is not a preference: whisper-urdu refuses every other sample rate with a
400 ("Invalid or unsupported audio file"), measured at 8, 24 and 48 kHz.

Both also invent words for silence and noise, measured: qwen3-asr heard "I."
and "Okay.", whisper-urdu "ملتے ہیں" and "Thank you.". Voice activity
detection keeps silence from reaching them at all; `is_phantom` is the second
line, for the click or breath that gets through.
"""

from __future__ import annotations

import io
import re
import time
import wave
from dataclasses import dataclass

import httpx
import numpy as np

from voicebot.assistant.config import AsrEndpoint
from voicebot.errors import ProviderUnavailable
from voicebot.utils.logging import get_logger

log = get_logger("assistant.asr")


class AsrError(ProviderUnavailable):
    """The recogniser could not be reached, or answered with something unreadable."""

    default_component = "asr"


@dataclass(frozen=True)
class Transcript:
    text: str
    endpoint: str  # "en" or "ur": which recogniser heard it
    latency_ms: int
    # Only whisper-urdu reports these (in verbose_json); None from qwen3-asr.
    language: str | None = None
    avg_logprob: float | None = None


def to_wav(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """16-bit mono PCM samples as a WAV file in memory. Nothing touches the disk."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(np.asarray(audio, dtype="<i2").tobytes())
    return buffer.getvalue()


class AsrClient:
    def __init__(
        self,
        endpoint: AsrEndpoint,
        name: str,
        timeout_s: float,
        retries: int,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.name = name
        self.retries = retries
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=5.0))

    def transcribe(self, wav: bytes, language: str | None) -> Transcript:
        """What was said. `language` None lets the recogniser decide."""
        form = {"model": self.endpoint.model, "response_format": self.endpoint.response_format}
        if language:
            form["language"] = language
        for attempt in range(1, self.retries + 2):
            start = time.perf_counter()
            try:
                response = self._client.post(
                    self.endpoint.url, files={"file": ("speech.wav", wav, "audio/wav")}, data=form
                )
                if response.status_code != 200:
                    raise AsrError(f"HTTP {response.status_code}: {response.text[:200]}")
                body = response.json()
                if not isinstance(body, dict) or not isinstance(body.get("text"), str):
                    raise AsrError(f"no text in the response: {str(body)[:200]}")
            except (httpx.HTTPError, AsrError, ValueError) as exc:
                if attempt > self.retries:
                    raise AsrError(f"{self.name} recogniser: {type(exc).__name__}: {exc}") from exc
                log.warning(
                    "%s recogniser attempt %d failed (%s), retrying", self.name, attempt, exc
                )
                continue
            segments = body.get("segments") or []
            logprobs = [
                s["avg_logprob"] for s in segments if isinstance(s.get("avg_logprob"), int | float)
            ]
            return Transcript(
                text=body["text"].strip(),
                endpoint=self.name,
                latency_ms=round((time.perf_counter() - start) * 1000),
                language=body.get("language") if isinstance(body.get("language"), str) else None,
                avg_logprob=float(np.mean(logprobs)) if logprobs else None,
            )
        raise AssertionError("unreachable")


_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def normalise(text: str) -> str:
    """Lower case, no punctuation (Urdu's included), single spaces."""
    return " ".join(_PUNCTUATION.sub(" ", text.lower()).split())


def is_phantom(
    transcript: Transcript,
    speech_ms: int,
    phrases: list[str],
    max_speech_ms: int,
    min_avg_logprob: float,
) -> bool:
    """True for what a recogniser makes up from silence, noise or a click.

    Nothing at all, or punctuation alone; a transcript the recogniser itself
    doubts (noise scored -1.03, real speech -0.08 to -0.22); or one of the
    stock phrases each recogniser produces, from too little speech to be one.
    """
    words = normalise(transcript.text)
    if not words:
        return True
    if transcript.avg_logprob is not None and transcript.avg_logprob < min_avg_logprob:
        return True
    known = {normalise(p) for p in phrases}
    return words in known and speech_ms <= max_speech_ms
