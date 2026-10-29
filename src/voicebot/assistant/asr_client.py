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
from voicebot.assistant.http_errors import from_status, from_transport
from voicebot.errors import ProviderUnavailable
from voicebot.utils.logging import get_logger
from voicebot.utils.resilience import RetryPolicy, breaker, call_with_retry

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
    """One office recogniser.

    Called through call_with_retry with its own circuit breaker, so a timeout
    or a 5xx is retried once with backoff, a 4xx never is, and a recogniser
    that keeps failing is skipped for a while -- the other one then decides
    alone. The response body is only ever logged at DEBUG: it can hold what
    the customer said.
    """

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
        self.timeout_s = timeout_s
        self.policy = RetryPolicy(attempts=max(0, retries) + 1)
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=5.0))

    def transcribe(self, wav: bytes, language: str | None) -> Transcript:
        """What was said. `language` None lets the recogniser decide."""
        form = {"model": self.endpoint.model, "response_format": self.endpoint.response_format}
        if language:
            form["language"] = language
        what = f"the {self.name} recogniser"

        def attempt(timeout: float) -> Transcript:
            start = time.perf_counter()
            try:
                response = self._client.post(
                    self.endpoint.url,
                    files={"file": ("speech.wav", wav, "audio/wav")},
                    data=form,
                    timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)),
                )
            except httpx.HTTPError as exc:
                raise from_transport(exc, what, AsrError, timeout) from exc
            if response.status_code != 200:
                raise from_status(response, what, AsrError)
            try:
                body = response.json()
            except ValueError as exc:
                raise AsrError(f"{what} sent something that is not JSON", cause=exc) from exc
            if not isinstance(body, dict) or not isinstance(body.get("text"), str):
                raise AsrError(f"{what} sent no text")
            segments = body.get("segments")
            logprobs = [
                s["avg_logprob"]
                for s in (segments if isinstance(segments, list) else [])
                if isinstance(s, dict) and isinstance(s.get("avg_logprob"), int | float)
            ]
            return Transcript(
                text=body["text"].strip(),
                endpoint=self.name,
                latency_ms=round((time.perf_counter() - start) * 1000),
                language=body.get("language") if isinstance(body.get("language"), str) else None,
                avg_logprob=float(np.mean(logprobs)) if logprobs else None,
            )

        return call_with_retry(
            attempt,
            component=f"asr-{self.name}",
            timeout_s=self.timeout_s,
            policy=self.policy,
            circuit=breaker(f"asr-{self.name}"),
            error_type=AsrError,
        )

    def close(self) -> None:
        self._client.close()


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
