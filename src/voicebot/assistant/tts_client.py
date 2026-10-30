"""The two voices: English (qwen3-tts) and Urdu (OmniVoice), as audio to play.

Both return a WAV of 24 kHz, mono, 16-bit PCM, measured. Neither streams:
both accept `stream: true`, but send the first byte only once the whole clip
is done. So a reply is synthesised a sentence at a time, the next one while
the current one plays -- and only one at a time, because each server handles
one request at a time and two at once took twice as long.

Each clip starts with up to 0.4 s of silence (measured on English), which is
cut: the first word then reaches the speaker that much sooner.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import numpy as np

from voicebot.assistant.config import LANGUAGES, AssistantConfig, Language, TtsEndpoint
from voicebot.assistant.http_errors import from_status, from_transport
from voicebot.errors import ProviderUnavailable
from voicebot.utils.logging import get_logger
from voicebot.utils.resilience import RetryPolicy, breaker, call_with_retry, checkpoint

log = get_logger("assistant.tts")


class TtsError(ProviderUnavailable):
    """The voice could not be reached, or did not send back audio."""

    default_component = "tts"


@dataclass(frozen=True)
class Speech:
    samples: np.ndarray  # int16, mono
    sample_rate: int
    text: str
    latency_ms: int
    # Which voice made it (TtsEndpoint.identity): a recording kept for later
    # is only ever played as that same voice.
    voice: str = ""

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


def parse_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Mono int16 samples and the sample rate from a 16-bit PCM WAV.

    Read by hand rather than with `wave`, because a WAV written while being
    streamed carries 0 or 0xFFFFFFFF as its data size, and `wave` rejects it.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise TtsError(f"not a WAV file ({len(data)} bytes, starts {data[:4]!r})")
    pos, channels, rate, bits, tag = 12, 0, 0, 0, 0
    while pos + 8 <= len(data):
        chunk, size = data[pos : pos + 4], int.from_bytes(data[pos + 4 : pos + 8], "little")
        body = pos + 8
        if chunk == b"fmt ":
            tag = int.from_bytes(data[body : body + 2], "little")
            channels = int.from_bytes(data[body + 2 : body + 4], "little")
            rate = int.from_bytes(data[body + 4 : body + 8], "little")
            bits = int.from_bytes(data[body + 14 : body + 16], "little")
        elif chunk == b"data":
            if tag not in (1, 0xFFFE) or bits != 16 or not channels or not rate:
                raise TtsError(f"unsupported WAV: format {tag}, {bits}-bit, {channels} channel(s)")
            available = len(data) - body
            length = available if size in (0, 0xFFFFFFFF) or size > available else size
            length -= length % (2 * channels)
            samples = np.frombuffer(data[body : body + length], dtype="<i2")
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
            return samples.astype(np.int16, copy=False), rate
        pos = body + size + (size & 1)
    raise TtsError("WAV has no data chunk")


def trim_silence(
    samples: np.ndarray,
    rate: int,
    threshold_db: float = -45.0,
    keep_start_ms: int = 30,
    keep_end_ms: int = 80,
) -> np.ndarray:
    """The clip without its leading and trailing silence, give or take a breath."""
    frame = max(1, rate // 100)
    usable = len(samples) // frame * frame
    if not usable:
        return samples
    frames = samples[:usable].astype(np.float32).reshape(-1, frame) / 32768
    loud = np.nonzero(20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9) > threshold_db)[0]
    if not len(loud):
        return samples
    start = max(0, loud[0] * frame - rate * keep_start_ms // 1000)
    end = min(len(samples), (loud[-1] + 1) * frame + rate * keep_end_ms // 1000)
    return samples[start:end]


class TtsClient:
    """One office voice, called the same way as the recognisers: see AsrClient."""

    def __init__(
        self,
        endpoint: TtsEndpoint,
        name: str,
        timeout_s: float,
        retries: int,
        trim: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.name = name
        self.retries = retries
        self.trim = trim
        self.timeout_s = timeout_s
        self.policy = RetryPolicy(attempts=max(0, retries) + 1)
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=5.0))
        self._reference_url: str | None = None

    def synthesize(self, text: str) -> Speech:
        body: dict[str, object] = {
            "model": self.endpoint.model,
            "input": text,
            "response_format": "wav",
        }
        if self.endpoint.voice:
            body["voice"] = self.endpoint.voice
        if self.endpoint.reference:
            body["ref_audio"] = self._reference()
            body["ref_text"] = self.endpoint.reference_text
        body.update(self.endpoint.extra)
        what = f"the {self.name} voice"

        def attempt(timeout: float) -> Speech:
            start = time.perf_counter()
            try:
                response = self._client.post(
                    self.endpoint.url,
                    json=body,
                    timeout=httpx.Timeout(timeout, connect=min(5.0, timeout)),
                )
            except httpx.HTTPError as exc:
                raise from_transport(exc, what, TtsError, timeout) from exc
            if response.status_code != 200:
                raise from_status(response, what, TtsError)
            samples, rate = parse_wav(response.content)
            if not len(samples):
                raise TtsError(f"{what} sent a WAV with no sound in it")
            if self.trim:
                samples = trim_silence(samples, rate)
            return Speech(
                samples,
                rate,
                text,
                round((time.perf_counter() - start) * 1000),
                self.endpoint.identity(),
            )

        return call_with_retry(
            attempt,
            component=f"tts-{self.name}",
            timeout_s=self.limit_for(text),
            policy=self.policy,
            circuit=breaker(f"tts-{self.name}"),
            error_type=TtsError,
        )

    # A piece takes ~2 s a request and ~0.1-0.2 s a word, measured. Several
    # times that, and the voice is not slow but stuck -- measured, qwen3-tts
    # once gave no answer to a short piece for 12 s, twice, and the caller
    # waited 24 s for the backup. The base leaves room for a request queued
    # behind another caller's (each voice makes one clip at a time). No limit
    # is above the configured one.
    LIMIT_BASE_S = 6.0
    LIMIT_PER_WORD_S = 0.4

    def limit_for(self, text: str) -> float:
        """How long one request for `text` may take before it is given up as stuck."""
        words = len(text.split())
        return min(self.timeout_s, self.LIMIT_BASE_S + self.LIMIT_PER_WORD_S * words)

    def _reference(self) -> str:
        """The reference recording as a data URL, read from disk once."""
        if self._reference_url is None:
            data = Path(self.endpoint.reference or "").read_bytes()
            self._reference_url = "data:audio/wav;base64," + base64.b64encode(data).decode()
        return self._reference_url

    def warm(self) -> None:
        """Open the connection now, with a health check, so the first real request need not.

        A failure is left for the real request to report.
        """
        parts = urlsplit(self.endpoint.url)
        try:
            self._client.get(f"{parts.scheme}://{parts.netloc}/health", timeout=3.0)
        except httpx.HTTPError:
            pass

    def close(self) -> None:
        self._client.close()


class BackedUpVoice:
    """A voice, and a backup of the same gender tried when it fails.

    The backup has its own breaker, so a main voice that keeps failing is
    skipped straight to it. Both genders were checked against the agent's at
    startup (AgentConfig), so the backup can never change who is speaking.
    """

    def __init__(self, main: TtsClient, backup: TtsClient) -> None:
        self.main = main
        self.backup = backup
        self.endpoint = main.endpoint
        self.name = main.name

    def synthesize(self, text: str) -> Speech:
        try:
            return self.main.synthesize(text)
        except TtsError as exc:
            # Nobody left to hear it -- the reply was replaced, or the caller
            # hung up -- and the backup's time is the next caller's.
            checkpoint("the backup voice")
            log.warning(
                "the %s voice failed (%s); trying its backup, %s",
                self.name,
                exc.code.value,
                self.backup.endpoint.label or self.backup.endpoint.voice or self.backup.name,
            )
            return self.backup.synthesize(text)

    def warm(self) -> None:
        self.main.warm()
        self.backup.warm()

    def close(self) -> None:
        self.main.close()
        self.backup.close()


def build_voices(config: AssistantConfig) -> dict[Language, TtsClient | BackedUpVoice]:
    """The agent's voice in each language, with its backup when it has one.

    The one place voices are made, for the assistant and the web bot alike,
    so nothing can pick a voice the profile did not.
    """
    tts = config.tts
    voices: dict[Language, TtsClient | BackedUpVoice] = {}
    for language in LANGUAGES:
        endpoint = tts.backup(language)
        # With a backup, a failed request goes straight to it: a voice that
        # did not answer is likelier stuck than unlucky, and asking it again
        # first doubled the wait.
        retries = 0 if endpoint is not None else tts.retries
        main = TtsClient(tts.voice(language), language, tts.timeout_s, retries, tts.trim_silence)
        voices[language] = (
            main
            if endpoint is None
            else BackedUpVoice(
                main,
                TtsClient(
                    endpoint, f"{language}-backup", tts.timeout_s, tts.retries, tts.trim_silence
                ),
            )
        )
    return voices
