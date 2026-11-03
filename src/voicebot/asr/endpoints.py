"""Speech recognition by the office endpoints, in English and in Urdu.

Both recognisers hear every utterance at once, the language is decided from
what they return, and what they make up from noise is dropped. The endpoints are named
in configs/assistant.yaml, the one place they are listed.

Every utterance reports the language it was heard in, so the turn -- and the
voice that answers it -- can follow the customer from one language to the
other.
"""

from __future__ import annotations

import math

import numpy as np

from voicebot.asr.base import ASREngine, ASRUnavailable
from voicebot.assistant.asr_client import (
    AsrClient,
    AsrError,
    is_phantom,
    normalise,
    to_rate,
    to_wav,
)
from voicebot.assistant.asr_client import Transcript as Heard
from voicebot.assistant.config import AsrConfig, AssistantConfig, Language
from voicebot.assistant.language import Decision, LanguageRouter, has_devanagari
from voicebot.schemas import AudioBuffer, Segment, Transcript
from voicebot.utils.logging import get_logger
from voicebot.utils.registry import register

log = get_logger("asr.endpoints")

RATE = 16000  # the one rate whisper-urdu accepts
# At most this many words, from audio whisper doubts, is noise (_noise).
NOISE_MAX_WORDS = 4


@register("asr", "endpoints")
class EndpointsASREngine(ASREngine):
    def __init__(self, router: LanguageRouter, config: AsrConfig) -> None:
        self.router = router
        self.config = config

    @classmethod
    def from_config(cls, config: AssistantConfig) -> EndpointsASREngine:
        asr = config.asr
        # Both recognisers hear every turn, and the language is decided from them.
        router = LanguageRouter(
            AsrClient(asr.en, "en", asr.timeout_s, asr.retries),
            AsrClient(asr.ur, "ur", asr.timeout_s, asr.retries),
        )
        return cls(router, asr)

    def transcribe(self, audio: AudioBuffer, language: str | None = None) -> Transcript:
        """`language`, the conversation's, breaks a tie when the recognisers cannot tell."""
        pcm = (np.clip(audio.samples, -1.0, 1.0) * 32767).astype(np.int16)
        pcm = to_rate(pcm, audio.sample_rate, RATE)
        tie_break: Language = "ur" if language == "ur" else "en"
        try:
            decision = self.router.transcribe(to_wav(pcm, RATE), tie_break)
        except AsrError as exc:
            raise ASRUnavailable(str(exc), code=exc.code, cause=exc) from exc

        decision = self._english_if_whisper_doubts(decision)
        language = decision.language
        heard = decision.heard.get(language) or Heard(decision.text, language, 0)
        checked = Heard(
            decision.text, heard.endpoint, heard.latency_ms, heard.language, heard.avg_logprob
        )
        speech_ms = round(audio.duration_s * 1000)
        if self._noise(decision) or is_phantom(
            checked,
            speech_ms,
            self.config.phantom_phrases[language],
            self.config.phantom_max_speech_ms,
            self.config.min_avg_logprob,
        ):
            log.info("nothing to answer in %d ms of speech", speech_ms)
            log.debug("discarded as noise: %r", decision.text)
            text = ""
        else:
            text = decision.text
        # Whisper's average log-probability is the confidence there is; the
        # English recogniser reports none.
        logprob = heard.avg_logprob
        confidence = math.exp(logprob) if logprob is not None else 1.0
        return Transcript(
            text=text,
            confidence=confidence,
            duration_s=audio.duration_s,
            segments=(Segment(text=text, start_s=0.0, end_s=audio.duration_s),) if text else (),
            language=language,
        )

    def _english_if_whisper_doubts(self, decision: Decision) -> Decision:
        """qwen3-asr's English sentence, when whisper's Urdu one is doubted.

        In the live benchmark whisper wrote an English question as an Urdu
        sentence and doubted it; its grammar words won it the language, then
        its doubt had it discarded, and qwen3-asr's English went with it --
        "nothing to answer". A doubted whisper transcript is no evidence of
        Urdu: if qwen3-asr heard a real sentence, that is what was said.
        """
        whisper, qwen = decision.heard.get("ur"), decision.heard.get("en")
        if (
            decision.language != "ur"
            or whisper is None
            or whisper.avg_logprob is None
            or whisper.avg_logprob >= self.config.min_avg_logprob
            or qwen is None
            or has_devanagari(qwen.text)  # Urdu that qwen3-asr wrote as Hindi
            or len(normalise(qwen.text).split()) <= NOISE_MAX_WORDS
        ):
            return decision
        log.info("whisper doubted its Urdu; qwen3-asr heard an English sentence")
        return Decision("en", qwen.text, "whisper doubted its Urdu", decision.heard)

    def _noise(self, decision: Decision) -> bool:
        """Noise that qwen3-asr put a few words to, whatever its length.

        On two to four seconds of noise whisper writes a full stop or one of
        its stock phrases, and doubts it (-0.88 to -1.57; speech scores -0.08
        to -0.22), while qwen3-asr, which says nothing of its confidence,
        makes up "Okay." or "I'm not sure." -- and was answered. When whisper
        doubts what it heard, a transcript that short is noise too; a real
        sentence is longer, and is kept.
        """
        whisper = decision.heard.get("ur")
        if whisper is None or whisper.avg_logprob is None:
            return False
        doubted = whisper.avg_logprob < self.config.min_avg_logprob
        return doubted and 0 < len(normalise(decision.text).split()) <= NOISE_MAX_WORDS

    def connect(self) -> None:
        """Open both recognisers' connections now, for a live session about to need them."""
        for client in (self.router.en, self.router.ur):
            warm = getattr(client, "warm", None)
            if callable(warm):
                warm()

    def close(self) -> None:
        self.router.close()
