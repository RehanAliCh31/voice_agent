"""Speech recognition by the office endpoints, in English and in Urdu.

The same recognisers as the hands-free assistant, through the same code:
both hear every utterance at once, the language is decided from what they
return, and what they make up from noise is dropped. The endpoints are named
in configs/assistant.yaml, the one place they are listed.

Every utterance reports the language it was heard in, so the turn -- and the
voice that answers it -- can follow the customer from one language to the
other.
"""

from __future__ import annotations

import math

import numpy as np

from voicebot.asr.base import ASREngine, ASRUnavailable
from voicebot.assistant.asr_client import AsrClient, AsrError, is_phantom, to_wav
from voicebot.assistant.asr_client import Transcript as Heard
from voicebot.assistant.config import AsrConfig, AssistantConfig, Language
from voicebot.assistant.language import LanguageRouter
from voicebot.assistant.player import to_rate
from voicebot.schemas import AudioBuffer, Segment, Transcript
from voicebot.utils.logging import get_logger
from voicebot.utils.registry import register

log = get_logger("asr.endpoints")

RATE = 16000  # the one rate whisper-urdu accepts


@register("asr", "endpoints")
class EndpointsASREngine(ASREngine):
    def __init__(self, router: LanguageRouter, config: AsrConfig) -> None:
        self.router = router
        self.config = config
        # The language of the last utterance, the tie-break when the
        # recognisers cannot tell.
        self._language: Language = "en"

    @classmethod
    def from_config(cls, config: AssistantConfig) -> EndpointsASREngine:
        asr = config.asr
        router = LanguageRouter(
            config.session.language,
            AsrClient(asr.en, "en", asr.timeout_s, asr.retries),
            AsrClient(asr.ur, "ur", asr.timeout_s, asr.retries),
        )
        return cls(router, asr)

    def transcribe(self, audio: AudioBuffer) -> Transcript:
        pcm = (np.clip(audio.samples, -1.0, 1.0) * 32767).astype(np.int16)
        pcm = to_rate(pcm, audio.sample_rate, RATE)
        try:
            decision = self.router.transcribe(to_wav(pcm, RATE), self._language)
        except AsrError as exc:
            raise ASRUnavailable(str(exc)) from exc

        language = decision.language
        self._language = language
        heard = decision.heard.get(language) or Heard(decision.text, language, 0)
        checked = Heard(
            decision.text, heard.endpoint, heard.latency_ms, heard.language, heard.avg_logprob
        )
        speech_ms = round(audio.duration_s * 1000)
        if is_phantom(
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
