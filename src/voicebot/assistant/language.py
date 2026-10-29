"""Which language was spoken, and so which transcript to believe.

Neither recogniser can be told the language in advance in auto mode, so both
hear every utterance at once (0.5-0.6 s for both, measured, against 0.3 and
0.6 s apart) and the evidence is weighed:

* whisper-urdu names the language it heard when not told one ("en" or "ur",
  correct on every clip measured), and writes Urdu in Urdu script and
  English in English;
* qwen3-asr, told English, writes Urdu speech as Hindi in Devanagari script
  -- a strong sign on its own that the speech was not English.

English goes to qwen3-asr's transcript, the English specialist; Urdu to
whisper-urdu's. Mixed speech goes to the language that dominates it, which
is what Whisper reports. Each turn is weighed afresh, so the customer can
switch language mid-conversation; the session's language is the tie-break.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from voicebot.assistant.asr_client import AsrClient, AsrError, Transcript
from voicebot.assistant.config import Language
from voicebot.utils.logging import get_logger

log = get_logger("assistant.language")

_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")


def urdu_share(text: str) -> float | None:
    """Share of the letters that are Urdu script; None when there are none of either."""
    urdu, latin = len(_ARABIC.findall(text)), len(_LATIN.findall(text))
    return urdu / (urdu + latin) if urdu + latin else None


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


@dataclass(frozen=True)
class Decision:
    language: Language
    text: str
    reason: str
    heard: dict[str, Transcript] = field(default_factory=dict)


def choose(en: Transcript | None, ur: Transcript | None, fallback: Language) -> Decision:
    """The language and transcript, from whatever the recognisers returned."""
    heard = {t.endpoint: t for t in (en, ur) if t is not None}
    en_text = en.text if en else ""
    ur_text = ur.text if ur else ""
    share = urdu_share(ur_text) if ur_text else None

    if ur_text and (ur.language == "ur" if ur else False):
        return Decision("ur", ur_text, f"whisper heard Urdu ({_pct(share)} Urdu script)", heard)
    if ur_text and share is not None and share >= 0.5:
        return Decision("ur", ur_text, f"{_pct(share)} of whisper's letters are Urdu script", heard)
    if has_devanagari(en_text):
        if ur_text:
            return Decision(
                "ur", ur_text, "qwen3-asr wrote Hindi script: the speech was Urdu", heard
            )
        return Decision("ur", en_text, "qwen3-asr wrote Hindi script; no Urdu transcript", heard)
    if en_text:
        why = "whisper heard English" if ur and ur.language == "en" else "English transcript"
        return Decision("en", en_text, why, heard)
    if ur_text:
        # qwen3-asr heard nothing, or failed; whisper's text is all there is.
        language: Language = "ur" if (share or 0) >= 0.5 else "en"
        return Decision(language, ur_text, "only whisper heard anything", heard)
    return Decision(fallback, "", "neither recogniser heard anything", heard)


def _pct(share: float | None) -> str:
    return "no" if share is None else f"{share:.0%}"


class LanguageRouter:
    """Sends an utterance to the right recogniser, or to both, and decides."""

    def __init__(self, mode: str, en: AsrClient, ur: AsrClient) -> None:
        self.mode = mode
        self.en = en
        self.ur = ur
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr")

    def transcribe(self, wav: bytes, session_language: Language) -> Decision:
        """Raises AsrError only when no recogniser could be used at all."""
        if self.mode == "en":
            transcript = self.en.transcribe(wav, "en")
            return Decision("en", transcript.text, "English only", {"en": transcript})
        if self.mode == "ur":
            transcript = self.ur.transcribe(wav, "ur")
            return Decision("ur", transcript.text, "Urdu only", {"ur": transcript})

        # Whisper is not told a language, so that it says which one it heard.
        jobs = {
            "en": self._pool.submit(self.en.transcribe, wav, "en"),
            "ur": self._pool.submit(self.ur.transcribe, wav, None),
        }
        results: dict[str, Transcript | None] = {}
        errors = []
        for name, job in jobs.items():
            try:
                results[name] = job.result()
            except AsrError as exc:
                results[name] = None
                errors.append(str(exc))
        if all(t is None for t in results.values()):
            raise AsrError("; ".join(errors))
        for error in errors:
            log.warning("one recogniser failed, deciding on the other: %s", error)
        decision = choose(results["en"], results["ur"], session_language)
        log.info("language %s: %s", decision.language, decision.reason)
        return decision

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
