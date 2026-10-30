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

import contextvars
import re
import threading
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
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


def spoken_language(text: str, fallback: Language = "en") -> Language:
    """The language some text is in, by its script: Urdu script is Urdu.

    Text with letters of neither script -- digits alone, say -- keeps the
    fallback, usually the language the conversation was already in.
    """
    share = urdu_share(text)
    if share is None:
        return fallback
    return "ur" if share >= 0.5 else "en"


# Urdu as it is typed on a phone, in Latin letters: "kia ap muje sun skte
# hain?". Nothing in the script says it is Urdu, so it is told by its small
# words -- the commonest of them, in their usual spellings, less any English
# has too ("main", "me", "to", "the", "sun", "par").
_ROMAN_URDU = frozenset("""
    aap ap apka apki apke apna apni apne mera meri mere mery mujhe muje mjhe mujhy
    hum hamara hamari hamare tum tumhara uska uski uske iska iski iske yeh ye woh wo
    ka ki ke ko se mein mai mei tak liye sath saath
    hai hain hy hn tha thi ho hoon hun hu hua hui raha rahi rahe gaya gayi
    kar karo karna karni karein karen kardo kro krna karta karti karte
    sakta sakti sakte skta skti skte chahiye chahta chahti chahte
    batao btao bataen bataein bataiye bataye dikhao bhejo dena dedo lena
    kia kya kyun kyu kab kahan kaun kon konsa kaunsa kitna kitni kitne kaise kese kaisy
    nahi nahin nai nhi haan ji jee theek thik theak acha accha achha bohat bahut bht
    zyada abhi phir bhi aur lekin magar agar toh tou shukriya meherbani bhai yaar
    kuch koi wala wali wale paise pese paisay aaj kal haal pata baat
    """.split())
_LATIN_WORD = re.compile(r"[a-z]+")


def typed_language(text: str, fallback: Language = "en") -> Language:
    """The language a customer wrote in: by its script, and Urdu in Latin letters too.

    Any Urdu script makes it Urdu -- "میرا balance" is an Urdu question with
    an English word in it. In Latin letters Urdu's words decide when they make
    up a third or more of it, so "mera balance kitna hai" is Urdu and "is the
    main branch open" is not. A word or two of anything else -- "yes",
    "savings", a name -- says nothing about the language, and `fallback`, the
    conversation's own, stands.
    """
    if spoken_language(text, fallback) == "ur" or _ARABIC.search(text):
        return "ur"
    words = _LATIN_WORD.findall(text.lower())
    urdu = sum(word in _ROMAN_URDU for word in words)
    if urdu and urdu * 3 >= len(words):
        return "ur"
    return fallback if len(words) <= 2 else "en"


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

    # Beyond the clients' own timeouts and retries, how long to wait for a
    # recogniser before deciding without it.
    wait_margin_s = 2.0

    def __init__(self, mode: str, en: AsrClient, ur: AsrClient) -> None:
        self.mode = mode
        self.en = en
        self.ur = ur
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr")
        self._jobs: set[Future[Transcript]] = set()
        self._jobs_lock = threading.Lock()

    def cancel_pending(self) -> int:
        """Drop the jobs still queued for a recogniser, for a turn that was interrupted.

        A job already running cannot be stopped -- its client's own timeout
        ends it -- but one still waiting would otherwise hold up the next
        turn's. Returns how many were dropped.
        """
        with self._jobs_lock:
            jobs = list(self._jobs)
        return sum(job.cancel() for job in jobs)

    def transcribe(self, wav: bytes, session_language: Language) -> Decision:
        """Raises AsrError only when no recogniser could be used at all."""
        if self.mode == "en":
            transcript = self.en.transcribe(wav, "en")
            return Decision("en", transcript.text, "English only", {"en": transcript})
        if self.mode == "ur":
            transcript = self.ur.transcribe(wav, "ur")
            return Decision("ur", transcript.text, "Urdu only", {"ur": transcript})

        # Whisper is not told a language, so that it says which one it heard.
        # Each job runs in the caller's context, so its log lines carry the
        # caller's call id.
        jobs = {
            "en": self._submit(self.en.transcribe, wav, "en"),
            "ur": self._submit(self.ur.transcribe, wav, None),
        }
        results: dict[str, Transcript | None] = {}
        errors = []
        try:
            for name, job in jobs.items():
                try:
                    # The clients bound their own requests; this bounds the
                    # wait for them too, so a recogniser that hangs past its
                    # timeout cannot hold the turn.
                    results[name] = job.result(timeout=self._wait_s(name))
                except AsrError as exc:
                    results[name] = None
                    errors.append(str(exc))
                except FutureTimeout:
                    job.cancel()
                    results[name] = None
                    errors.append(f"the {name} recogniser did not finish in time")
                except CancelledError:
                    results[name] = None
                    errors.append(f"the {name} recogniser's job was cancelled")
        finally:
            with self._jobs_lock:
                self._jobs.difference_update(jobs.values())
        if all(t is None for t in results.values()):
            raise AsrError("; ".join(errors))
        for error in errors:
            log.warning("one recogniser failed, deciding on the other: %s", error)
        decision = choose(results["en"], results["ur"], session_language)
        log.info("language %s: %s", decision.language, decision.reason)
        return decision

    def _submit(
        self,
        transcribe: Callable[[bytes, str | None], Transcript],
        wav: bytes,
        language: str | None,
    ) -> Future[Transcript]:
        job = self._pool.submit(contextvars.copy_context().run, transcribe, wav, language)
        with self._jobs_lock:
            self._jobs.add(job)
        return job

    def _wait_s(self, name: str) -> float:
        client = self.en if name == "en" else self.ur
        timeout = float(getattr(client, "timeout_s", 10.0))
        attempts = int(getattr(client, "retries", 1)) + 1
        return timeout * attempts + self.wait_margin_s

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        for client in (self.en, self.ur):
            close = getattr(client, "close", None)
            if callable(close):
                close()
