"""Which language was spoken, and so which transcript to believe.

Neither recogniser can be told the language in advance in auto mode, so both
hear every utterance at once (0.5-0.6 s for both, measured, against 0.3 and
0.6 s apart) and the evidence is weighed:

* whisper-urdu names the language it heard when not told one, "en" or "ur".
  Right on Urdu, but it called 12 of 30 English clips Urdu (most spoken with
  a Pakistani accent), writing the English out in Urdu letters -- "کین یو شو
  مائی ٹرانزیکشنز" -- which no Urdu sentence looks like: Urdu's small grammar
  words are missing;
* qwen3-asr, told English, writes Urdu speech as Hindi in Devanagari script,
  or as Roman Urdu -- strong signs that the speech was not English -- or now
  and then translates it into English, which is a weak sign of anything.

Each sign is counted for one language or the other, and the larger count
wins; whisper's verdict settles a tie. English goes to qwen3-asr's
transcript, the English specialist; Urdu to whisper-urdu's. Mixed speech goes
to the language its grammar is in. Each turn is weighed afresh, so the
customer can switch language mid-conversation; the session's language is
used only when neither recogniser heard anything.
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


# The small words Urdu grammar cannot do without: postpositions, the verb "to
# be", pronouns, question words, particles. A real Urdu sentence of three
# words or more has at least one; English written out in Urdu letters has
# none. Left out: words English speech also turns into ("تو" for "to", "اس"
# for "is", "دس" for "this").
_URDU_GRAMMAR = frozenset("""
    ہے ہیں ہوں ہو تھا تھی تھے گا گی گے کا کی کے کو سے نے میں پر بھی اور یا
    کیا کیوں کب کہاں کون کتنا کتنی کتنے کیسے مجھے میرا میری میرے مرے ہمارا
    آپ ہم یہ وہ نہیں جی ہاں کر کریں کرنا کرنی کرو دیں دے دو لیے ساتھ ہی
    """.split())
# English words as whisper-urdu writes them when it takes English speech for
# Urdu, measured: "کین یو شو مائی ٹرانزیکشنز", "فریز دہ کارڈ".
_ENGLISH_IN_URDU_LETTERS = frozenset(
    "دہ مائی یو ٹو فور فار فر ای ایز آف اینڈ اٹ دیٹ وٹ واٹ ہاؤ پلیز کین ڈو یور".split()
)
_WORD = re.compile(r"[^\W\d_]+")


def _weigh(en_text: str, ur: Transcript) -> tuple[int, int, str]:
    """Evidence for English and for Urdu from both transcripts, and what it was."""
    english = urdu = 0
    notes = []
    if ur.language == "ur":
        urdu += 1
        notes.append("whisper said Urdu")
    elif ur.language == "en":
        english += 1
        notes.append("whisper said English")
    share = urdu_share(ur.text)
    if share is not None and share < 0.5:
        english += 2
        notes.append("whisper wrote English letters")
    elif share is not None:
        words = [w for w in _WORD.findall(ur.text) if _ARABIC.search(w)]
        grammar = sum(w in _URDU_GRAMMAR for w in words)
        spelled = sum(w in _ENGLISH_IN_URDU_LETTERS for w in words)
        urdu += min(grammar, 3)
        english += min(spelled, 3)
        if not grammar and len(words) >= 3:
            english += 1
        notes.append(f"{grammar} Urdu grammar words, {spelled} English in Urdu letters")
    if has_devanagari(en_text):
        urdu += 2
        notes.append("qwen3-asr wrote Hindi script")
    else:
        latin = _LATIN_WORD.findall(en_text.lower())
        roman = sum(word in _ROMAN_URDU for word in latin)
        if roman >= 2 and roman * 3 >= len(latin):
            urdu += 1
            notes.append("qwen3-asr wrote Roman Urdu")
        elif not roman and len(latin) >= 3:
            # Weak: told English, qwen3-asr sometimes translates Urdu speech
            # into fluent English ("I need to talk to someone").
            english += 1
            notes.append("qwen3-asr wrote English")
    return english, urdu, "; ".join(notes)


def choose(en: Transcript | None, ur: Transcript | None, fallback: Language) -> Decision:
    """The language and transcript, from whatever the recognisers returned.

    whisper-urdu's own verdict is not enough by itself. It called English
    spoken with a Pakistani accent -- and some spoken without one -- Urdu, 12
    of 30 English clips measured, and then wrote the English out in Urdu
    letters or in English ones. So when both recognisers heard something,
    the evidence of both transcripts is weighed, and whisper's verdict
    settles a tie.
    """
    heard = {t.endpoint: t for t in (en, ur) if t is not None}
    en_text = en.text if en else ""
    ur_text = ur.text if ur else ""
    share = urdu_share(ur_text) if ur_text else None
    language: Language

    if en_text and ur_text and ur is not None:
        english, urdu, why = _weigh(en_text, ur)
        if english > urdu:
            return Decision("en", en_text, f"English, {english} to {urdu}: {why}", heard)
        if urdu > english:
            return Decision("ur", ur_text, f"Urdu, {urdu} to {english}: {why}", heard)
        if ur.language in ("en", "ur"):
            language = "ur" if ur.language == "ur" else "en"
        else:
            language = "ur" if (share or 0) >= 0.5 else "en"
        text = ur_text if language == "ur" else en_text
        return Decision(language, text, f"even, {urdu} each, so whisper's verdict: {why}", heard)
    if has_devanagari(en_text):
        return Decision("ur", en_text, "qwen3-asr wrote Hindi script; no Urdu transcript", heard)
    if en_text:
        return Decision("en", en_text, "only qwen3-asr heard anything: English transcript", heard)
    if ur_text:
        # qwen3-asr heard nothing, or failed; whisper's text is all there is.
        language = "ur" if (share or 0) >= 0.5 else "en"
        return Decision(language, ur_text, "only whisper heard anything", heard)
    return Decision(fallback, "", "neither recogniser heard anything", heard)


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
