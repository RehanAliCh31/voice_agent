"""What a reply must be before it is spoken: in one language, and in the bot's own gender.

Two checks, for the web bot and the hands-free assistant alike.

**Language.** The conversation's language is decided from what the customer
said (voicebot.assistant.language), never from what the model wrote. A reply
is held to it: an English reply has no Urdu or Hindi letters in it, and an
Urdu reply no English words -- the Urdu voice reads them with an English it
does not have, and the customer hears two languages in one sentence. Names
the bank's own records hold in English letters ("Netflix", a payee) are the
exception, and so are a few letters every Urdu speaker says as letters: PIN,
ATM. A reply that fails is written again once, then replaced by one of the
bot's own sentences (`reply_failed`).

**Gender.** Urdu verbs agree with the speaker: "میں دیکھ رہی ہوں" is a woman
speaking, "میں دیکھ رہا ہوں" a man. The model is told which the bot is, and
mostly obeys; where it slips, the first-person forms below are put right
before the reply is spoken, so the voice and the words never disagree. Only
forms that can only be the bot speaking of itself are touched: "ٹرانسفر نہیں
ہو سکا" (the transfer could not happen) is about the transfer and stays.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from voicebot.profile import Gender

_ARABIC_LETTER = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")

# Said as letters in Urdu too, and read that way by the Urdu voice.
LETTERS_SAID_IN_URDU = frozenset({"pin", "atm", "otp", "sms", "pkr", "usd", "id"})

# The bank's own everyday words. Its records hold them in English, but Urdu
# has words for them (چیکنگ، سیونگز، کرایہ، بل), and an Urdu reply says those.
BANK_WORDS = frozenset(
    "checking savings account accounts card cards debit credit bill bills balance "
    "rent groceries electricity water phone internet transfer payment deposit".split()
)


def latin_words(values: Iterable[Any]) -> frozenset[str]:
    """The names in English letters in some values -- a turn's tool results, say.

    What the bank's records call a payee, a merchant or a subscription
    ("sarah", "Netflix", "spotify") may be said as the records hold it; its
    everyday words (BANK_WORDS) may not. Measured: asked for this month's
    bills in Urdu, the model kept "Spotify" and "Netflix" in English letters
    even when asked again, and the customer heard no answer at all. JSON
    text, as tool results are kept in the history, is read for its values.
    """
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, str):
            if value[:1] in "{[":
                try:
                    walk(json.loads(value))
                    return
                except ValueError:
                    pass
            found.update(w.lower() for w in _LATIN_WORD.findall(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    for value in values:
        walk(value)
    return frozenset(found - BANK_WORDS)


def language_problems(text: str, language: str, allowed: Iterable[str] = ()) -> list[str]:
    """Why `text` is not a reply in `language`; empty when it is one.

    `allowed` is words in English letters an Urdu reply may hold: the names
    in this turn's tool results. The problems say what kind of slip it was,
    never which words, so they are safe to log.
    """
    problems = []
    if _DEVANAGARI.search(text):
        problems.append("Hindi script")
    if language == "ur":
        permitted = LETTERS_SAID_IN_URDU | {w.lower() for w in allowed}
        stray = [w for w in _LATIN_WORD.findall(text) if w.lower() not in permitted]
        if stray:
            problems.append(f"{len(stray)} word(s) in English letters in an Urdu reply")
        if not _ARABIC_LETTER.search(text) and any(c.isalpha() for c in text):
            problems.append("no Urdu script at all")
    elif _ARABIC_LETTER.search(text):
        problems.append("Urdu script in an English reply")
    return problems


# --- how the bot speaks of itself in Urdu ---------------------------------------

# Verb stems that take an agent: "I could not do / understand / hear ...".
# Never "ہو" (happen) or "مل" (be found): their subject is a thing.
_AGENT = r"(?:کر|سمجھ|سن|دیکھ|بتا|ڈھونڈ|پہچان|پڑھ|جان)"

# The words that carry the gender, masculine to feminine. Participles in -تا
# (کرتا، سکتا، چاہتا) turn into -تی by rule; these do not.
_TO_FEMININE = {
    "رہا": "رہی",
    "سکا": "سکی",
    "چکا": "چکی",
    "گیا": "گئی",
    "ہوا": "ہوئی",
    "آیا": "آئی",
    "گا": "گی",
}
_TO_MASCULINE = {v: k for k, v in _TO_FEMININE.items()}


def _patterns(speaking_as: Gender) -> list[re.Pattern[str]]:
    """Where the *other* gender's first-person forms are; the word itself is group `w`.

    Each needs a first-person sign -- "ہوں" (I am), a future in -وں, or an
    agent verb with "can" -- so a third person is never touched.
    """
    masculine = speaking_as == "female"  # a woman's slip is a masculine form, and back
    wrong = list(_TO_FEMININE if masculine else _TO_MASCULINE)
    future = "گا" if masculine else "گی"
    past = "|".join(w for w in wrong if w != future)
    participle = r"\w+تا" if masculine else r"\w+تی"
    can = "سکتا|سکا" if masculine else "سکتی|سکی"
    progressive = "رہا" if masculine else "رہی"
    return [
        # کرتا ہوں، سکتا ہوں، چاہتا ہوں، بتاتا چلوں
        re.compile(rf"\b(?P<w>{participle})(?=\s+(?:ہوں|چلوں)\b)"),
        # رہا ہوں، سکا ہوں، چکا ہوں، گیا ہوں
        re.compile(rf"\b(?P<w>{past})(?=\s+ہوں\b)"),
        # کروں گا، دوں گا، ہوں گا
        re.compile(rf"\b\w+(?:وں|ؤں)\s+(?P<w>{future})\b"),
        # نہیں کر سکا، نہیں کر سکتا، سمجھ نہیں سکا
        re.compile(rf"\b(?:نہیں\s+{_AGENT}|{_AGENT}\s+نہیں)\s+(?P<w>{can})\b"),
        # کر پا رہا: the negative progressive drops its ہوں
        re.compile(rf"\b{_AGENT}\s+پا\s+(?P<w>{progressive})\b"),
    ]


def _as(word: str, gender: Gender) -> str:
    """One gendered word, in `gender`."""
    table = _TO_FEMININE if gender == "female" else _TO_MASCULINE
    if word in table:
        return table[word]
    stem = word[:-2]
    return stem + ("تی" if gender == "female" else "تا")


def wrong_gender_forms(text: str, gender: Gender) -> int:
    """How many first-person forms in `text` belong to the other gender."""
    return sum(len(pattern.findall(text)) for pattern in _patterns(gender))


def in_own_gender(text: str, gender: Gender) -> tuple[str, int]:
    """`text` with the bot's first-person forms in `gender`, and how many were changed."""

    def swap(m: re.Match[str]) -> str:
        whole, start, end = m.group(0), m.start("w") - m.start(), m.end("w") - m.start()
        return whole[:start] + _as(m.group("w"), gender) + whole[end:]

    changed = 0
    for pattern in _patterns(gender):
        text, n = pattern.subn(swap, text)
        changed += n
    return text, changed
